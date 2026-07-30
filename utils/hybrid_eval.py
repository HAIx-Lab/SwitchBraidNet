"""
Hybrid Brain-Computer Interface (hBCI) evaluation pipeline.

This module evaluates trained models across:
  - **Single paradigms**: MI, SSVEP, ERP (standard accuracy, F1, Kappa, ITR).
  - **Hybrid paradigm pairs**: All ordered pairs for sequential mode, and
    unique unordered pairs for simultaneous mode.
  - **Per-subject and pooled evaluation**: Each subject is evaluated
    individually, plus a pooled "All" evaluation using all test data.

Hybrid BCI Modes:
  - **Sequential (Seq)**: Paradigms are executed one after another.
    Latency = sum of individual trial times.
  - **Simultaneous (Sim)**: Paradigms are executed in parallel.
    Latency = max of individual trial times.

Hybrid metrics are derived from the Kronecker product of individual confusion
matrices, enabling computation of joint accuracy, F1, precision, recall,
Kappa, and ITR for the combined decision space.

Usage:
    This module is called by ``evaluate_models.py`` — it is not intended to
    be run directly.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from joblib import Parallel, delayed

from utils.data import HybridDataset, collect_dataset_paths, DEFAULT_LABEL_KEYS, group_paths_by_subject
from utils.helpers import project_root, apply_default_env, resolve_model_dir
from utils.metrics import itr_bits_per_min, model_size_mb
from utils.qat import QATModelWrapper


@dataclass
class HBciConfig:
    """Configuration for hybrid BCI evaluation.

    Attributes:
        batch_size: Batch size for inference data loaders.
        epoch_time: Dict mapping paradigm → trial duration in seconds.
        n_classes: Dict mapping paradigm → number of output classes.
        all_modalities: List of paradigms to evaluate.
        channels_by_mode: Dict mapping paradigm → number of EEG channels.
        label_key_by_mode: Dict mapping paradigm → label key in ``.npz`` files.
        num_samples: Number of time samples per trial.
        max_folds: Maximum number of folds to evaluate (``None`` for all).
        k_folds: Total number of cross-validation folds.
        dataset_path: Relative path from project root to the dataset.
    """

    batch_size: int = 32
    epoch_time: Dict[str, float] = field(default_factory=lambda: {"mi": 1.0, "ssvep": 1.0, "erp": 1.0})
    n_classes: Dict[str, int] = field(default_factory=lambda: {"mi": 2, "ssvep": 4, "erp": 2})
    all_modalities: List[str] = field(default_factory=lambda: ["mi", "ssvep", "erp"])
    channels_by_mode: Dict[str, int] = field(default_factory=lambda: {"mi": 20, "ssvep": 10, "erp": 28})
    label_key_by_mode: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_LABEL_KEYS))
    num_samples: int = 256
    max_folds: Optional[int] = None
    k_folds: int = 5
    dataset_path: str = "../Dataset/openbmi/"


def load_data_paths(cfg: HBciConfig, test_limit=None) -> Dict[str, Dict[str, List[str]]]:
    """Load test data paths grouped by subject for all modalities.

    Args:
        cfg: Evaluation configuration.
        test_limit: Optional limit on the number of test files.

    Returns:
        Dict mapping subject_id → {modality → list of file paths}.
        Includes a special ``"All"`` key containing all test data.
    """
    print("Loading Data Paths", end="", flush=True)
    subject_paths: Dict[str, Dict[str, List[str]]] = {}
    dataset_root = project_root() / cfg.dataset_path

    for modality in cfg.all_modalities:
        _, test_paths = collect_dataset_paths(modality, root=dataset_root)
        if test_limit is not None:
            if isinstance(test_limit, float) and test_limit < 1.0:
                n = int(len(test_paths) * test_limit)
                test_paths = test_paths[:max(1, n)]
            elif isinstance(test_limit, int):
                test_paths = test_paths[:test_limit]

        groups = {"All": test_paths}
        groups.update(group_paths_by_subject(test_paths))

        for subj_id, paths in groups.items():
            if subj_id not in subject_paths:
                subject_paths[subj_id] = {}
            subject_paths[subj_id][modality] = paths

    print(" - Done", flush=True)
    return subject_paths


def load_models_for_fold(
    model_class,
    model_name: str,
    bits: int,
    device: torch.device,
    cfg: HBciConfig,
    fold: int,
    optimizer_builder=None,
) -> Tuple[Dict[str, torch.nn.Module], Dict[str, float]]:
    """Load trained model checkpoints for all modalities in a given fold.

    Args:
        model_class: The model class to instantiate.
        model_name: Model name (used to locate checkpoint directories).
        bits: Quantization bit-width.
        device: Target device for the loaded models.
        cfg: Evaluation configuration.
        fold: Fold number (1-indexed).
        optimizer_builder: Optional optimizer builder for checkpoint loading.

    Returns:
        Tuple of (models_dict, sizes_dict) where:
          - models_dict maps modality → loaded model (in eval mode).
          - sizes_dict maps modality → estimated model size in MB.

    Raises:
        FileNotFoundError: If checkpoint directory or file is missing.
    """
    specs = {
        mode: {"num_channels": cfg.channels_by_mode.get(mode, 10)}
        for mode in cfg.all_modalities
    }
    model_dir = resolve_model_dir(model_name)
    logs_dir = model_dir / "logs"
    models: Dict[str, torch.nn.Module] = {}
    sizes_mb_dict: Dict[str, float] = {}

    bit_val = 32
    if bits in ("FP16", 16, "16"): bit_val = 16
    if bits in ("FP8", 8, "8"): bit_val = 8
    if bits in ("FP4", "INT4", 4, "4"): bit_val = 4

    for mode, spec in specs.items():
        # Load the pooled "All" model checkpoint
        ckpt_folder = logs_dir / f"{model_name}_{mode}_{bits}_All_f{fold}"
        if not ckpt_folder.exists():
            raise FileNotFoundError(f"Missing checkpoint folder for {mode} in {ckpt_folder}")

        ckpt_files = list(ckpt_folder.glob("version_*/checkpoints/*.ckpt"))
        if not ckpt_files:
            raise FileNotFoundError(f"Missing checkpoint for {mode} in {ckpt_folder}")

        ckpt = ckpt_files[0]

        m = QATModelWrapper.load_from_checkpoint(
            checkpoint_path=str(ckpt),
            model_class=model_class,
            mode=mode,
            bits=bits,
            num_channels=spec["num_channels"],
            num_samples=cfg.num_samples,
            n_classes_by_mode=cfg.n_classes,
            optimizer_builder=optimizer_builder,
            weights_only=True,
        ).to(device)
        m.eval()
        models[mode] = m

        sizes_mb_dict[mode] = model_size_mb(m) * (bit_val / 32.0)

    return models, sizes_mb_dict


@torch.inference_mode()
def collect_preds(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference and collect predictions and labels.

    Args:
        model: Trained model in eval mode.
        loader: DataLoader yielding (input, label) batches.
        device: Device to run inference on.

    Returns:
        Tuple of (predictions, labels) as numpy int32 arrays.
    """
    all_preds, all_labels = [], []
    for x, y in loader:
        p = model(x.to(device)).argmax(dim=-1)
        all_preds.append(p.cpu())
        all_labels.append(y)

    return torch.cat(all_preds).numpy().astype(np.int32), torch.cat(all_labels).numpy().astype(np.int32)


def eval_single(modality: str, preds: np.ndarray, labels: np.ndarray, cfg: HBciConfig) -> Dict:
    """Evaluate a single-paradigm classification result.

    Computes accuracy, macro F1, precision, recall, Cohen's Kappa, and ITR.

    Args:
        modality: BCI paradigm name.
        preds: Predicted class labels.
        labels: Ground-truth class labels.
        cfg: Evaluation configuration.

    Returns:
        Dict with keys: n_classes, test_acc, f1, precision, recall, kappa,
        itr, latency_ms.
    """
    correct = preds == labels
    acc = float(correct.mean())
    f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    prec = float(precision_score(labels, preds, average="macro", zero_division=0))
    rec = float(recall_score(labels, preds, average="macro", zero_division=0))
    kappa = float(cohen_kappa_score(labels, preds)) if len(np.unique(labels)) > 1 else 0.0
    n_cls = cfg.n_classes[modality]
    lat = cfg.epoch_time[modality]
    itr = itr_bits_per_min(n_cls, acc, lat)
    return {
        "n_classes": n_cls,
        "test_acc": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "kappa": kappa,
        "itr": itr,
        "latency_ms": lat * 1000.0,
    }


def calculate_hybrid_metrics(
    p1: np.ndarray, l1: np.ndarray, p2: np.ndarray, l2: np.ndarray, n_cls1: int, n_cls2: int
) -> Tuple[float, float, float, float]:
    """Compute hybrid BCI metrics from two paradigm predictions.

    Constructs the joint confusion matrix via Kronecker product of individual
    confusion matrices, then derives macro F1, precision, recall, and Kappa.

    Args:
        p1, l1: Predictions and labels for paradigm 1.
        p2, l2: Predictions and labels for paradigm 2.
        n_cls1, n_cls2: Number of classes for each paradigm.

    Returns:
        Tuple of (macro_f1, macro_precision, macro_recall, kappa).
    """
    from sklearn.metrics import confusion_matrix

    cm1 = confusion_matrix(l1, p1, labels=np.arange(n_cls1)).astype(np.int64)
    cm2 = confusion_matrix(l2, p2, labels=np.arange(n_cls2)).astype(np.int64)
    cm_hybrid = np.kron(cm1, cm2)

    TP = np.diag(cm_hybrid)
    FP = cm_hybrid.sum(axis=0) - TP
    FN = cm_hybrid.sum(axis=1) - TP

    prec = np.divide(TP, TP + FP, out=np.zeros_like(TP, dtype=float), where=(TP + FP) != 0)
    rec = np.divide(TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec, dtype=float), where=(prec + rec) != 0)

    macro_f1 = f1.mean()
    macro_prec = prec.mean()
    macro_rec = rec.mean()

    N = cm_hybrid.sum()
    if N == 0:
        kappa = 0.0
    else:
        p_o = TP.sum() / float(N)
        marg_cols = cm_hybrid.sum(axis=0, dtype=np.float64)
        marg_rows = cm_hybrid.sum(axis=1, dtype=np.float64)
        p_e = np.sum((marg_cols / N) * (marg_rows / N))
        kappa = 0.0 if (1.0 - p_e == 0) else (p_o - p_e) / (1.0 - p_e)

    return float(macro_f1), float(macro_prec), float(macro_rec), float(kappa)


def eval_hybrid_cartesian(
    combo: List[str], preds_dict: Dict[str, np.ndarray], labels_dict: Dict[str, np.ndarray], cfg: HBciConfig, mode: str = "Seq"
) -> Dict:
    """Evaluate a hybrid paradigm pair using Cartesian product semantics.

    For **Sequential** mode, latency is the sum of individual trial times.
    For **Simultaneous** mode, latency is the max of individual trial times.

    Args:
        combo: List of two paradigm names (e.g., ``["mi", "ssvep"]``).
        preds_dict: Dict mapping paradigm → predicted labels.
        labels_dict: Dict mapping paradigm → ground-truth labels.
        cfg: Evaluation configuration.
        mode: ``"Seq"`` for sequential or ``"Sim"`` for simultaneous.

    Returns:
        Dict with keys: n_classes, test_acc, itr, f1, precision, recall,
        kappa, latency_ms.
    """
    m1, m2 = combo[0], combo[1]
    p1, l1 = preds_dict[m1], labels_dict[m1]
    p2, l2 = preds_dict[m2], labels_dict[m2]

    n1, n2 = len(p1), len(p2)
    total_combinations = n1 * n2

    correct1 = p1 == l1
    correct2 = p2 == l2

    if mode == "Seq":
        n_completed = int(correct1.sum()) * int(correct2.sum())
        total_round_time = n_completed * (cfg.epoch_time[m1] + cfg.epoch_time[m2])
    else:
        n_completed = int(correct1.sum()) * int(correct2.sum())
        total_round_time = n_completed * max(cfg.epoch_time[m1], cfg.epoch_time[m2])

    acc = n_completed / total_combinations
    n_cls = cfg.n_classes[m1] * cfg.n_classes[m2]
    mean_lat = (total_round_time / n_completed) if n_completed > 0 else (cfg.epoch_time[m1] + cfg.epoch_time[m2])

    itr = itr_bits_per_min(n_cls, acc, mean_lat)
    f1, prec, rec, kappa = calculate_hybrid_metrics(p1, l1, p2, l2, cfg.n_classes[m1], cfg.n_classes[m2])

    return {
        "n_classes": n_cls,
        "test_acc": acc,
        "itr": itr,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "kappa": kappa,
        "latency_ms": mean_lat * 1000.0,
    }


def _evaluate_subject(
    subj_id: str,
    subj_paths_dict: Dict[str, List[str]],
    model_class,
    model_name: str,
    bits: int,
    cfg: HBciConfig,
    num_folds: int,
    optimizer_builder,
    ordered_pairs,
    sim_pairs,
) -> List[Dict]:
    """Evaluate a single subject (or pooled "All") across all folds.

    This function is designed to run in parallel via ``joblib``.

    Args:
        subj_id: Subject identifier (or ``"All"`` for pooled).
        subj_paths_dict: Dict mapping modality → list of test file paths.
        model_class: Model class for checkpoint loading.
        model_name: Model name.
        bits: Quantization bit-width.
        cfg: Evaluation configuration.
        num_folds: Number of folds to evaluate.
        optimizer_builder: Optimizer builder for checkpoint loading.
        ordered_pairs: List of (m1, m2) pairs for sequential hybrid eval.
        sim_pairs: List of (m1, m2) pairs for simultaneous hybrid eval.

    Returns:
        List of result dictionaries.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # Construct dataloaders
    loaders = {}
    for modality, paths in subj_paths_dict.items():
        loaders[modality] = DataLoader(
            HybridDataset(np.array(paths), mode=modality, label_key_by_mode=cfg.label_key_by_mode),
            batch_size=cfg.batch_size * 8,
            shuffle=False,
            num_workers=0,  # 0 workers inside a parallel process
            pin_memory=torch.cuda.is_available(),
        )

    results = []
    for fold in range(1, num_folds + 1):
        try:
            models, sizes_mb_dict = load_models_for_fold(
                model_class, model_name, bits, device, cfg, fold, optimizer_builder
            )
        except Exception as e:
            print(f"[WARN] Skipping Fold {fold} for {model_name} ({bits}) Subj {subj_id}: {e}")
            continue

        preds_cache: Dict[str, np.ndarray] = {}
        labels_cache: Dict[str, np.ndarray] = {}
        for modality in cfg.all_modalities:
            if modality in loaders:
                preds, labels = collect_preds(models[modality], loaders[modality], device)
                preds_cache[modality] = preds
                labels_cache[modality] = labels

        # Single Modalities
        for modality in cfg.all_modalities:
            if modality not in preds_cache: continue
            m = eval_single(modality, preds_cache[modality], labels_cache[modality], cfg)
            results.append({
                "model": model_name, "bits": bits, "subject": subj_id, "fold": fold,
                "combination": modality, "mode": "-", "size_mb": sizes_mb_dict[modality],
                **{k: round(v, 6) for k, v in m.items()}
            })

        # Sequential Hybrid
        for pair in ordered_pairs:
            combo = list(pair)
            if combo[0] not in preds_cache or combo[1] not in preds_cache: continue
            combo_name = "+".join(combo)
            res_seq = eval_hybrid_cartesian(combo, preds_cache, labels_cache, cfg, mode="Seq")
            size_hybrid = sizes_mb_dict[combo[0]] + sizes_mb_dict[combo[1]]
            results.append({
                "model": model_name, "bits": bits, "subject": subj_id, "fold": fold,
                "combination": combo_name, "mode": "Seq", "size_mb": size_hybrid,
                **{k: round(v, 6) for k, v in res_seq.items()}
            })

        # Simultaneous Hybrid
        for pair in sim_pairs:
            combo = list(pair)
            if combo[0] not in preds_cache or combo[1] not in preds_cache: continue
            combo_name = "+".join(combo)
            res_sim = eval_hybrid_cartesian(combo, preds_cache, labels_cache, cfg, mode="Sim")
            size_hybrid = sizes_mb_dict[combo[0]] + sizes_mb_dict[combo[1]]
            results.append({
                "model": model_name, "bits": bits, "subject": subj_id, "fold": fold,
                "combination": combo_name, "mode": "Sim", "size_mb": size_hybrid,
                **{k: round(v, 6) for k, v in res_sim.items()}
            })

        for m in models.values():
            del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def run_hbci_eval(
    spec,
    bits,
    config_dict: dict,
) -> List[Dict]:
    """Run the full hybrid BCI evaluation for a model at a given bit-width.

    Evaluates all subjects (individual + pooled) across all folds, for both
    single and hybrid paradigm combinations.

    Args:
        spec: A ``ModelSpec`` instance from the registry.
        bits: Quantization bit-width.
        config_dict: Experiment configuration dictionary (from ``config.py``).

    Returns:
        List of result dictionaries (one per subject × fold × paradigm combo).
    """
    apply_default_env()

    cfg = HBciConfig(
        batch_size=config_dict.get("batch_size", 32),
        epoch_time={"mi": 1.0, "ssvep": 1.0, "erp": 1.0},
        n_classes=config_dict.get("n_classes", {"mi": 2, "ssvep": 4, "erp": 2}),
        all_modalities=config_dict.get("modes", ["mi", "ssvep", "erp"]),
        channels_by_mode=config_dict.get("channels", {"mi": 20, "ssvep": 10, "erp": 28}),
        label_key_by_mode=DEFAULT_LABEL_KEYS,
        num_samples=config_dict.get("num_samples_time", 256),
        k_folds=config_dict.get("k_folds", 5),
        max_folds=config_dict.get("max_folds", 1),
        dataset_path=config_dict.get("dataset_path", "../Dataset/openbmi/")
    )

    model_name = spec.name
    model_class = spec.model_class
    optimizer_builder = spec.optimizer_builder

    print(f"\n[EVAL] Model: {model_name} | Precision: {bits}")

    test_limit = 100 if config_dict.get("debug", False) else config_dict.get("test_data_limit", 1.0)
    subject_paths_dict = load_data_paths(cfg, test_limit=test_limit)

    ordered_pairs = [(a, b) for a in cfg.all_modalities for b in cfg.all_modalities]
    sim_pairs = []
    for idx, a in enumerate(cfg.all_modalities):
        for b in cfg.all_modalities[idx + 1:]:
            sim_pairs.append((a, b))

    num_folds = cfg.max_folds if cfg.max_folds is not None else cfg.k_folds

    n_workers = config_dict.get("max_concurrent_eval_workers", 1)

    # Run evaluation across subjects in parallel
    print(f"Distributing evaluation across {n_workers} concurrent workers...")

    parallel_outputs = Parallel(n_jobs=n_workers, backend="loky")(
        delayed(_evaluate_subject)(
            subj_id, subj_paths, model_class, model_name, bits, cfg, num_folds, optimizer_builder, ordered_pairs, sim_pairs
        )
        for subj_id, subj_paths in subject_paths_dict.items()
    )

    results = []
    for r in parallel_outputs:
        results.extend(r)

    return results
