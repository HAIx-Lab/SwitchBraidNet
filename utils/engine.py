"""
Training engine with k-fold cross-validation.

Orchestrates the full training loop for a given model specification:
  1. Loads and splits data using Stratified K-Fold.
  2. Trains a pooled ("All subjects") model per fold using PyTorch Lightning.
  3. Applies Quantization-Aware Training via the QATModelWrapper.
  4. Saves checkpoints and training curve plots.

Usage:
    This module is called by ``train_models.py`` — it is not intended to be
    run directly.
"""

import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from pathlib import Path

from utils.data import (
    HybridDataset,
    collect_dataset_paths,
    load_labels,
    group_paths_by_subject,
)
from utils.helpers import (
    project_root,
    apply_default_env,
    resolve_model_dir,
    save_fold_plots,
)
from utils.qat import QATModelWrapper
from utils.metrics import compute_itr, compute_kappa, model_size_mb


def get_limited_paths(paths, limit):
    """Limit the number of data paths for quick testing.

    Args:
        paths (list): List of file paths.
        limit: ``None`` or ``1.0`` to use all; a float < 1.0 for a fraction;
            an int for an absolute count.

    Returns:
        list: Truncated list of paths (at least 1 element).
    """
    if limit is None:
        return paths
    if isinstance(limit, float):
        if limit >= 1.0:
            return paths
        n = int(len(paths) * limit)
    elif isinstance(limit, int):
        n = limit
    else:
        return paths
    return paths[:max(1, n)]


def train_model(spec, mode, bits, config):
    """Train a model for a given paradigm and bit-width using k-fold CV.

    This function:
      1. Resolves dataset paths and creates stratified folds.
      2. For each fold, trains a pooled model across all subjects.
      3. Saves the best checkpoint (by validation loss) and training curves.

    Args:
        spec: A ``ModelSpec`` instance from the registry.
        mode (str): BCI paradigm (``"mi"``, ``"ssvep"``, or ``"erp"``).
        bits (int): Quantization bit-width (32, 16, or 8).
        config (dict): Experiment configuration dictionary (from ``config.py``).
    """
    apply_default_env()

    model_name = spec.name
    model_class = spec.model_class
    optimizer_builder = spec.optimizer_builder
    extra_trainer_fit = spec.extra_trainer_fit

    model_dir = resolve_model_dir(model_name)
    logs_dir = model_dir / config["logs_dir_name"]
    plots_dir = model_dir / "plots"
    logs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Handle Debug Mode
    if config.get("debug", False):
        print(f"[DEBUG MODE] Overriding epochs to 2 and samples to 100")
        epochs = 2
        train_limit = 100
    else:
        epochs = config["epochs"]
        train_limit = config["train_data_limit"]

    # Resolve Dataset Path
    dataset_root = project_root() / config["dataset_path"]

    # Load Data
    train_paths, _ = collect_dataset_paths(mode, root=dataset_root)
    train_paths = get_limited_paths(train_paths, train_limit)

    skf = StratifiedKFold(n_splits=config["k_folds"], shuffle=True, random_state=config["seed"])

    n_classes_map = config["n_classes"]
    n_classes = n_classes_map.get(mode, 2)
    n_channels = config["channels"].get(mode, 32)

    # Extract labels and paths for the pooled dataset
    all_paths = np.array(train_paths)
    all_labels = np.array([load_labels(f, mode) for f in all_paths])

    splits = list(skf.split(np.zeros(len(all_labels)), all_labels))

    for fold, (t_idx, v_idx) in enumerate(splits):
        if config["max_folds"] is not None and fold >= config["max_folds"]:
            break

        print(f"\n{'='*50}\nStarting Fold {fold+1}\n{'='*50}")

        # Train pooled ("All") model
        subj_paths = all_paths

        train_loader = DataLoader(
            HybridDataset(subj_paths[t_idx], mode=mode),
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            HybridDataset(subj_paths[v_idx], mode=mode),
            batch_size=config["batch_size"],
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

        model = QATModelWrapper(
            model_class=model_class,
            mode=mode,
            bits=bits,
            num_channels=n_channels,
            num_samples=config["num_samples_time"],
            n_classes_by_mode=n_classes_map,
            optimizer_builder=optimizer_builder,
        )

        # Checkpoint name includes _All to indicate pooled training
        ckpt_name_all = f"{model_name}_{mode}_{bits}_All_f{fold + 1}"
        callbacks = [
            ModelCheckpoint(monitor="val_loss", filename=ckpt_name_all, mode="min", save_top_k=1, save_weights_only=True),
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            TQDMProgressBar()
        ]

        logger = CSVLogger(save_dir=str(logs_dir), name=ckpt_name_all)

        current_precision = 16 if bits == 16 else 32

        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="auto",
            logger=logger,
            callbacks=callbacks,
            precision=current_precision,
            enable_model_summary=False,
        )

        print(f"\n[TRAIN] Model: {model_name} | Mode: {mode} | Bits: {bits} | Subj: Pooled (All) | Fold: {fold+1}")
        trainer.fit(model, train_loader, val_loader)

        if extra_trainer_fit:
            extra_trainer_fit(trainer, model, val_loader)

        save_fold_plots(trainer, plots_dir, model_name, mode, bits, fold + 1)

        best_all_ckpt = trainer.checkpoint_callback.best_model_path
        print(f"[DONE] Saved pooled checkpoint at: {best_all_ckpt}")

        # Cleanup
        del model, trainer, train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
