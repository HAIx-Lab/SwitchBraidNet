"""
General-purpose project utilities.

Provides path resolution, environment configuration, model directory
management, and training curve visualization helpers.
"""

import os
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
from tqdm import tqdm


def project_root() -> Path:
    """Return the root directory of the project (one level above this file)."""
    return Path(__file__).resolve().parents[1]


def apply_default_env():
    """Apply default environment variables for performance and reproducibility."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    torch.set_float32_matmul_precision("high")


def resolve_model_dir(model_name: str, model_dir: Optional[Path] = None) -> Path:
    """Resolve the directory where model artifacts (logs, checkpoints) are stored.

    Args:
        model_name: Name of the model (used as subdirectory name).
        model_dir: Optional explicit path. If provided, returned directly.

    Returns:
        Path to the model artifact directory.
    """
    if model_dir is not None:
        return Path(model_dir)
    return project_root() / model_name


def system_cooldown(seconds: int) -> None:
    """Pause execution to allow hardware (GPU) to cool down.

    Args:
        seconds: Number of seconds to wait. No-op if <= 0.
    """
    if seconds <= 0:
        return
    print(f"\n[INFO] Cooling Down For {seconds // 60} Minutes...")
    for _ in tqdm(range(seconds), desc="Cooling down", unit="s"):
        time.sleep(1)
    print("[INFO] System Cooled. Resuming Experiments.\n")


def log_to_txt(path: Path, content: str, mode: str = "a") -> None:
    """Append content to a text log file.

    Args:
        path: Path to the log file.
        content: String content to write.
        mode: File open mode. Default: ``"a"`` (append).
    """
    with path.open(mode) as f:
        f.write(content + "\n")


def save_fold_plots(
    trainer, plots_dir: Path, model_name: str, mode: str, bits: int, fold: int
) -> None:
    """Generate and save training loss/accuracy curves from CSVLogger metrics.

    Reads the ``metrics.csv`` file produced by PyTorch Lightning's CSVLogger
    and creates a two-panel figure: training loss (left) and validation
    accuracy (right).

    Args:
        trainer: PyTorch Lightning Trainer instance (with CSVLogger).
        plots_dir: Directory to save the plot PDF.
        model_name: Model name for the filename.
        mode: BCI paradigm mode (e.g., ``"mi"``, ``"ssvep"``, ``"erp"``).
        bits: Quantization bit-width.
        fold: Cross-validation fold number.
    """
    metrics_path = Path(trainer.logger.log_dir) / "metrics.csv"
    acc_col = "val_acc"

    if not metrics_path.exists():
        return

    try:
        metrics_df = pd.read_csv(metrics_path)
        train_df = metrics_df.dropna(subset=["train_loss"])
        val_df = metrics_df.dropna(subset=[acc_col])

        fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
        sns.lineplot(data=train_df, x="step", y="train_loss", ax=ax[0], label="Train Loss")
        ax[0].set_title(f"Loss (Fold {fold})")

        if not val_df.empty:
            sns.lineplot(data=val_df, x="step", y=acc_col, ax=ax[1], label="Val Acc")
            ax[1].set_title(f"Accuracy ({mode}, Fold {fold})")

        plt.tight_layout()
        plots_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plots_dir / f"curves_{model_name}_{mode}_{bits}_f{fold}.pdf", format="pdf")
        plt.close()
    except Exception as e:
        print(f"[WARN] Could not save plots for {model_name}: {e}")
