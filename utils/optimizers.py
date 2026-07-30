"""
Optimizer and learning rate scheduler builders.

Each builder function takes a PyTorch Lightning module and returns either:
  - A single ``torch.optim.Optimizer``, or
  - A dict with ``"optimizer"`` and ``"lr_scheduler"`` keys (PL convention).
"""

import torch


def default_optimizer_builder(module):
    """Build a default Adam optimizer.

    Args:
        module: PyTorch Lightning module whose parameters will be optimized.

    Returns:
        torch.optim.Adam: Optimizer with learning rate 1e-3.
    """
    return torch.optim.Adam(module.parameters(), lr=1e-3)


def switchbraid_optimizer_builder(module):
    """Build the SwitchBraidNet optimizer with cosine annealing warm restarts.

    Uses Adam with weight decay and a CosineAnnealingWarmRestarts scheduler
    for stable convergence during quantization-aware training.

    Args:
        module: PyTorch Lightning module whose parameters will be optimized.

    Returns:
        dict: PL-compatible dict with ``"optimizer"`` and ``"lr_scheduler"``.
    """
    optimizer = torch.optim.Adam(module.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6,
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        },
    }
