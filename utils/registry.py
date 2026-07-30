"""
Model registry for the experiment pipeline.

Provides a centralized registry of model specifications (``ModelSpec``) that
encapsulate model class, channel configurations, optimizer builders, and any
extra training hooks. The :func:`get_model_spec` function resolves a model
by name for use in the training and evaluation pipelines.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Type

import torch.nn as nn

from models.switchbraidnet import SwitchBraidNet
from .data import DEFAULT_CHANNELS
from .optimizers import switchbraid_optimizer_builder


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a registered model.

    Attributes:
        name: Human-readable model name (used in configs and logs).
        model_class: The ``nn.Module`` subclass to instantiate.
        channels_by_mode: Dict mapping BCI paradigm → number of EEG channels.
        optimizer_builder: Optional callable returning optimizer (or PL dict).
        extra_trainer_fit: Optional post-training hook (e.g., fine-tuning step).
    """

    name: str
    model_class: Type[nn.Module]
    channels_by_mode: Dict[str, int]
    optimizer_builder: Optional[Callable] = None
    extra_trainer_fit: Optional[Callable] = None


def _switchbraid_extra_fit(trainer, model, val_loader):
    """Extra fine-tuning step for SwitchBraidNet after initial training."""
    trainer.fit_loop.max_epochs = trainer.current_epoch + 2
    trainer.fit(model, val_loader, val_loader)


MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(
        "SwitchBraidNet",
        SwitchBraidNet,
        dict(DEFAULT_CHANNELS),
        optimizer_builder=switchbraid_optimizer_builder,
        extra_trainer_fit=_switchbraid_extra_fit,
    ),
]


def get_model_spec(name: str) -> ModelSpec:
    """Look up a model specification by name (case-insensitive).

    Args:
        name: Model name to look up.

    Returns:
        ModelSpec: The matching model specification.

    Raises:
        KeyError: If no model with the given name is registered.
    """
    for spec in MODEL_SPECS:
        if spec.name.lower() == name.lower():
            return spec
    available = ", ".join(spec.name for spec in MODEL_SPECS)
    raise KeyError(f"Unknown model: {name}. Available: {available}")
