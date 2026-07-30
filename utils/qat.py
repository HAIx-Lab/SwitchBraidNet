"""
Quantization-Aware Training (QAT) wrapper using PyTorch Lightning.

This module provides:
  - ``LowBitFakeQuantize``: A custom fake-quantization module for sub-8-bit
    (e.g., 4-bit) precision simulation during training.
  - ``QATModelWrapper``: A PyTorch Lightning module that wraps any EEG model
    class, applies QAT configuration based on the target bit-width, and
    handles training/validation/test steps with cross-entropy loss and
    accuracy tracking.

Supported bit-widths:
  - **32-bit (FP32)**: No quantization applied.
  - **16-bit (FP16)**: Mixed-precision training via PL Trainer precision flag.
  - **8-bit (INT8)**: Standard QAT using ``fbgemm`` backend.
  - **4-bit / 2-bit**: Custom fake quantization via ``LowBitFakeQuantize``.
"""

import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
import torch.ao.quantization as tq
from torch.ao.quantization import FakeQuantize, MinMaxObserver

from .optimizers import default_optimizer_builder


class LowBitFakeQuantize(FakeQuantize):
    """Fake quantization module for sub-8-bit precision (e.g., 4-bit, 2-bit).

    Simulates low-bit quantization during training by rounding activations
    and weights to the nearest quantization level. This enables
    quantization-aware training for extremely low bit-widths.

    Args:
        bits (int): Target bit-width (e.g., 4 for INT4). Default: 4.
    """

    def __init__(self, bits=4, **kwargs):
        super().__init__(
            observer=MinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False,
        )
        self.bits = bits
        self.levels = 2**bits

    def forward(self, x):
        x = super().forward(x)
        step = 255 / (self.levels - 1)
        return torch.round(x / step) * step


def _build_qat_config(bits):
    """Build a PyTorch quantization config for the specified bit-width.

    Args:
        bits (int): Target bit-width (32, 16, 8, 4, or 2).

    Returns:
        QConfig or None: Quantization config, or ``None`` for FP32/FP16.

    Raises:
        ValueError: For unsupported bit-widths.
    """
    if bits >= 32 or bits == 16:
        return None
    if bits == 8:
        return tq.get_default_qat_qconfig("fbgemm")
    if bits in (2, 4):
        return tq.QConfig(
            activation=LowBitFakeQuantize.with_args(bits=bits),
            weight=LowBitFakeQuantize.with_args(bits=bits),
        )
    raise ValueError(f"Unsupported bit width: {bits}")


class QATModelWrapper(pl.LightningModule):
    """PyTorch Lightning wrapper for Quantization-Aware Training.

    Wraps any EEG model class, applies the appropriate QAT configuration,
    and provides standard training/validation/test step implementations.

    Args:
        model_class: The model class to instantiate (e.g., ``SwitchBraidNet``).
        mode (str): BCI paradigm (``"mi"``, ``"ssvep"``, or ``"erp"``).
        bits (int): Target quantization bit-width.
        num_channels (int): Number of EEG channels. Default: 62.
        num_samples (int): Number of time samples per trial. Default: 256.
        n_classes_by_mode (dict): Mapping from paradigm → number of classes.
        optimizer_builder (callable): Function that builds the optimizer.
            Defaults to :func:`default_optimizer_builder`.
        **model_kwargs: Additional keyword arguments passed to ``model_class``.
    """

    def __init__(
        self,
        model_class,
        mode,
        bits,
        num_channels=62,
        num_samples=256,
        n_classes_by_mode=None,
        optimizer_builder=None,
        **model_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["model_class", "optimizer_builder", "n_classes_by_mode"]
        )
        self.mode = mode
        self.bits = bits
        self._optimizer_builder = optimizer_builder or default_optimizer_builder

        if n_classes_by_mode and mode in n_classes_by_mode:
            n_classes = n_classes_by_mode[mode]
        else:
            n_classes = 2 if mode in ("mi", "erp") else 4
        self.net = model_class(
            num_classes=n_classes,
            num_channels=num_channels,
            num_samples=num_samples,
            **model_kwargs,
        )

        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.loss_fn = nn.CrossEntropyLoss()

        self._prepare_quantization()

    def _prepare_quantization(self):
        """Apply QAT configuration to the wrapped model if applicable."""
        qat_config = _build_qat_config(self.bits)
        if qat_config is None:
            return
        self.net.qconfig = qat_config
        tq.prepare_qat(self.net, inplace=True)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.acc(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.acc(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.acc(logits, y)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("test_acc", acc, prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        return self._optimizer_builder(self)
