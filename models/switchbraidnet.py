"""
SwitchBraidNet: A lightweight EEG classification architecture for hybrid BCIs.

This model combines dual-scale temporal convolutions with a Squeeze-and-Excitation
(SE) attention mechanism and log-variance feature extraction. The architecture is
designed for efficient deployment on resource-constrained edge devices and is
compatible with Quantization-Aware Training (QAT) at reduced bit-widths (16-bit,
8-bit).

Architecture Overview:
    1. Dual Temporal Convolutions — Two parallel Conv2d layers with different kernel
       sizes (64 and 32 samples) capture both slow and fast EEG dynamics.
    2. SE Attention Block — Channel-wise recalibration using squeeze-and-excitation
       to emphasize informative frequency bands.
    3. Depthwise Spatial Convolution — Learns spatial filters across EEG channels.
    4. 1×1 Pointwise Mixer — Reduces feature dimensionality.
    5. Log-Variance Layer — Computes log-variance over the temporal axis as a
       compact, noise-robust feature representation.
    6. Linear Classifier — Final fully-connected layer for class prediction.

Reference:
    This architecture is proposed in:
    "Quantized Hybrid Brain-Computer Interfaces: Evaluating Quantization-Aware
     Training for Multi-Paradigm EEG Classification on Edge Devices"
    Accepted at IEEE International Conference on Systems, Man, and Cybernetics
    (SMC), 2026.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogVarLayer(nn.Module):
    """Log-variance feature extraction layer.

    Computes the log of the mean squared activation along the temporal
    dimension, followed by batch normalization. This provides a compact,
    noise-robust feature representation commonly used in EEG decoding.

    Args:
        channels (int): Number of input feature channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = torch.mean(x**2, dim=-1, keepdim=True)
        x = torch.log(x + 1e-6)
        return self.bn(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (SE) attention block.

    Performs channel-wise recalibration by computing global average and max
    pooling, passing through a bottleneck FC network, and applying the
    resulting attention weights to the input feature map.

    Args:
        channels (int): Number of input/output channels.
        reduction (int): Reduction ratio for the bottleneck. Default: 4.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU6(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Hardsigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y_avg = x.view(b, c, -1).mean(dim=2)
        y_max, _ = x.view(b, c, -1).max(dim=2)
        y = self.fc(y_avg + y_max).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SwitchBraidNet(nn.Module):
    """SwitchBraidNet EEG classification model.

    A lightweight architecture combining dual-scale temporal convolutions,
    squeeze-and-excitation attention, depthwise spatial filtering, and
    log-variance pooling for efficient EEG-based brain-computer interfaces.

    Args:
        num_classes (int): Number of output classes.
        num_channels (int): Number of EEG channels. Default: 62.
        num_samples (int): Number of time samples per trial. Default: 256.
    """

    def __init__(self, num_classes, num_channels=62, num_samples=256):
        super().__init__()
        self.temp_deep = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
        self.temp_fast = nn.Conv2d(1, 16, (1, 32), padding=(0, 16), bias=False)
        self.bn_t = nn.BatchNorm2d(32)

        self.se = SEBlock(32)
        self.spatial = nn.Conv2d(32, 32, (num_channels, 1), groups=32, bias=False)
        self.bn_s = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout2d(0.2)

        self.mixer = nn.Sequential(
            nn.Conv2d(32, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
        )

        self.logvar = LogVarLayer(16)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input EEG tensor of shape
                ``(batch, 1, num_channels, num_samples)``.

        Returns:
            torch.Tensor: Class logits of shape ``(batch, num_classes)``.
        """
        t1 = self.temp_deep(x)
        t2 = self.temp_fast(x)
        x = torch.cat([t1, t2], dim=1)
        x = F.elu(self.bn_t(x))

        x = self.se(x)
        x = F.elu(self.bn_s(self.spatial(x)))
        x = self.dropout(x)
        x = self.mixer(x)

        x = self.logvar(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
