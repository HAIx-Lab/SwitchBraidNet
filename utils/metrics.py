"""
Evaluation metrics for Brain-Computer Interface systems.

Provides functions for computing:
  - Information Transfer Rate (ITR) in bits per minute.
  - Cohen's Kappa coefficient.
  - Model size estimation in MB and KB.
"""

import math


def compute_itr(acc, n_classes, trial_time=1.0):
    """Compute Information Transfer Rate (ITR) in bits per minute.

    Uses the Wolpaw ITR formula:
        B = log2(N) + P·log2(P) + (1−P)·log2((1−P)/(N−1))
        ITR = B × (60 / T)

    Args:
        acc (float): Classification accuracy (0 to 1).
        n_classes (int): Number of classes.
        trial_time (float): Trial duration in seconds.

    Returns:
        float: ITR in bits per minute, clipped to non-negative.
    """
    if acc <= 0 or n_classes <= 1 or trial_time <= 0:
        return 0.0
    p = acc
    n = n_classes
    itr = (math.log2(n) + p * math.log2(p) + (1 - p) * math.log2((1 - p) / (n - 1))) * (
        60.0 / trial_time
    )
    return max(itr, 0.0)


def compute_kappa(acc, n_classes):
    """Compute Cohen's Kappa from accuracy assuming uniform chance.

    Args:
        acc (float): Classification accuracy (0 to 1).
        n_classes (int): Number of classes.

    Returns:
        float: Kappa coefficient.
    """
    if n_classes <= 1:
        return 0.0
    p0 = acc
    pe = 1.0 / n_classes
    return (p0 - pe) / (1 - pe)


def itr_bits_per_min(n_classes, accuracy, mean_trial_sec):
    """Compute ITR in bits per minute with input clamping.

    Similar to :func:`compute_itr` but clamps accuracy to ``(1e-9, 1−1e-9)``
    to avoid ``log(0)`` errors.

    Args:
        n_classes (int): Number of classes.
        accuracy (float): Classification accuracy (0 to 1).
        mean_trial_sec (float): Mean trial duration in seconds.

    Returns:
        float: ITR in bits per minute.
    """
    if n_classes <= 1 or mean_trial_sec <= 0.0 or accuracy <= 0.0:
        return 0.0
    p = max(min(float(accuracy), 1.0 - 1e-9), 1e-9)
    b = math.log2(n_classes)
    b += p * math.log2(p) + (1.0 - p) * math.log2((1.0 - p) / (n_classes - 1))
    return max(b, 0.0) / mean_trial_sec * 60.0


def model_size_mb(model, bytes_per_param=4):
    """Estimate model size in megabytes.

    Args:
        model: PyTorch model.
        bytes_per_param (int): Bytes per parameter (e.g., 4 for FP32).

    Returns:
        float: Estimated model size in MB.
    """
    total_params = sum(p.numel() for p in model.parameters())
    return (total_params * bytes_per_param) / (1024**2)


def model_size_kb(model):
    """Compute actual model size in kilobytes (parameters + buffers).

    Args:
        model: PyTorch model.

    Returns:
        float: Model size in KB.
    """
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / 1024
