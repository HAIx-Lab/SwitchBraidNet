"""
Experiment configuration for the Quantized Hybrid BCI pipeline.

This file defines all hyperparameters, paths, and settings used by the
training (``train_models.py``), evaluation (``evaluate_models.py``), and
analysis (``analyze_results.py``) scripts.

Modify this file to adjust models, paradigms, precision levels, training
parameters, or dataset paths before running experiments.
"""

import os

# Configuration for OpenBMI Model Workflow
CONFIG = {
    # Models to include in the workflow.
    "models": [
        "SwitchBraidNet",
    ],

    # Brain-Computer Interface paradigms
    "modes": [
        "mi",       # Motor Imagery
        "ssvep",    # Steady-State Visual Evoked Potential
        "erp",      # Event-Related Potential
    ],

    # Quantization precisions (bit-widths)
    "precisions": [
        32,     # FP32 (baseline)
        16,     # FP16 (mixed-precision)
        8       # INT8 (quantization-aware training)
    ],

    # -------------------------------------------------------------------------
    # Training Parameters
    # -------------------------------------------------------------------------
    "debug": False,              # Set True for quick test (100 samples, 2 epochs)
    "epochs": 50,                # Maximum training epochs per fold
    "batch_size": 32,            # Training batch size
    "k_folds": 5,               # Number of cross-validation folds
    "max_folds": None,           # Set to int to limit folds (None = run all)
    "cooldown_seconds": 0,       # GPU cooldown between experiments (seconds)
    "seed": 1607,                # Random seed for reproducibility
    "max_concurrent_eval_workers": 4,  # Parallel workers for evaluation

    # -------------------------------------------------------------------------
    # Data Parameters
    # -------------------------------------------------------------------------
    "num_samples_time": 256,     # Number of time points (samples) per trial

    # Data sample control: fraction of data files to use (1.0 = all)
    "train_data_limit": 1.0,
    "test_data_limit": 1.0,

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    "dataset_path": "../Dataset/openbmi/",   # Path to the OpenBMI dataset
    "results_dir": "Results",                # Directory for evaluation outputs
    "logs_dir_name": "logs",                 # Subdirectory for training logs

    # -------------------------------------------------------------------------
    # Hardware / BCI Parameters
    # -------------------------------------------------------------------------
    # Number of EEG channels per paradigm (OpenBMI dataset)
    "channels": {
        "mi": 20,
        "ssvep": 10,
        "erp": 28
    },
    # Number of output classes per paradigm
    "n_classes": {
        "mi": 2,
        "ssvep": 4,
        "erp": 2
    },
}
