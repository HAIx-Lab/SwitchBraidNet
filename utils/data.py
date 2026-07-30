"""
Dataset loading and preprocessing for OpenBMI EEG data.

This module handles:
  - Resolving dataset directory paths by session and BCI paradigm.
  - Collecting ``.npz`` file paths for train/test splits.
  - Loading labels from ``.npz`` files with flexible key resolution.
  - A PyTorch ``Dataset`` class (``HybridDataset``) for batched data loading.
  - Subject-level grouping utilities for per-subject evaluation.

Expected dataset directory structure (OpenBMI)::

    <dataset_root>/
    ├── train_sess1_MI/
    │   ├── S01_01_000.npz
    │   └── ...
    ├── test_sess1_MI/
    ├── train_sess1_SSVEP/
    ├── test_sess1_SSVEP/
    ├── train_sess1_ERP/
    ├── test_sess1_ERP/
    ├── train_sess2_MI/
    └── ...

Each ``.npz`` file contains:
  - ``data``: EEG trial array of shape ``(1, num_channels, num_samples)``.
  - ``labels``: Integer class label for the trial.
"""

from pathlib import Path
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from .helpers import project_root

# Default EEG channel counts per paradigm (OpenBMI dataset)
DEFAULT_CHANNELS = {"mi": 20, "ssvep": 10, "erp": 32}

# Key names used to load labels from .npz files
DEFAULT_LABEL_KEYS = {"mi": "labels", "ssvep": "labels", "erp": "labels"}

# Mapping from lowercase paradigm codes to directory suffixes
MODE_DIR_NAMES = {"mi": "MI", "ssvep": "SSVEP", "erp": "ERP"}

# Default sessions to load
DEFAULT_SESSIONS = (1, 2)


def _mode_dir_name(mode):
    """Map a paradigm code (e.g., ``'mi'``) to its directory suffix (``'MI'``)."""
    return MODE_DIR_NAMES.get(mode, mode.upper())


def _dataset_dir(root, session, mode, kind):
    """Build the path to a specific dataset split directory."""
    return root / f"{kind}_sess{session}_{_mode_dir_name(mode)}"


def collect_dataset_paths(mode, sessions=DEFAULT_SESSIONS, root=None):
    """Collect sorted lists of ``.npz`` file paths for a given paradigm.

    Args:
        mode: BCI paradigm (``"mi"``, ``"ssvep"``, or ``"erp"``).
        sessions: Tuple of session IDs to include. Default: ``(1, 2)``.
        root: Dataset root directory. Defaults to ``project_root()``.

    Returns:
        Tuple of (train_paths, test_paths), each a sorted list of file paths.
    """
    root_path = Path(root) if root else project_root()
    train_paths = []
    test_paths = []
    for sess in sessions:
        train_dir = _dataset_dir(root_path, sess, mode, "train")
        test_dir = _dataset_dir(root_path, sess, mode, "test")
        train_paths.extend(sorted(glob.glob(str(train_dir / "*.npz"))))
        test_paths.extend(sorted(glob.glob(str(test_dir / "*.npz"))))
    return train_paths, test_paths


def resolve_label_key(npz_file, key_spec, mode=None):
    """Resolve the correct label key from an ``.npz`` file.

    Args:
        npz_file: Loaded numpy ``.npz`` archive.
        key_spec: Expected key name(s) — a string or list of strings.
        mode: Paradigm name (for error messages).

    Returns:
        The resolved key string.

    Raises:
        KeyError: If no matching key is found in the archive.
    """
    if isinstance(key_spec, (list, tuple)):
        for key in key_spec:
            if key in npz_file.files:
                return key
    elif key_spec in npz_file.files:
        return key_spec

    available = ", ".join(npz_file.files)
    raise KeyError(
        f"Label key not found for mode={mode}. Requested={key_spec}. "
        f"Available keys: {available}"
    )


def load_labels(file_path, mode, label_key_by_mode=None):
    """Load the label array from a single ``.npz`` file.

    Args:
        file_path: Path to the ``.npz`` file.
        mode: BCI paradigm (used to look up the label key).
        label_key_by_mode: Optional dict mapping paradigm → label key.

    Returns:
        numpy.ndarray: The label value(s) from the file.
    """
    label_map = label_key_by_mode or DEFAULT_LABEL_KEYS
    key_spec = label_map.get(mode, "labels")
    with np.load(file_path) as d:
        label_key = resolve_label_key(d, key_spec, mode)
        return d[label_key]


class HybridDataset(Dataset):
    """PyTorch Dataset for loading EEG trials from ``.npz`` files.

    Each ``.npz`` file is expected to contain:
      - ``"data"``: EEG array of shape ``(1, num_channels, num_samples)``.
      - A label key (resolved via ``label_key_by_mode``).

    Args:
        files: List of ``.npz`` file paths.
        mode: BCI paradigm (``"mi"``, ``"ssvep"``, or ``"erp"``).
        label_key_by_mode: Optional dict overriding default label key names.
    """

    def __init__(self, files, mode, label_key_by_mode=None):
        self.files = files
        self.mode = mode
        self.label_key_by_mode = label_key_by_mode or DEFAULT_LABEL_KEYS

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        key_spec = self.label_key_by_mode.get(self.mode, "labels")
        label_key = resolve_label_key(d, key_spec, self.mode)
        return (
            torch.from_numpy(d["data"]).float(),
            torch.tensor(d[label_key]).long(),
        )


def extract_subject_id(file_path):
    """Extract the subject ID from a file path.

    Assumes filename format: ``S01_02_000.npz`` → subject ``"S01"``.

    Args:
        file_path: Path to a ``.npz`` data file.

    Returns:
        str: The subject identifier (e.g., ``"S01"``).
    """
    return Path(file_path).stem.split("_")[0]


def group_paths_by_subject(paths):
    """Group a list of file paths by subject ID.

    Args:
        paths: List of ``.npz`` file paths.

    Returns:
        dict: Mapping from subject ID → list of file paths.
    """
    from collections import defaultdict

    subject_paths = defaultdict(list)
    for p in paths:
        subj_id = extract_subject_id(p)
        subject_paths[subj_id].append(p)
    return dict(subject_paths)
