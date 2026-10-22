from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


_SEGMENT_SUFFIX = re.compile(r"_\d+_\d+\.wav$", flags=re.IGNORECASE)


def fine_label_from_large_filename(filename: str) -> str:
    """Recover the fine-grained label used by the 80,207-row notebook.

    Example:
        C0303_113_1.wav -> C0303
    """
    return _SEGMENT_SUFFIX.sub("", Path(filename).name)


def fine_label_ids_in_original_order(filenames: pd.Series) -> tuple[np.ndarray, dict[str, int]]:
    """Match the recovered notebook's order-dependent mapping behavior."""
    labels = filenames.map(fine_label_from_large_filename)
    unique = pd.unique(labels)
    mapping = {label: i for i, label in enumerate(unique)}
    return labels.map(mapping).to_numpy(), mapping


def legacy_11_label_from_filename(filename: str) -> int:
    """Label convention used by the smaller 1,152-row branch.

    Example:
        10.1.9.wav -> 10
    """
    return int(Path(filename).name.split(".")[0])
