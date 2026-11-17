"""Shared validation for image descriptors."""

from __future__ import annotations

from typing import Any

import numpy as np


def as_finite_image(image: Any) -> np.ndarray:
    try:
        values = np.asarray(image, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("image values must be numeric") from error
    if values.ndim not in {2, 3}:
        raise ValueError("image must have shape (H,W) or (H,W,C)")
    if values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("image spatial dimensions must be positive")
    if values.ndim == 3 and values.shape[2] < 1:
        raise ValueError("image must contain at least one channel")
    if not np.all(np.isfinite(values)):
        raise ValueError("image values must contain only finite values")
    return values


def channel_count(image: np.ndarray) -> int:
    return 1 if image.ndim == 2 else int(image.shape[2])


def grayscale_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[..., 0]
    if image.shape[2] >= 3:
        return np.tensordot(
            image[..., :3],
            np.array([0.2126, 0.7152, 0.0722]),
            axes=([-1], [0]),
        )
    return image.mean(axis=2)

