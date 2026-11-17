"""Histogram of Oriented Gradients image descriptors."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from skimage.feature import hog

from ._validation import as_finite_image


BlockNorm = Literal["L1", "L1-sqrt", "L2", "L2-Hys"]


def _validate_positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be a positive integer")
    if int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_pair(value: Any, *, name: str) -> tuple[int, int]:
    try:
        values = tuple(value)
    except TypeError as error:
        raise TypeError(f"{name} must contain two positive integers") from error
    if len(values) != 2:
        raise ValueError(f"{name} must contain two positive integers")
    return (
        _validate_positive_integer(values[0], name=name),
        _validate_positive_integer(values[1], name=name),
    )


def _validate_block_norm(block_norm: str) -> BlockNorm:
    if block_norm not in {"L1", "L1-sqrt", "L2", "L2-Hys"}:
        raise ValueError("block_norm must be 'L1', 'L1-sqrt', 'L2', or 'L2-Hys'")
    return block_norm  # type: ignore[return-value]


def _validate_hog_parameters(
    *,
    orientations: int,
    pixels_per_cell: tuple[int, int],
    cells_per_block: tuple[int, int],
    block_norm: str,
    transform_sqrt: bool,
) -> tuple[int, tuple[int, int], tuple[int, int], BlockNorm, bool]:
    orientation_count = _validate_positive_integer(
        orientations, name="orientations"
    )
    pixel_shape = _validate_pair(pixels_per_cell, name="pixels_per_cell")
    block_shape = _validate_pair(cells_per_block, name="cells_per_block")
    norm = _validate_block_norm(block_norm)
    if not isinstance(transform_sqrt, (bool, np.bool_)):
        raise TypeError("transform_sqrt must be boolean")
    return orientation_count, pixel_shape, block_shape, norm, bool(transform_sqrt)


def _validate_image_size(
    image: np.ndarray,
    pixels_per_cell: tuple[int, int],
    cells_per_block: tuple[int, int],
) -> None:
    cell_rows = image.shape[0] // pixels_per_cell[0]
    cell_columns = image.shape[1] // pixels_per_cell[1]
    if cell_rows < cells_per_block[0] or cell_columns < cells_per_block[1]:
        required_rows = pixels_per_cell[0] * cells_per_block[0]
        required_columns = pixels_per_cell[1] * cells_per_block[1]
        raise ValueError(
            "image is too small for HOG parameters; requires at least "
            f"({required_rows},{required_columns}) pixels"
        )


def hog_features(
    image: Any,
    *,
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (8, 8),
    cells_per_block: tuple[int, int] = (2, 2),
    block_norm: BlockNorm = "L2-Hys",
    transform_sqrt: bool = False,
) -> np.ndarray:
    """Return one flattened Dalal--Triggs HOG descriptor."""
    (
        orientations,
        pixels_per_cell,
        cells_per_block,
        block_norm,
        transform_sqrt,
    ) = _validate_hog_parameters(
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        transform_sqrt=transform_sqrt,
    )
    values = as_finite_image(image)
    _validate_image_size(values, pixels_per_cell, cells_per_block)
    if transform_sqrt and np.any(values < 0):
        raise ValueError("transform_sqrt=True requires non-negative image values")
    descriptor = hog(
        values,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        visualize=False,
        transform_sqrt=transform_sqrt,
        feature_vector=True,
        channel_axis=None if values.ndim == 2 else -1,
    )
    descriptor = np.asarray(descriptor, dtype=float).ravel()
    if not np.all(np.isfinite(descriptor)):
        raise RuntimeError("HOG computation produced non-finite values")
    return descriptor


def hog_feature_names(feature_count: int) -> tuple[str, ...]:
    feature_count = _validate_positive_integer(feature_count, name="feature_count")
    width = max(5, len(str(feature_count - 1)))
    return tuple(f"hog__{index:0{width}d}" for index in range(feature_count))

