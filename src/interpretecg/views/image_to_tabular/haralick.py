"""Haralick gray-level co-occurrence texture descriptors.

The default follows Haralick, Shanmugam & Dinstein (1973): per-image
equal-probability quantization, distance-one co-occurrences at 0, 45, 90,
and 135 degrees, and the mean and range of each angular feature.  All 14
measures are retained, producing 28 values for the default distance.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from skimage.feature import graycomatrix

from ._validation import as_finite_image, channel_count, grayscale_image


Quantization = Literal["equal_probability", "linear"]

DEFAULT_DISTANCES = (1,)
DEFAULT_ANGLES = (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0)

HARALICK_MEASURE_NAMES = (
    "angular_second_moment",
    "contrast",
    "correlation",
    "sum_of_squares_variance",
    "inverse_difference_moment",
    "sum_average",
    "sum_variance",
    "sum_entropy",
    "entropy",
    "difference_variance",
    "difference_entropy",
    "information_correlation_1",
    "information_correlation_2",
    "maximal_correlation_coefficient",
)


def _validate_levels(levels: Any) -> int:
    if isinstance(levels, (bool, np.bool_)) or not isinstance(
        levels, (int, np.integer)
    ):
        raise TypeError("levels must be an integer between 2 and 256")
    levels = int(levels)
    if not 2 <= levels <= 256:
        raise ValueError("levels must be an integer between 2 and 256")
    return levels


def _validate_distances(distances: Sequence[int]) -> tuple[int, ...]:
    values = tuple(distances)
    if not values:
        raise ValueError("distances must not be empty")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in values
    ):
        raise TypeError("distances must contain positive integers")
    values = tuple(int(value) for value in values)
    if any(value < 1 for value in values):
        raise ValueError("distances must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError("distances must be unique")
    return values


def _validate_angles(angles: Sequence[float]) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in angles)
    except (TypeError, ValueError) as error:
        raise TypeError("angles must contain finite numeric values") from error
    if not values:
        raise ValueError("angles must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("angles must contain finite numeric values")
    return values


def _validate_quantization(quantization: str) -> Quantization:
    if quantization not in {"equal_probability", "linear"}:
        raise ValueError("quantization must be 'equal_probability' or 'linear'")
    return quantization  # type: ignore[return-value]


def quantize_gray_image(
    image: Any,
    *,
    levels: int = 16,
    method: Quantization = "equal_probability",
    thresholds: Sequence[float] | None = None,
) -> np.ndarray:
    """Convert an image into integer gray levels without using labels.

    Equal-probability quantization is invariant to strictly monotonic tone
    transformations. Linear quantization divides the per-image value range
    uniformly. RGB inputs are converted with Rec. 709 luminance weights.

    ``thresholds`` (``levels - 1`` increasing cut points) replaces the
    per-image quantization with a fixed one, estimated beforehand on training
    images.  Per-image quantization discards the image's scale; fixed cut
    points keep it, so two images that differ only in amplitude get different
    gray levels.
    """
    levels = _validate_levels(levels)
    method = _validate_quantization(method)
    gray = grayscale_image(as_finite_image(image))
    if thresholds is not None:
        cuts = np.asarray(thresholds, dtype=float)
        if cuts.shape != (levels - 1,) or np.any(np.diff(cuts) < 0):
            raise ValueError(f"thresholds must be {levels - 1} non-decreasing values")
        quantized = np.searchsorted(cuts, gray, side="right")
        return np.asarray(quantized, dtype=np.uint8 if levels <= 256 else np.uint16)
    minimum = float(np.min(gray))
    maximum = float(np.max(gray))
    if minimum == maximum:
        return np.zeros(gray.shape, dtype=np.uint8)

    if method == "equal_probability":
        probabilities = np.linspace(0.0, 1.0, levels + 1)[1:-1]
        thresholds = np.quantile(gray, probabilities, method="linear")
        quantized = np.searchsorted(thresholds, gray, side="right")
    else:
        scaled = (gray - minimum) / (maximum - minimum)
        quantized = np.floor(scaled * levels).astype(np.int64)
        quantized = np.minimum(quantized, levels - 1)
    dtype = np.uint8 if levels <= 256 else np.uint16
    return np.asarray(quantized, dtype=dtype)


def _entropy(probabilities: np.ndarray) -> float:
    positive = probabilities > 0
    return -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))


def _maximal_correlation_coefficient(
    matrix: np.ndarray,
    marginal_x: np.ndarray,
    marginal_y: np.ndarray,
) -> float:
    denominator = marginal_x[:, None] * marginal_y[None, :]
    normalized = np.divide(
        matrix,
        denominator,
        out=np.zeros_like(matrix),
        where=denominator > 0,
    )
    q_matrix = normalized @ matrix.T
    eigenvalues = np.linalg.eigvals(q_matrix)
    real_values = np.real(eigenvalues)
    real_values[np.abs(real_values) < 1e-14] = 0.0
    ordered = np.sort(real_values)
    second_largest = ordered[-2] if len(ordered) > 1 else 0.0
    return float(np.sqrt(max(second_largest, 0.0)))


def _haralick_measures(matrix: np.ndarray) -> np.ndarray:
    probability = np.asarray(matrix, dtype=float)
    total = float(np.sum(probability))
    if total <= 0:
        raise ValueError("co-occurrence matrix has no valid pixel pairs")
    probability /= total
    levels = probability.shape[0]
    indices = np.arange(levels, dtype=float)
    rows, columns = np.meshgrid(indices, indices, indexing="ij")
    marginal_x = probability.sum(axis=1)
    marginal_y = probability.sum(axis=0)
    mean_x = float(indices @ marginal_x)
    mean_y = float(indices @ marginal_y)
    std_x = float(np.sqrt(np.sum((indices - mean_x) ** 2 * marginal_x)))
    std_y = float(np.sqrt(np.sum((indices - mean_y) ** 2 * marginal_y)))

    angular_second_moment = float(np.sum(probability**2))
    contrast = float(np.sum((rows - columns) ** 2 * probability))
    if std_x == 0.0 or std_y == 0.0:
        correlation = 1.0
    else:
        correlation = float(
            (np.sum(rows * columns * probability) - mean_x * mean_y)
            / (std_x * std_y)
        )
    sum_of_squares_variance = float(
        np.sum((rows - mean_x) ** 2 * probability)
    )
    inverse_difference_moment = float(
        np.sum(probability / (1.0 + (rows - columns) ** 2))
    )

    sum_indices = (rows + columns).astype(int)
    sum_distribution = np.bincount(
        sum_indices.ravel(),
        weights=probability.ravel(),
        minlength=2 * levels - 1,
    )
    sum_axis = np.arange(len(sum_distribution), dtype=float)
    sum_average = float(sum_axis @ sum_distribution)
    sum_entropy = _entropy(sum_distribution)
    # Equation f7 in the original paper uses f8 (sum entropy) as center.
    sum_variance = float(
        np.sum((sum_axis - sum_entropy) ** 2 * sum_distribution)
    )
    entropy = _entropy(probability)

    difference_indices = np.abs(rows - columns).astype(int)
    difference_distribution = np.bincount(
        difference_indices.ravel(),
        weights=probability.ravel(),
        minlength=levels,
    )
    difference_axis = np.arange(levels, dtype=float)
    difference_mean = float(difference_axis @ difference_distribution)
    difference_variance = float(
        np.sum(
            (difference_axis - difference_mean) ** 2 * difference_distribution
        )
    )
    difference_entropy = _entropy(difference_distribution)

    entropy_x = _entropy(marginal_x)
    entropy_y = _entropy(marginal_y)
    independent = marginal_x[:, None] * marginal_y[None, :]
    positive_probability = probability > 0
    hxy1 = -float(
        np.sum(
            probability[positive_probability]
            * np.log(independent[positive_probability])
        )
    )
    hxy2 = _entropy(independent)
    entropy_denominator = max(entropy_x, entropy_y)
    information_correlation_1 = (
        0.0
        if entropy_denominator == 0.0
        else (entropy - hxy1) / entropy_denominator
    )
    information_gap = max(hxy2 - entropy, 0.0)
    information_correlation_2 = float(
        np.sqrt(max(1.0 - np.exp(-2.0 * information_gap), 0.0))
    )
    maximal_correlation_coefficient = _maximal_correlation_coefficient(
        probability,
        marginal_x,
        marginal_y,
    )

    values = np.asarray(
        [
            angular_second_moment,
            contrast,
            correlation,
            sum_of_squares_variance,
            inverse_difference_moment,
            sum_average,
            sum_variance,
            sum_entropy,
            entropy,
            difference_variance,
            difference_entropy,
            information_correlation_1,
            information_correlation_2,
            maximal_correlation_coefficient,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Haralick computation produced non-finite values")
    return values


def haralick_feature_names(
    *,
    distances: Sequence[int] = DEFAULT_DISTANCES,
) -> tuple[str, ...]:
    """Return names aligned with :func:`haralick_features`."""
    distance_values = _validate_distances(distances)
    return tuple(
        f"haralick__{measure}__distance_{distance}__{aggregation}"
        for distance in distance_values
        for measure in HARALICK_MEASURE_NAMES
        for aggregation in ("mean", "range")
    )


def haralick_features(
    image: Any,
    *,
    levels: int = 16,
    distances: Sequence[int] = DEFAULT_DISTANCES,
    angles: Sequence[float] = DEFAULT_ANGLES,
    quantization: Quantization = "equal_probability",
    thresholds: Sequence[float] | None = None,
) -> np.ndarray:
    """Return Haralick angular means and ranges for one image.

    ``thresholds`` fixes the gray-level cut points; see `quantize_gray_image`.
    """
    levels = _validate_levels(levels)
    distance_values = _validate_distances(distances)
    angle_values = _validate_angles(angles)
    quantization = _validate_quantization(quantization)
    quantized = quantize_gray_image(
        image,
        levels=levels,
        method=quantization,
        thresholds=thresholds,
    )
    if max(distance_values) >= min(quantized.shape):
        raise ValueError(
            "every image dimension must be larger than the largest distance"
        )
    matrices = graycomatrix(
        quantized,
        distances=distance_values,
        angles=angle_values,
        levels=levels,
        symmetric=True,
        normed=True,
    )

    output = []
    for distance_index in range(len(distance_values)):
        directional = np.stack(
            [
                _haralick_measures(matrices[:, :, distance_index, angle_index])
                for angle_index in range(len(angle_values))
            ]
        )
        for measure_index in range(len(HARALICK_MEASURE_NAMES)):
            values = directional[:, measure_index]
            output.extend((float(np.mean(values)), float(np.ptp(values))))
    result = np.asarray(output, dtype=float)
    if result.shape != (len(haralick_feature_names(distances=distance_values)),):
        raise RuntimeError("Haralick output width does not match feature names")
    return result

