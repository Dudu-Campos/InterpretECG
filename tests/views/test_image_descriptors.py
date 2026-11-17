from __future__ import annotations

import numpy as np
import pytest

from interpretecg.views.image_to_tabular import (
    HARALICK_MEASURE_NAMES,
    haralick_feature_names,
    haralick_features,
    hog_features,
    quantize_gray_image,
)


def _texture(size: int = 32) -> np.ndarray:
    rows, columns = np.indices((size, size))
    return (3 * rows + 5 * columns + (rows % 3) * 7).astype(float)


def test_equal_probability_quantization_is_monotonic_tone_invariant():
    image = _texture(16)
    first = quantize_gray_image(image)
    second = quantize_gray_image(4.5 * image + 17.0)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.uint8
    assert np.min(first) >= 0
    assert np.max(first) < 16


def test_haralick_default_has_fourteen_mean_and_range_pairs():
    features = haralick_features(_texture())
    names = haralick_feature_names()
    assert len(HARALICK_MEASURE_NAMES) == 14
    assert features.shape == (28,)
    assert len(names) == 28
    assert len(set(names)) == 28
    assert np.all(np.isfinite(features))


def test_haralick_mean_range_is_quarter_turn_invariant():
    image = _texture()
    first = haralick_features(image)
    second = haralick_features(np.rot90(image))
    np.testing.assert_allclose(first, second, rtol=1e-10, atol=1e-10)


def test_constant_image_haralick_is_finite():
    features = haralick_features(np.ones((16, 16)))
    assert np.all(np.isfinite(features))
    correlation_index = HARALICK_MEASURE_NAMES.index("correlation") * 2
    assert features[correlation_index] == pytest.approx(1.0)


def test_hog_default_width_matches_dalal_triggs_geometry():
    # 32x32 -> 4x4 cells -> 3x3 overlapping blocks; each block has
    # 2x2 cells and 9 orientations: 3*3*2*2*9 = 324.
    features = hog_features(_texture())
    assert features.shape == (324,)
    assert np.all(np.isfinite(features))


def test_hog_distinguishes_horizontal_and_vertical_edges():
    vertical = np.zeros((32, 32))
    vertical[:, 16:] = 1.0
    horizontal = vertical.T
    assert not np.allclose(hog_features(vertical), hog_features(horizontal))



def test_descriptor_validation_rejects_small_and_nonfinite_images():
    with pytest.raises(ValueError, match="too small"):
        hog_features(np.ones((8, 8)))
    image = np.ones((16, 16))
    image[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        haralick_features(image)
    with pytest.raises(ValueError, match="larger than"):
        haralick_features(np.ones((2, 2)), distances=(2,))

