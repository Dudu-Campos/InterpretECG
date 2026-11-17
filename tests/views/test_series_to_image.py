"""Contract tests and synthetic positive controls for series-to-image."""

from __future__ import annotations


import numpy as np
import pytest


from interpretecg.views.series_to_image import (
    gramian_angular_field,
    markov_transition_field,
    morlet_cwt,
    recurrence_plot,
    scalogram,
    spectrogram,
)


def test_all_operators_are_deterministic_and_finite():
    x = np.sin(2.0 * np.pi * np.arange(128) / 16.0)
    calls = [
        lambda: spectrogram(x, nperseg=32),
        lambda: scalogram(x, n_scales=12),
        lambda: gramian_angular_field(x, image_size=32),
        lambda: markov_transition_field(x, n_bins=8, image_size=32),
        lambda: recurrence_plot(x, recurrence_rate=0.1),
    ]
    for call in calls:
        first = call()
        second = call()
        assert first.ndim == 2
        assert np.all(np.isfinite(first))
        np.testing.assert_array_equal(first, second)


def test_spectrogram_localizes_a_stationary_sinusoid():
    length = 256
    frequency = 1.0 / 16.0
    time = np.arange(length)
    image = spectrogram(time * 0 + np.sin(2 * np.pi * frequency * time), nperseg=64)
    dominant_bin = int(np.argmax(image.mean(axis=1)))
    expected_bin = round(frequency * 64)
    assert abs(dominant_bin - expected_bin) <= 1
    assert image[dominant_bin].mean() > 20 * np.median(image.mean(axis=1))


def test_morlet_scalogram_localizes_scale_of_a_sinusoid():
    length = 256
    period = 16.0
    time = np.arange(length)
    x = np.sin(2 * np.pi * time / period)
    scales = np.arange(1.0, 33.0)
    coefficients = morlet_cwt(x, scales=scales, omega0=6.0)
    image = scalogram(x, scales=scales, omega0=6.0)
    np.testing.assert_allclose(image, np.abs(coefficients) ** 2)
    dominant_scale = scales[np.argmax(image[:, 40:-40].mean(axis=1))]
    expected_scale = 6.0 * period / (2.0 * np.pi)
    assert abs(dominant_scale - expected_scale) <= 2.0


def test_gasf_diagonal_recovers_nonnegative_normalized_series():
    x = np.array([4.0, 7.0, 5.0, 10.0, 8.0])
    normalized = (x - x.min()) / (x.max() - x.min())
    field = gramian_angular_field(x)
    recovered = np.sqrt(np.maximum(0.0, (np.diag(field) + 1.0) / 2.0))
    np.testing.assert_allclose(recovered, normalized, atol=1e-12)
    np.testing.assert_allclose(field, field.T, atol=1e-12)


def test_gadf_is_antisymmetric_with_zero_diagonal():
    x = np.array([0.0, 0.2, 0.9, 0.4])
    field = gramian_angular_field(x, method="difference")
    np.testing.assert_allclose(field, -field.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(field), 0.0, atol=1e-12)


def test_markov_transition_field_uses_empirical_transition_probabilities():
    x = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    field = markov_transition_field(x, n_bins=2)
    # Both observed states transition deterministically to the other state.
    expected = (x[:, None] != x[None, :]).astype(float)
    np.testing.assert_array_equal(field, expected)


def test_field_reduction_has_requested_shape():
    x = np.linspace(-1.0, 1.0, 31) ** 3
    assert gramian_angular_field(x, image_size=7).shape == (7, 7)
    assert markov_transition_field(x, image_size=7).shape == (7, 7)


def test_recurrence_plot_contract_and_target_rate():
    rng = np.random.default_rng(7)
    x = rng.normal(size=200)
    target = 0.08
    plot = recurrence_plot(x, dimension=2, delay=2, recurrence_rate=target)
    assert plot.shape == (198, 198)
    np.testing.assert_array_equal(plot, plot.T)
    np.testing.assert_array_equal(np.diag(plot), np.ones(198))
    upper = plot[np.triu_indices(len(plot), k=1)]
    assert abs(upper.mean() - target) < 0.01


def test_recurrence_distance_mode_is_symmetric_and_unthresholded():
    x = np.array([0.0, 1.0, 3.0, 6.0])
    distances = recurrence_plot(x, mode="distance")
    expected = np.abs(x[:, None] - x[None, :])
    np.testing.assert_allclose(distances, expected)


@pytest.mark.parametrize(
    "call",
    [
        lambda x: spectrogram(x),
        lambda x: scalogram(x),
        lambda x: gramian_angular_field(x),
        lambda x: markov_transition_field(x),
        lambda x: recurrence_plot(x),
    ],
)
def test_rejects_nonfinite_and_matrix_inputs(call):
    with pytest.raises(ValueError, match="finite"):
        call(np.array([0.0, np.nan, 1.0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        call(np.ones((4, 4)))
