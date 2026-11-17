"""Tests for the public ``conversions.serie_to_image`` package."""

from __future__ import annotations

import numpy as np

from interpretecg.views.series_to_image import (
    gramian_angular_field,
    markov_transition_field,
    recurrence_plot,
    scalogram,
    spectrogram,
)


def test_public_operators_are_deterministic_finite_images():
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


def test_mtf_preserves_tied_binary_states():
    x = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    field = markov_transition_field(x, n_bins=2)
    expected = (x[:, None] != x[None, :]).astype(float)
    np.testing.assert_array_equal(field, expected)
