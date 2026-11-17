"""Markov Transition Field robust to tied quantiles.

Reference
---------
Wang & Oates (2015), *Imaging Time-Series to Improve Classification and
Imputation*.

Contract
--------
- Input: one finite 1D series.
- Output: a square float image.
- Construction is deterministic, training-free, and never receives labels.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _as_series(x: np.ndarray | Sequence[float]) -> np.ndarray:
    series = np.asarray(x, dtype=float)
    if series.ndim != 1:
        raise ValueError(f"x must be one-dimensional, got shape {series.shape}")
    if len(series) < 2:
        raise ValueError(f"series too short: length={len(series)}, minimum=2")
    if not np.all(np.isfinite(series)):
        raise ValueError("x must contain only finite values")
    return series


def _quantile_states(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign empirical-quantile states using tie-preserving midranks."""
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 2:
        raise ValueError(f"n_bins must be an integer >= 2, got {n_bins}")

    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    cumulative_before = np.concatenate(([0], np.cumsum(counts[:-1])))
    mid_quantiles = (cumulative_before + 0.5 * counts) / len(x)
    states_by_value = np.minimum(
        np.floor(mid_quantiles * n_bins).astype(int), n_bins - 1
    )
    return states_by_value[inverse]


def _aggregate_square(image: np.ndarray, output_size: int) -> np.ndarray:
    length = image.shape[0]
    if not isinstance(output_size, (int, np.integer)):
        raise TypeError("image_size must be an integer or None")
    if not 1 <= output_size <= length:
        raise ValueError(
            f"image_size must be in [1, {length}], got {output_size}"
        )
    if output_size == length:
        return image.copy()

    groups = np.array_split(np.arange(length), output_size)
    reduced = np.empty((output_size, output_size), dtype=float)
    for row, row_indices in enumerate(groups):
        for column, column_indices in enumerate(groups):
            reduced[row, column] = image[np.ix_(row_indices, column_indices)].mean()
    return reduced


def markov_transition_field(
    x: np.ndarray | Sequence[float],
    *,
    n_bins: int = 8,
    image_size: int | None = None,
    bin_edges: Sequence[float] | None = None,
) -> np.ndarray:
    """Return a first-order quantile-binned Markov Transition Field.

    ``bin_edges`` (``n_bins - 1`` increasing interior edges, from training
    data) replaces the per-series quantiles, which make the field invariant
    to any monotonic transform of the series, amplitude included.

    ``field[i, j]`` is the empirical transition probability from the state
    observed at time ``i`` to the state observed at time ``j``. Equal values
    always receive the same state, even when empirical quantiles are tied.
    Empty transition rows remain zero.
    """
    series = _as_series(x)
    if bin_edges is None:
        states = _quantile_states(series, n_bins)
    else:
        edges = np.asarray(bin_edges, dtype=float)
        if edges.shape != (n_bins - 1,) or np.any(np.diff(edges) < 0):
            raise ValueError(f"bin_edges must be {n_bins - 1} non-decreasing values")
        states = np.searchsorted(edges, series, side="right")

    counts = np.zeros((n_bins, n_bins), dtype=float)
    np.add.at(counts, (states[:-1], states[1:]), 1.0)
    row_sums = counts.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        counts,
        row_sums,
        out=np.zeros_like(counts),
        where=row_sums > 0,
    )

    field = probabilities[states[:, None], states[None, :]]
    return field if image_size is None else _aggregate_square(field, image_size)
