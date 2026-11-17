"""Recurrence quantification analysis of a binary recurrence plot.

This is the image-side readout of a recurrence matrix, as asked for by section
4.2 of the protocol: recurrence rate, determinism, laminarity and diagonal
line lengths.  It is **not** from the CNA project; see `../PROVENANCE.md`.

Haralick texture is the wrong readout for a thresholded recurrence plot.  The
matrix is binary with a fixed small recurrence rate, so per-image
equal-probability quantization collapses it to a single gray level: the
co-occurrence matrix degenerates and every measure takes the same value for
every record.  Gray-level texture belongs to the continuous-valued images
(Gramian angular fields, Markov transition fields, spectrograms, scalograms).

Measures follow Marwan, Romano, Thiel & Kurths (2007), *Recurrence plots for
the analysis of complex systems*, Physics Reports 438(5-6):237-329.

The line of identity and the Theiler band are excluded from the diagonal
measures, so `determinism` reflects genuine recurrent trajectories rather
than a point's proximity to itself.
"""

from __future__ import annotations

import numpy as np

from ._validation import as_finite_image, grayscale_image

#: Minimum length for a diagonal or vertical line to count, in samples.
DEFAULT_MIN_LINE = 2

RQA_MEASURE_NAMES: tuple[str, ...] = (
    "recurrence_rate",
    "determinism",
    "diagonal_mean_length",
    "diagonal_max_length",
    "divergence",
    "diagonal_entropy",
    "laminarity",
    "trapping_time",
    "vertical_max_length",
)


def rqa_feature_names() -> tuple[str, ...]:
    return tuple(f"rqa__{name}" for name in RQA_MEASURE_NAMES)


def _run_lengths(flags: np.ndarray) -> np.ndarray:
    """Lengths of the runs of True in a 1-D boolean array."""
    if not flags.size:
        return np.empty(0, dtype=int)
    padded = np.concatenate(([False], flags, [False]))
    changes = np.diff(padded.astype(np.int8))
    return np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)


def _diagonal_runs(matrix: np.ndarray, theiler_window: int) -> np.ndarray:
    """Run lengths along every diagonal outside the Theiler band.

    Only the upper triangle is scanned: a recurrence matrix is symmetric, so
    the lower triangle would double every count without adding information.
    """
    size = matrix.shape[0]
    runs = [
        _run_lengths(np.diagonal(matrix, offset))
        for offset in range(theiler_window + 1, size)
    ]
    return np.concatenate(runs) if runs else np.empty(0, dtype=int)


def _vertical_runs(matrix: np.ndarray, theiler_window: int) -> np.ndarray:
    """Run lengths along every column, with the Theiler band masked out."""
    size = matrix.shape[0]
    masked = matrix.copy()
    rows, columns = np.indices(masked.shape)
    masked[np.abs(rows - columns) <= theiler_window] = False
    return np.concatenate([_run_lengths(masked[:, index]) for index in range(size)])


def _line_measures(runs: np.ndarray, recurrent: int, min_line: int) -> tuple[float, float, float]:
    """Return (fraction in lines, mean line length, max line length)."""
    if not runs.size or recurrent == 0:
        return 0.0, 0.0, 0.0
    lines = runs[runs >= min_line]
    if not lines.size:
        return 0.0, 0.0, 0.0
    return (
        float(lines.sum() / recurrent),
        float(lines.mean()),
        float(lines.max()),
    )


def rqa_features(
    image,
    *,
    min_line: int = DEFAULT_MIN_LINE,
    theiler_window: int = 1,
) -> np.ndarray:
    """Return the nine recurrence quantification measures of a binary plot.

    `image` must be a square recurrence matrix.  Nonzero entries are taken as
    recurrent, so both a boolean and a 0/1 float matrix are accepted.
    """
    values = grayscale_image(as_finite_image(image))
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(
            f"recurrence matrix must be square, got shape {values.shape}"
        )
    if not isinstance(min_line, (int, np.integer)) or min_line < 2:
        raise ValueError(f"min_line must be an integer >= 2, got {min_line}")
    if not isinstance(theiler_window, (int, np.integer)) or theiler_window < 0:
        raise ValueError(
            f"theiler_window must be a non-negative integer, got {theiler_window}"
        )
    size = values.shape[0]
    if size <= theiler_window + 1:
        raise ValueError(
            f"matrix of size {size} leaves no pairs outside a Theiler band "
            f"of {theiler_window}"
        )

    matrix = values != 0.0
    rows, columns = np.indices(matrix.shape)
    outside = np.abs(rows - columns) > theiler_window
    eligible = int(outside.sum())
    recurrent = int((matrix & outside).sum())
    recurrence_rate = recurrent / eligible if eligible else 0.0

    # Diagonal measures are computed on the upper triangle; halve the point
    # count so that `determinism` stays a fraction of the points scanned.
    diagonal_runs = _diagonal_runs(matrix, theiler_window)
    determinism, diagonal_mean, diagonal_max = _line_measures(
        diagonal_runs, recurrent // 2, min_line
    )
    divergence = 1.0 / diagonal_max if diagonal_max else 0.0

    lines = diagonal_runs[diagonal_runs >= min_line]
    if lines.size:
        counts = np.bincount(lines)
        probabilities = counts[counts > 0] / counts.sum()
        diagonal_entropy = float(-(probabilities * np.log(probabilities)).sum())
    else:
        diagonal_entropy = 0.0

    vertical_runs = _vertical_runs(matrix, theiler_window)
    laminarity, trapping_time, vertical_max = _line_measures(
        vertical_runs, recurrent, min_line
    )

    return np.array(
        [
            recurrence_rate,
            determinism,
            diagonal_mean,
            diagonal_max,
            divergence,
            diagonal_entropy,
            laminarity,
            trapping_time,
            vertical_max,
        ],
        dtype=float,
    )
