"""Canonical conversions from a one-dimensional series to an image.

Every public operator is deterministic and training-free: it receives only
the series being converted and never receives labels.  The functions return
plain NumPy arrays so that resizing, normalization, and the image readout can
remain explicit pipeline stages.

References
----------
Gabor (1946); Allen (1977); Portnoff (1980); Cohen (1989, 1995): STFT.
Grossmann & Morlet (1984); Mallat (1989): continuous wavelet transform.
Wang & Oates (2015): Gramian angular fields.
Eckmann, Kamphorst & Ruelle (1987): recurrence plots.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.signal import fftconvolve, stft
from scipy.spatial.distance import cdist


def _as_series(x: np.ndarray | Sequence[float], *, min_length: int = 2) -> np.ndarray:
    """Return a finite 1D float array without silently flattening matrices."""
    series = np.asarray(x, dtype=float)
    if series.ndim != 1:
        raise ValueError(f"x must be one-dimensional, got shape {series.shape}")
    if len(series) < min_length:
        raise ValueError(
            f"series too short: length={len(series)}, minimum={min_length}"
        )
    if not np.all(np.isfinite(series)):
        raise ValueError("x must contain only finite values")
    return series


def _to_decibels(power: np.ndarray, eps: float) -> np.ndarray:
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    return 10.0 * np.log10(np.maximum(power, eps))


def _paa(x: np.ndarray, output_size: int) -> np.ndarray:
    """Piecewise aggregate approximation with exact fractional boundaries."""
    length = len(x)
    if not isinstance(output_size, (int, np.integer)):
        raise TypeError("image_size must be an integer or None")
    if not 1 <= output_size <= length:
        raise ValueError(
            f"image_size must be in [1, {length}], got {output_size}"
        )
    if output_size == length:
        return x.copy()

    boundaries = np.linspace(0.0, float(length), output_size + 1)
    cumulative = np.concatenate(([0.0], np.cumsum(x, dtype=float)))

    def integral_at(position: float) -> float:
        if position >= length:
            return float(cumulative[-1])
        index = int(np.floor(position))
        fraction = position - index
        return float(cumulative[index] + fraction * x[index])

    integrals = np.fromiter(
        (integral_at(point) for point in boundaries),
        dtype=float,
        count=output_size + 1,
    )
    return np.diff(integrals) / np.diff(boundaries)


def spectrogram(
    x: np.ndarray | Sequence[float],
    *,
    sampling_rate: float = 1.0,
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    window: str = "hann",
    to_db: bool = False,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return the one-sided STFT power spectrogram ``|Z(f,t)|^2``.

    The default window length is ``min(64, len(x))`` and the default overlap
    is 50%.  No boundary extension or end padding is used, keeping every
    output pixel supported by observed samples.
    """
    series = _as_series(x)
    if sampling_rate <= 0:
        raise ValueError(f"sampling_rate must be positive, got {sampling_rate}")

    segment = min(64, len(series)) if nperseg is None else int(nperseg)
    if not 2 <= segment <= len(series):
        raise ValueError(
            f"nperseg must be in [2, {len(series)}], got {segment}"
        )
    overlap = segment // 2 if noverlap is None else int(noverlap)
    if not 0 <= overlap < segment:
        raise ValueError(f"noverlap must be in [0, nperseg), got {overlap}")
    if nfft is not None and int(nfft) < segment:
        raise ValueError(f"nfft must be >= nperseg ({segment}), got {nfft}")

    _, _, coefficients = stft(
        series,
        fs=float(sampling_rate),
        window=window,
        nperseg=segment,
        noverlap=overlap,
        nfft=None if nfft is None else int(nfft),
        detrend=False,
        return_onesided=True,
        boundary=None,
        padded=False,
    )
    power = np.abs(coefficients) ** 2
    return _to_decibels(power, eps) if to_db else power


def _default_scales(length: int, n_scales: int) -> np.ndarray:
    if n_scales < 2:
        raise ValueError(f"n_scales must be >= 2, got {n_scales}")
    maximum = max(2.0, min(64.0, length / 4.0))
    return np.geomspace(1.0, maximum, num=n_scales)


def morlet_cwt(
    x: np.ndarray | Sequence[float],
    *,
    scales: np.ndarray | Sequence[float] | None = None,
    n_scales: int = 32,
    omega0: float = 6.0,
) -> np.ndarray:
    """Return complex Morlet continuous-wavelet coefficients.

    Rows follow increasing scale (high to low pseudo-frequency).  The Morlet
    mother wavelet is ``pi^(-1/4) exp(i*omega0*t) exp(-t^2/2)``.  The default
    ``omega0=6`` is the conventional near-admissible choice.
    """
    series = _as_series(x)
    if omega0 <= 0:
        raise ValueError(f"omega0 must be positive, got {omega0}")

    scale_values = (
        _default_scales(len(series), n_scales)
        if scales is None
        else np.asarray(scales, dtype=float)
    )
    if scale_values.ndim != 1 or len(scale_values) == 0:
        raise ValueError("scales must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(scale_values)) or np.any(scale_values <= 0):
        raise ValueError("scales must contain only finite positive values")

    coefficients = np.empty((len(scale_values), len(series)), dtype=complex)
    normalization = np.pi ** (-0.25)
    for row, scale in enumerate(scale_values):
        # +/-5 scales captures the Gaussian envelope while bounding cost.
        half_width = min(len(series) - 1, max(1, int(np.ceil(5.0 * scale))))
        time = np.arange(-half_width, half_width + 1, dtype=float)
        wavelet = (
            normalization
            * np.exp(1j * omega0 * time / scale)
            * np.exp(-0.5 * (time / scale) ** 2)
            / np.sqrt(scale)
        )
        coefficients[row] = fftconvolve(
            series, np.conjugate(wavelet[::-1]), mode="same"
        )
    return coefficients


def scalogram(
    x: np.ndarray | Sequence[float],
    *,
    scales: np.ndarray | Sequence[float] | None = None,
    n_scales: int = 32,
    omega0: float = 6.0,
    to_db: bool = False,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return the Morlet CWT power scalogram ``|W(scale,t)|^2``."""
    power = np.abs(
        morlet_cwt(x, scales=scales, n_scales=n_scales, omega0=omega0)
    ) ** 2
    return _to_decibels(power, eps) if to_db else power


def _minmax_scale(x: np.ndarray, value_range: tuple[float, float]) -> np.ndarray:
    lower, upper = value_range
    if (lower, upper) not in {(0.0, 1.0), (-1.0, 1.0)}:
        raise ValueError("value_range must be (0, 1) or (-1, 1)")
    minimum = float(x.min())
    maximum = float(x.max())
    if maximum == minimum:
        return np.full_like(x, lower, dtype=float)
    return lower + (x - minimum) * (upper - lower) / (maximum - minimum)


def gramian_angular_field(
    x: np.ndarray | Sequence[float],
    *,
    method: str = "summation",
    image_size: int | None = None,
    value_range: tuple[float, float] = (0.0, 1.0),
    scale: float | None = None,
) -> np.ndarray:
    """Return a Gramian angular summation or difference field.

    With ``scale`` the series is divided by that fixed amplitude and clipped
    to [-1, 1] instead of min-max rescaled per series, so the field keeps the
    series' amplitude.  ``scale`` must come from training data only.

    PAA is applied before construction when ``image_size < len(x)``.
    ``value_range=(0, 1)`` is the default because the GASF diagonal then
    determines the normalized series as ``sqrt((diag + 1) / 2)``.  Using
    ``(-1, 1)`` loses the sign on that inverse.
    """
    series = _as_series(x)
    if image_size is not None:
        series = _paa(series, image_size)
    if scale is None:
        scaled = np.clip(_minmax_scale(series, value_range), -1.0, 1.0)
    else:
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"scale must be finite and positive, got {scale}")
        lower, upper = value_range
        if (lower, upper) not in {(0.0, 1.0), (-1.0, 1.0)}:
            raise ValueError("value_range must be (0, 1) or (-1, 1)")
        unit = np.clip(series / scale, -1.0, 1.0)
        scaled = lower + (unit + 1.0) * (upper - lower) / 2.0
    cosine = scaled
    sine = np.sqrt(np.maximum(0.0, 1.0 - cosine**2))

    if method == "summation":
        return np.outer(cosine, cosine) - np.outer(sine, sine)
    if method == "difference":
        return np.outer(sine, cosine) - np.outer(cosine, sine)
    raise ValueError("method must be 'summation' or 'difference'")


def _delay_embedding(x: np.ndarray, dimension: int, delay: int) -> np.ndarray:
    if not isinstance(dimension, (int, np.integer)) or dimension < 1:
        raise ValueError(f"dimension must be an integer >= 1, got {dimension}")
    if not isinstance(delay, (int, np.integer)) or delay < 1:
        raise ValueError(f"delay must be an integer >= 1, got {delay}")
    count = len(x) - (dimension - 1) * delay
    if count < 2:
        raise ValueError(
            "delay embedding has fewer than two points: "
            f"length={len(x)}, dimension={dimension}, delay={delay}"
        )
    return np.stack(
        [x[offset * delay : offset * delay + count] for offset in range(dimension)],
        axis=1,
    )


def recurrence_plot(
    x: np.ndarray | Sequence[float],
    *,
    dimension: int = 1,
    delay: int = 1,
    recurrence_rate: float = 0.10,
    epsilon: float | None = None,
    theiler_window: int = 0,
    mode: str = "binary",
) -> np.ndarray:
    """Return a recurrence plot from delay-embedded Euclidean distances.

    In ``mode='binary'``, ``epsilon`` overrides the distance quantile selected
    by ``recurrence_rate``.  The target rate is measured only over pairs with
    ``|i-j| > theiler_window``; the main diagonal is always recurrent.
    ``mode='distance'`` returns the unthresholded distance image.
    """
    series = _as_series(x)
    points = _delay_embedding(series, dimension, delay)
    distances = cdist(points, points, metric="euclidean")
    if mode == "distance":
        return distances
    if mode != "binary":
        raise ValueError("mode must be 'binary' or 'distance'")
    if not isinstance(theiler_window, (int, np.integer)) or theiler_window < 0:
        raise ValueError(
            f"theiler_window must be a non-negative integer, got {theiler_window}"
        )

    count = len(points)
    rows, columns = np.triu_indices(count, k=theiler_window + 1)
    if len(rows) == 0:
        raise ValueError("theiler_window leaves no eligible off-diagonal pairs")
    eligible_distances = distances[rows, columns]

    if epsilon is None:
        if not 0.0 < recurrence_rate <= 1.0:
            raise ValueError(
                f"recurrence_rate must be in (0, 1], got {recurrence_rate}"
            )
        threshold = float(np.quantile(eligible_distances, recurrence_rate))
    else:
        threshold = float(epsilon)
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError(f"epsilon must be finite and non-negative, got {epsilon}")

    plot = distances <= threshold
    temporal_distance = np.abs(np.subtract.outer(np.arange(count), np.arange(count)))
    plot[(temporal_distance <= theiler_window) & (temporal_distance != 0)] = False
    np.fill_diagonal(plot, True)
    return plot.astype(float)
