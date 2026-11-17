"""Controles positivos para a quantificação de recorrência.

Os alvos vêm da teoria de recurrence plots (Marwan et al. 2007): uma série
periódica produz diagonais longas e determinismo próximo de 1, enquanto ruído
i.i.d. produz pontos isolados, determinismo baixo e diagonais de comprimento
mínimo.
"""

from __future__ import annotations

import numpy as np
import pytest

from interpretecg.views.image_to_tabular.rqa import (
    RQA_MEASURE_NAMES,
    rqa_feature_names,
    rqa_features,
)
from interpretecg.views.series_to_image import recurrence_plot

RECURRENCE_RATE = 0.05
THEILER = 1


def _plot(x: np.ndarray) -> np.ndarray:
    return recurrence_plot(
        x,
        dimension=7,
        delay=10,
        recurrence_rate=RECURRENCE_RATE,
        theiler_window=THEILER,
    )


def _measures(x: np.ndarray) -> dict[str, float]:
    values = rqa_features(_plot(x), theiler_window=THEILER)
    return dict(zip(RQA_MEASURE_NAMES, values))


def test_names_match_the_vector_width():
    x = np.sin(np.linspace(0, 20 * np.pi, 400))
    assert len(rqa_features(_plot(x), theiler_window=THEILER)) == len(RQA_MEASURE_NAMES)
    assert len(rqa_feature_names()) == len(RQA_MEASURE_NAMES)


def test_recurrence_rate_reproduces_the_requested_rate():
    """O limiar é escolhido para atingir a taxa pedida, fora da banda de Theiler."""
    x = np.random.default_rng(5).normal(size=600)
    measured = _measures(x)["recurrence_rate"]
    assert abs(measured - RECURRENCE_RATE) < 0.005, measured


def test_periodic_is_more_deterministic_than_noise():
    periodic = _measures(np.sin(np.linspace(0, 20 * np.pi, 1000)))
    noise = _measures(np.random.default_rng(0).normal(size=1000))
    assert periodic["determinism"] > 0.9, periodic["determinism"]
    assert noise["determinism"] < 0.3, noise["determinism"]
    assert periodic["diagonal_mean_length"] > 10 * noise["diagonal_mean_length"]
    assert periodic["diagonal_entropy"] > noise["diagonal_entropy"]


def test_noise_diagonals_collapse_to_the_minimum_length():
    """Sem estrutura temporal, uma diagonal raramente passa de poucos pontos."""
    noise = _measures(np.random.default_rng(1).normal(size=1000))
    assert 2.0 <= noise["diagonal_mean_length"] < 3.0, noise["diagonal_mean_length"]


def test_divergence_is_the_inverse_of_the_longest_diagonal():
    values = _measures(np.sin(np.linspace(0, 12 * np.pi, 500)))
    assert values["diagonal_max_length"] > 0
    assert values["divergence"] == pytest.approx(1.0 / values["diagonal_max_length"])


def test_rejects_non_square_and_degenerate_matrices():
    with pytest.raises(ValueError, match="square"):
        rqa_features(np.zeros((10, 12)))
    with pytest.raises(ValueError, match="Theiler band"):
        rqa_features(np.zeros((2, 2)), theiler_window=4)
    with pytest.raises(ValueError, match="min_line"):
        rqa_features(np.zeros((10, 10)), min_line=1)


def test_empty_matrix_yields_zeros_instead_of_nan():
    """Uma matriz sem recorrência não deve produzir NaN no cache de features."""
    values = rqa_features(np.zeros((40, 40)))
    assert np.all(np.isfinite(values))
    assert np.all(values == 0.0)
