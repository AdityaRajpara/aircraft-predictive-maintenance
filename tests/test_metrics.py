import numpy as np

from src.models.metrics import nasa_score, rmse


def test_rmse():
    assert rmse([0, 0], [3, 4]) == np.sqrt(12.5)


def test_nasa_score_zero_when_perfect():
    assert nasa_score([10, 20], [10, 20]) == 0.0


def test_nasa_score_penalises_late_more_than_early():
    late = nasa_score([50], [60])   # predicted too much life
    early = nasa_score([50], [40])
    assert late > early > 0
    assert np.isclose(late, np.exp(1) - 1)
    assert np.isclose(early, np.exp(10 / 13) - 1)
