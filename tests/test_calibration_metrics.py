"""Tests for calibration metrics."""

import numpy as np
import pytest

from wat_mod_giz.calibration.metrics import kge, log_nse, mae, nse, pbias, rmse


class TestNSE:
    def test_perfect_match_returns_one(self) -> None:
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        sim = np.array([1.0, 2.0, 3.0, 4.0])
        assert nse(obs, sim) == pytest.approx(1.0)

    def test_mean_simulation_returns_zero(self) -> None:
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        sim = np.full(4, np.mean(obs))
        assert nse(obs, sim) == pytest.approx(0.0)

    def test_zero_variance_observed_returns_negative_infinity(self) -> None:
        obs = np.full(4, 5.0)
        sim = np.array([1.0, 2.0, 3.0, 4.0])
        assert nse(obs, sim) == float(-np.inf)


class TestLogNSE:
    def test_perfect_match_returns_one(self) -> None:
        obs = np.array([0.0, 1.0, 2.0, 3.0])
        sim = np.array([0.0, 1.0, 2.0, 3.0])
        assert log_nse(obs, sim) == pytest.approx(1.0)


class TestKGE:
    def test_perfect_match_returns_one(self) -> None:
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        sim = np.array([1.0, 2.0, 3.0, 4.0])
        assert kge(obs, sim) == pytest.approx(1.0)

    def test_zero_variance_observed_stays_finite(self) -> None:
        obs = np.full(4, 3.0)
        sim = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.isfinite(kge(obs, sim))


class TestErrorMetrics:
    def test_pbias_perfect_match_is_zero(self) -> None:
        obs = np.array([1.0, 2.0, 3.0])
        sim = np.array([1.0, 2.0, 3.0])
        assert pbias(obs, sim) == pytest.approx(0.0)

    def test_pbias_zero_observed_sum_returns_infinity(self) -> None:
        obs = np.zeros(3)
        sim = np.array([1.0, 2.0, 3.0])
        assert pbias(obs, sim) == float(np.inf)

    def test_rmse_and_mae_perfect_match_are_zero(self) -> None:
        obs = np.array([1.0, 2.0, 3.0])
        sim = np.array([1.0, 2.0, 3.0])
        assert rmse(obs, sim) == pytest.approx(0.0)
        assert mae(obs, sim) == pytest.approx(0.0)
