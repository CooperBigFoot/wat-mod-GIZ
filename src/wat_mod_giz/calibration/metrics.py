"""Hydrological objective functions used by calibration."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from wat_mod_giz.calibration.types import ObjectiveName

MetricFunction = Callable[[np.ndarray, np.ndarray], float]

_EPSILON: float = 1.0e-6


def _as_arrays(observed: np.ndarray, simulated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(observed, dtype=np.float64)
    sim = np.asarray(simulated, dtype=np.float64)
    if obs.ndim != 1:
        raise ValueError(f"observed must be 1D, got {obs.ndim}D")
    if sim.ndim != 1:
        raise ValueError(f"simulated must be 1D, got {sim.ndim}D")
    if len(obs) != len(sim):
        raise ValueError(f"observed length {len(obs)} does not match simulated length {len(sim)}")
    return obs, sim


def nse(observed: np.ndarray, simulated: np.ndarray) -> float:
    obs, sim = _as_arrays(observed, simulated)
    denominator = float(np.sum((obs - np.mean(obs)) ** 2))
    if denominator == 0.0:
        return float(-np.inf)
    numerator = float(np.sum((obs - sim) ** 2))
    return 1.0 - numerator / denominator


def log_nse(observed: np.ndarray, simulated: np.ndarray) -> float:
    obs, sim = _as_arrays(observed, simulated)
    obs_log = np.log(obs + _EPSILON)
    sim_log = np.log(sim + _EPSILON)
    denominator = float(np.sum((obs_log - np.mean(obs_log)) ** 2))
    if denominator == 0.0:
        return float(-np.inf)
    numerator = float(np.sum((obs_log - sim_log) ** 2))
    return 1.0 - numerator / denominator


def kge(observed: np.ndarray, simulated: np.ndarray) -> float:
    obs, sim = _as_arrays(observed, simulated)
    obs_mean = float(np.mean(obs))
    sim_mean = float(np.mean(sim))
    obs_std = float(np.std(obs))
    sim_std = float(np.std(sim))

    if obs_std == 0.0 or sim_std == 0.0:
        correlation = 1.0 if np.allclose(obs, sim) else 0.0
    else:
        correlation = float(np.corrcoef(obs, sim)[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0

    alpha = 1.0 if obs_std == sim_std == 0.0 else (sim_std / obs_std if obs_std != 0.0 else 0.0)
    beta = 1.0 if obs_mean == sim_mean == 0.0 else (sim_mean / obs_mean if obs_mean != 0.0 else 0.0)
    return 1.0 - float(np.sqrt((correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def pbias(observed: np.ndarray, simulated: np.ndarray) -> float:
    obs, sim = _as_arrays(observed, simulated)
    denominator = float(np.sum(obs))
    if denominator == 0.0:
        return float(np.inf)
    return 100.0 * float(np.sum(sim - obs)) / denominator


def rmse(observed: np.ndarray, simulated: np.ndarray) -> float:
    obs, sim = _as_arrays(observed, simulated)
    return float(np.sqrt(np.mean((obs - sim) ** 2)))


def mae(observed: np.ndarray, simulated: np.ndarray) -> float:
    obs, sim = _as_arrays(observed, simulated)
    return float(np.mean(np.abs(obs - sim)))


_METRICS: dict[ObjectiveName, tuple[MetricFunction, str]] = {
    "nse": (nse, "maximize"),
    "kge": (kge, "maximize"),
    "log_nse": (log_nse, "maximize"),
    "rmse": (rmse, "minimize"),
    "mae": (mae, "minimize"),
    "pbias": (pbias, "minimize"),
}


def get_metric(name: ObjectiveName) -> tuple[MetricFunction, str]:
    """Return the metric function and optimization direction."""
    return _METRICS[name]


def list_metrics() -> list[ObjectiveName]:
    """Return supported calibration objective names."""
    return sorted(_METRICS)
