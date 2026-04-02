"""Elevation helpers for multi-layer snow and glacier support."""

from __future__ import annotations

import math

import numpy as np

GRAD_T_DEFAULT: float = 0.6
GRAD_P_DEFAULT: float = 0.00041
ELEV_CAP_PRECIP: float = 4000.0
GRAD_P_LINEAR_DEFAULT: float = 0.0004


def derive_layers(hypsometric_curve: np.ndarray, n_layers: int) -> tuple[np.ndarray, np.ndarray]:
    """Derive representative elevations and equal-area fractions for elevation bands."""
    layer_elevations = np.empty(n_layers)
    layer_fractions = np.full(n_layers, 1.0 / n_layers)
    percentiles = np.linspace(0, 100, 101)

    for i in range(n_layers):
        lower_percentile = i * 100.0 / n_layers
        upper_percentile = (i + 1) * 100.0 / n_layers
        lower_elev = np.interp(lower_percentile, percentiles, hypsometric_curve)
        upper_elev = np.interp(upper_percentile, percentiles, hypsometric_curve)
        layer_elevations[i] = (lower_elev + upper_elev) / 2.0

    return layer_elevations, layer_fractions


def extrapolate_temperature(
    input_temp: float,
    input_elevation: float,
    layer_elevation: float,
    gradient: float = GRAD_T_DEFAULT,
) -> float:
    """Extrapolate temperature to a target elevation."""
    return input_temp - gradient * (layer_elevation - input_elevation) / 100.0


def extrapolate_precipitation(
    input_precip: float,
    input_elevation: float,
    layer_elevation: float,
    gradient: float = GRAD_P_DEFAULT,
    elev_cap: float = ELEV_CAP_PRECIP,
) -> float:
    """Extrapolate precipitation with an exponential elevation gradient."""
    effective_input_elevation = min(input_elevation, elev_cap)
    effective_layer_elevation = min(layer_elevation, elev_cap)
    return input_precip * math.exp(gradient * (effective_layer_elevation - effective_input_elevation))


def extrapolate_precipitation_linear(
    input_precip: float,
    input_elevation: float,
    layer_elevation: float,
    gradient: float = GRAD_P_LINEAR_DEFAULT,
    elev_cap: float = ELEV_CAP_PRECIP,
) -> float:
    """Extrapolate precipitation with a linear elevation gradient."""
    effective_input_elevation = min(input_elevation, elev_cap)
    effective_layer_elevation = min(layer_elevation, elev_cap)
    return max(0.0, input_precip * (1.0 + gradient * (effective_layer_elevation - effective_input_elevation)))
