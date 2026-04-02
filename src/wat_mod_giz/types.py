"""Shared domain types for wat_mod_giz."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class Resolution(StrEnum):
    """Temporal resolution of forcing data."""

    daily = "daily"

    @property
    def days_per_timestep(self) -> float:
        """Return the nominal timestep length in days."""
        return 1.0


class PrecipGradientType(StrEnum):
    """Precipitation gradient form for elevation extrapolation."""

    exponential = "exponential"
    linear = "linear"


@dataclass(frozen=True)
class Catchment:
    """Static catchment metadata used by snow and glacier components."""

    mean_annual_solid_precip: float
    n_layers: int = 1
    hypsometric_curve: np.ndarray | None = None
    input_elevation: float | None = None
    glacier_fractions: np.ndarray | None = None
    temp_gradient: float | None = None
    precip_gradient: float | None = None
    precip_gradient_type: PrecipGradientType = PrecipGradientType.exponential

    def __post_init__(self) -> None:
        """Validate multi-layer and glacier configuration."""
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        if self.hypsometric_curve is not None:
            object.__setattr__(self, "hypsometric_curve", np.asarray(self.hypsometric_curve, dtype=np.float64))

        if self.glacier_fractions is not None:
            object.__setattr__(self, "glacier_fractions", np.asarray(self.glacier_fractions, dtype=np.float64))

        if self.n_layers > 1:
            if self.hypsometric_curve is None:
                raise ValueError("hypsometric_curve is required when n_layers > 1")
            if self.input_elevation is None:
                raise ValueError("input_elevation is required when n_layers > 1")
            if len(self.hypsometric_curve) != 101:
                raise ValueError("hypsometric_curve must have 101 percentile points")

        if self.glacier_fractions is None:
            return

        if self.glacier_fractions.ndim != 1:
            raise ValueError(f"glacier_fractions must be 1D, got {self.glacier_fractions.ndim}D")
        if len(self.glacier_fractions) != self.n_layers:
            raise ValueError("glacier_fractions length must match n_layers")
        if np.any(self.glacier_fractions < 0.0) or np.any(self.glacier_fractions > 1.0):
            raise ValueError("glacier_fractions values must be in [0, 1]")
