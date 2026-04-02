"""Typed streamflow series container for wat_mod_giz."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wat_mod_giz.forcing import validate_time_spacing
from wat_mod_giz.types import Resolution


@dataclass(frozen=True)
class StreamflowSeries:
    """Time-aligned observed streamflow series."""

    time: np.ndarray
    streamflow: np.ndarray
    resolution: Resolution = Resolution.daily

    def __post_init__(self) -> None:
        """Validate time-aligned streamflow arrays."""
        time = np.asarray(self.time)
        if time.ndim != 1:
            raise ValueError(f"time array must be 1D, got {time.ndim}D")
        time = time.astype("datetime64[ns]")

        streamflow = np.asarray(self.streamflow, dtype=np.float64)
        if streamflow.ndim != 1:
            raise ValueError(f"streamflow array must be 1D, got {streamflow.ndim}D")
        if np.any(np.isnan(streamflow)):
            raise ValueError("streamflow array contains NaN values")
        if len(streamflow) != len(time):
            raise ValueError(f"streamflow length {len(streamflow)} does not match time length {len(time)}")

        validate_time_spacing(time, self.resolution)

        object.__setattr__(self, "time", time)
        object.__setattr__(self, "streamflow", streamflow)

    def __len__(self) -> int:
        """Return the number of timesteps."""
        return len(self.time)
