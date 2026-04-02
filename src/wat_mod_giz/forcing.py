"""Typed forcing container for wat_mod_giz."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wat_mod_giz.types import Resolution

_RESOLUTION_TOLERANCES: dict[Resolution, tuple[float, float]] = {
    Resolution.daily: (22.0, 26.0),
}


def validate_time_spacing(time: np.ndarray, resolution: Resolution) -> None:
    """Raise ValueError if median time spacing doesn't match the resolution."""
    if len(time) <= 1:
        return

    median_gap_hours = float(np.median(np.diff(time)) / np.timedelta64(1, "h"))
    min_hours, max_hours = _RESOLUTION_TOLERANCES[resolution]
    if not (min_hours <= median_gap_hours <= max_hours):
        raise ValueError(
            f"time spacing (median {median_gap_hours:.1f} hours) does not match resolution '{resolution.value}'"
        )


@dataclass(frozen=True)
class Forcing:
    """Time-aligned meteorological forcing arrays."""

    time: np.ndarray
    precip: np.ndarray
    pet: np.ndarray
    temp: np.ndarray | None = None
    resolution: Resolution = Resolution.daily

    def __post_init__(self) -> None:
        """Validate time-aligned forcing arrays."""
        time = np.asarray(self.time)
        if time.ndim != 1:
            raise ValueError(f"time array must be 1D, got {time.ndim}D")
        time = time.astype("datetime64[ns]")

        precip = np.asarray(self.precip, dtype=np.float64)
        pet = np.asarray(self.pet, dtype=np.float64)
        temp = None if self.temp is None else np.asarray(self.temp, dtype=np.float64)

        for name, arr in (("precip", precip), ("pet", pet), ("temp", temp)):
            if arr is None:
                continue
            if arr.ndim != 1:
                raise ValueError(f"{name} array must be 1D, got {arr.ndim}D")
            if np.any(np.isnan(arr)):
                raise ValueError(f"{name} array contains NaN values")

        n = len(time)
        if len(precip) != n:
            raise ValueError(f"precip length {len(precip)} does not match time length {n}")
        if len(pet) != n:
            raise ValueError(f"pet length {len(pet)} does not match time length {n}")
        if temp is not None and len(temp) != n:
            raise ValueError(f"temp length {len(temp)} does not match time length {n}")

        validate_time_spacing(time, self.resolution)

        object.__setattr__(self, "time", time)
        object.__setattr__(self, "precip", precip)
        object.__setattr__(self, "pet", pet)
        object.__setattr__(self, "temp", temp)

    def __len__(self) -> int:
        """Return the number of timesteps."""
        return len(self.time)
