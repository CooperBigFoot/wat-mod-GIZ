"""Structured model outputs for wat_mod_giz."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SnowOutput:
    """Catchment-aggregated CemaNeige outputs."""

    precip_raw: np.ndarray
    snow_pliq: np.ndarray
    snow_psol: np.ndarray
    snow_pack: np.ndarray
    snow_thermal_state: np.ndarray
    snow_gratio: np.ndarray
    snow_pot_melt: np.ndarray
    snow_melt: np.ndarray
    snow_pliq_and_melt: np.ndarray
    snow_temp: np.ndarray
    snow_gthreshold: np.ndarray
    snow_glocalmax: np.ndarray

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert to a dictionary of arrays."""
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class SnowLayerOutputs:
    """Per-layer CemaNeige outputs for multi-layer runs."""

    layer_elevations: np.ndarray
    layer_fractions: np.ndarray
    snow_pack: np.ndarray
    snow_thermal_state: np.ndarray
    snow_gratio: np.ndarray
    snow_melt: np.ndarray
    snow_pliq_and_melt: np.ndarray
    layer_temp: np.ndarray
    layer_precip: np.ndarray

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert to a dictionary of arrays."""
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class ModelOutput[F]:
    """Combined model outputs with aligned time and optional snow diagnostics."""

    time: np.ndarray
    fluxes: F
    snow: SnowOutput | None = None
    snow_layers: SnowLayerOutputs | None = None

    @property
    def streamflow(self) -> np.ndarray:
        """Return the streamflow array from the model fluxes."""
        return self.fluxes.streamflow  # type: ignore[attr-defined, return-value]

    def __len__(self) -> int:
        """Return the number of timesteps."""
        return len(self.time)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert fluxes and optional snow diagnostics to a dataframe."""
        data = self.fluxes.to_dict()  # type: ignore[attr-defined]
        if self.snow is not None:
            data.update(self.snow.to_dict())  # type: ignore[attr-defined]

        dataframe = pd.DataFrame(data, index=self.time)
        dataframe.index.name = "time"
        return dataframe
