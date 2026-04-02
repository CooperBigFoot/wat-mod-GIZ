"""Standalone GR6J daily rainfall-runoff model."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from wat_mod_giz.forcing import Forcing
from wat_mod_giz.outputs import ModelOutput
from wat_mod_giz.processes.gr6j import (
    direct_branch,
    exponential_store_update,
    groundwater_exchange,
    percolation,
    production_store_update,
    routing_store_update,
)
from wat_mod_giz.unit_hydrographs import NH, compute_uh_ordinates, convolve_uh

B: float = 0.9
C: float = 0.4
PARAM_NAMES: tuple[str, ...] = ("x1", "x2", "x3", "x4", "x5", "x6")
STATE_SIZE: int = 63


@dataclass(frozen=True)
class Parameters:
    """GR6J parameter set."""

    x1: float
    x2: float
    x3: float
    x4: float
    x5: float
    x6: float

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        array = np.array([self.x1, self.x2, self.x3, self.x4, self.x5, self.x6], dtype=np.float64)
        if dtype is not None:
            array = array.astype(dtype)
        return array

    @classmethod
    def from_array(cls, array: np.ndarray) -> Parameters:
        """Reconstruct parameters from array form."""
        if len(array) != len(PARAM_NAMES):
            raise ValueError(f"Expected array of length {len(PARAM_NAMES)}, got {len(array)}")
        return cls(*(float(value) for value in array))


@dataclass
class State:
    """Mutable GR6J model state."""

    production_store: float
    routing_store: float
    exponential_store: float
    uh1_states: np.ndarray
    uh2_states: np.ndarray

    @classmethod
    def initialize(cls, params: Parameters) -> State:
        """Build the default GR6J initial state from parameters."""
        return cls(
            production_store=0.3 * params.x1,
            routing_store=0.5 * params.x3,
            exponential_store=0.0,
            uh1_states=np.zeros(NH, dtype=np.float64),
            uh2_states=np.zeros(2 * NH, dtype=np.float64),
        )

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        array = np.empty(STATE_SIZE, dtype=np.float64)
        array[0] = self.production_store
        array[1] = self.routing_store
        array[2] = self.exponential_store
        array[3:23] = self.uh1_states
        array[23:63] = self.uh2_states
        if dtype is not None:
            array = array.astype(dtype)
        return array

    @classmethod
    def from_array(cls, array: np.ndarray) -> State:
        """Reconstruct state from array form."""
        if len(array) != STATE_SIZE:
            raise ValueError(f"Expected array of length {STATE_SIZE}, got {len(array)}")
        return cls(
            production_store=float(array[0]),
            routing_store=float(array[1]),
            exponential_store=float(array[2]),
            uh1_states=array[3:23].copy(),
            uh2_states=array[23:63].copy(),
        )


@dataclass(frozen=True)
class GR6JFluxes:
    """GR6J timeseries flux outputs."""

    pet: np.ndarray
    precip: np.ndarray
    production_store: np.ndarray
    net_rainfall: np.ndarray
    storage_infiltration: np.ndarray
    actual_et: np.ndarray
    percolation: np.ndarray
    effective_rainfall: np.ndarray
    q9: np.ndarray
    q1: np.ndarray
    routing_store: np.ndarray
    exchange: np.ndarray
    actual_exchange_routing: np.ndarray
    actual_exchange_direct: np.ndarray
    actual_exchange_total: np.ndarray
    qr: np.ndarray
    qrexp: np.ndarray
    exponential_store: np.ndarray
    qd: np.ndarray
    streamflow: np.ndarray

    def to_dict(self) -> dict[str, np.ndarray]:
        """Return a dictionary representation of the flux arrays."""
        return {field.name: getattr(self, field.name) for field in fields(self)}


GR6JOutput = GR6JFluxes


def step(
    state: State,
    params: Parameters,
    precip: float,
    pet: float,
    uh1_ordinates: np.ndarray,
    uh2_ordinates: np.ndarray,
) -> tuple[State, dict[str, float]]:
    """Execute one GR6J timestep."""
    prod_store_after_ps, actual_et, net_rainfall_pn, effective_rainfall_pr = production_store_update(
        precip,
        pet,
        state.production_store,
        params.x1,
    )
    storage_infiltration = net_rainfall_pn - effective_rainfall_pr if precip >= pet else 0.0

    prod_store_after_perc, percolation_amount = percolation(prod_store_after_ps, params.x1)
    total_effective_rainfall = effective_rainfall_pr + percolation_amount

    uh1_input = B * total_effective_rainfall
    uh2_input = (1.0 - B) * total_effective_rainfall
    new_uh1_states, q9 = convolve_uh(state.uh1_states, uh1_input, uh1_ordinates)
    new_uh2_states, q1 = convolve_uh(state.uh2_states, uh2_input, uh2_ordinates)

    exchange = groundwater_exchange(state.routing_store, params.x2, params.x3, params.x5)

    routing_input = (1.0 - C) * q9
    new_routing_store, qr, actual_exchange_routing = routing_store_update(
        state.routing_store,
        routing_input,
        exchange,
        params.x3,
    )

    exp_input = C * q9
    new_exp_store, qrexp = exponential_store_update(state.exponential_store, exp_input, exchange, params.x6)

    qd, actual_exchange_direct = direct_branch(q1, exchange)
    actual_exchange_total = actual_exchange_routing + actual_exchange_direct
    streamflow = max(qr + qrexp + qd, 0.0)

    new_state = State(
        production_store=prod_store_after_perc,
        routing_store=new_routing_store,
        exponential_store=new_exp_store,
        uh1_states=new_uh1_states,
        uh2_states=new_uh2_states,
    )
    fluxes = {
        "pet": pet,
        "precip": precip,
        "production_store": prod_store_after_perc,
        "net_rainfall": net_rainfall_pn,
        "storage_infiltration": storage_infiltration,
        "actual_et": actual_et,
        "percolation": percolation_amount,
        "effective_rainfall": total_effective_rainfall,
        "q9": q9,
        "q1": q1,
        "routing_store": new_routing_store,
        "exchange": exchange,
        "actual_exchange_routing": actual_exchange_routing,
        "actual_exchange_direct": actual_exchange_direct,
        "actual_exchange_total": actual_exchange_total,
        "qr": qr,
        "qrexp": qrexp,
        "exponential_store": new_exp_store,
        "qd": qd,
        "streamflow": streamflow,
    }
    return new_state, {key: float(value) for key, value in fluxes.items()}


def run(
    params: Parameters,
    forcing: Forcing,
    initial_state: State | None = None,
) -> ModelOutput[GR6JFluxes]:
    """Run GR6J over a forcing timeseries."""
    state = State.initialize(params) if initial_state is None else initial_state
    uh1_ordinates, uh2_ordinates = compute_uh_ordinates(params.x4)

    results: list[dict[str, float]] = []
    for precip, pet in zip(forcing.precip, forcing.pet, strict=True):
        state, fluxes = step(state, params, float(precip), float(pet), uh1_ordinates, uh2_ordinates)
        results.append(fluxes)

    arrays = {
        key: np.array([row[key] for row in results], dtype=np.float64)
        for key in GR6JFluxes.__dataclass_fields__
    }

    return ModelOutput(
        time=forcing.time,
        fluxes=GR6JFluxes(**arrays),
        snow=None,
        snow_layers=None,
    )


__all__ = [
    "B",
    "C",
    "GR6JFluxes",
    "GR6JOutput",
    "PARAM_NAMES",
    "Parameters",
    "STATE_SIZE",
    "State",
    "run",
    "step",
]
