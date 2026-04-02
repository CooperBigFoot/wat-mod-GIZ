"""Coupled GR6J+CemaNeige+glacier model."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from wat_mod_giz.elevation import (
    GRAD_P_DEFAULT,
    GRAD_P_LINEAR_DEFAULT,
    derive_layers,
    extrapolate_precipitation,
    extrapolate_precipitation_linear,
    extrapolate_temperature,
)
from wat_mod_giz.forcing import Forcing
from wat_mod_giz.models.gr6j import Parameters as GR6JParameters
from wat_mod_giz.models.gr6j import State as GR6JState
from wat_mod_giz.models.gr6j import step as gr6j_step
from wat_mod_giz.models.gr6j_cemaneige import (
    SNOW_LAYER_STATE_SIZE,
    STATE_SIZE_BASE,
    State,
    _aggregate_fluxes,
    _single_layer_step,
    compute_state_size,
)
from wat_mod_giz.models.gr6j_cemaneige import Parameters as CemaNeigeParameters
from wat_mod_giz.outputs import ModelOutput, SnowLayerOutputs, SnowOutput
from wat_mod_giz.processes.glacier import compute_ice_melt, compute_layer_glacier_melt
from wat_mod_giz.types import Catchment
from wat_mod_giz.unit_hydrographs import compute_uh_ordinates

PARAM_NAMES: tuple[str, ...] = ("x1", "x2", "x3", "x4", "x5", "x6", "ctg", "kf", "fi", "tm", "swe_th")
STATE_SIZE: int = compute_state_size(1)


@dataclass(frozen=True)
class Parameters:
    """Flat parameter set for GR6J+CemaNeige+glacier."""

    x1: float
    x2: float
    x3: float
    x4: float
    x5: float
    x6: float
    ctg: float
    kf: float
    fi: float
    tm: float
    swe_th: float

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        array = np.array(
            [self.x1, self.x2, self.x3, self.x4, self.x5, self.x6, self.ctg, self.kf, self.fi, self.tm, self.swe_th],
            dtype=np.float64,
        )
        if dtype is not None:
            array = array.astype(dtype)
        return array

    @classmethod
    def from_array(cls, array: np.ndarray) -> Parameters:
        """Reconstruct parameters from array form."""
        if len(array) != len(PARAM_NAMES):
            raise ValueError(f"Expected array of length {len(PARAM_NAMES)}, got {len(array)}")
        return cls(*(float(value) for value in array))

    def to_gr6j(self) -> GR6JParameters:
        """Project the parameter set onto the GR6J subset."""
        return GR6JParameters(self.x1, self.x2, self.x3, self.x4, self.x5, self.x6)

    def to_cemaneige(self) -> CemaNeigeParameters:
        """Project the parameter set onto the GR6J+CemaNeige subset."""
        return CemaNeigeParameters(self.x1, self.x2, self.x3, self.x4, self.x5, self.x6, self.ctg, self.kf)


@dataclass(frozen=True)
class GR6JCemaNeigeGlacierFluxes:
    """Combined snow, glacier, and GR6J flux outputs."""

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
    glacier_ice_melt: np.ndarray
    glacier_melt: np.ndarray
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


def _snow_and_glacier_step(
    state: State,
    params: Parameters,
    precip: float,
    temp: float,
    *,
    layer_elevations: np.ndarray | None = None,
    layer_fractions: np.ndarray | None = None,
    glacier_fractions: np.ndarray | None = None,
    input_elevation: float | None = None,
    temp_gradient: float | None = None,
    precip_gradient: float | None = None,
    precip_gradient_type: str | None = None,
) -> tuple[np.ndarray, dict[str, float], list[dict[str, float]] | None]:
    """Advance snow and glacier states and return aggregated diagnostics."""
    if glacier_fractions is None:
        glacier_fractions = np.zeros(state.n_layers, dtype=np.float64)

    if state.n_layers == 1 or layer_elevations is None or layer_fractions is None or input_elevation is None:
        new_snow_state, snow_fluxes = _single_layer_step(state.snow_layer_states[0], params.to_cemaneige(), precip, temp)
        glacier_ice_melt = compute_ice_melt(new_snow_state[0], temp, params.fi, params.tm, params.swe_th)
        glacier_melt = compute_layer_glacier_melt(glacier_ice_melt, float(glacier_fractions[0]))
        snow_layer_states = np.array([new_snow_state], dtype=np.float64)
        aggregated_fluxes = {
            **snow_fluxes,
            "glacier_ice_melt": glacier_ice_melt,
            "glacier_melt": glacier_melt,
            "layer_temp": temp,
            "layer_precip": precip,
        }
        return snow_layer_states, aggregated_fluxes, None

    per_layer_fluxes: list[dict[str, float]] = []
    snow_layer_states = np.zeros((state.n_layers, SNOW_LAYER_STATE_SIZE), dtype=np.float64)
    for idx in range(state.n_layers):
        layer_temp = extrapolate_temperature(
            temp,
            input_elevation,
            float(layer_elevations[idx]),
            gradient=temp_gradient if temp_gradient is not None else 0.6,
        )
        if precip_gradient_type == "linear":
            layer_precip = extrapolate_precipitation_linear(
                precip,
                input_elevation,
                float(layer_elevations[idx]),
                gradient=precip_gradient if precip_gradient is not None else GRAD_P_LINEAR_DEFAULT,
            )
        else:
            layer_precip = extrapolate_precipitation(
                precip,
                input_elevation,
                float(layer_elevations[idx]),
                gradient=precip_gradient if precip_gradient is not None else GRAD_P_DEFAULT,
            )

        new_layer_state, layer_fluxes = _single_layer_step(
            state.snow_layer_states[idx],
            params.to_cemaneige(),
            layer_precip,
            layer_temp,
        )
        glacier_ice_melt = compute_ice_melt(new_layer_state[0], layer_temp, params.fi, params.tm, params.swe_th)
        layer_fluxes["glacier_ice_melt"] = glacier_ice_melt
        layer_fluxes["glacier_melt"] = compute_layer_glacier_melt(glacier_ice_melt, float(glacier_fractions[idx]))
        layer_fluxes["layer_temp"] = layer_temp
        layer_fluxes["layer_precip"] = layer_precip
        snow_layer_states[idx] = new_layer_state
        per_layer_fluxes.append(layer_fluxes)

    aggregated_fluxes = _aggregate_fluxes(per_layer_fluxes, layer_fractions)
    return snow_layer_states, aggregated_fluxes, per_layer_fluxes


def step(
    state: State,
    params: Parameters,
    precip: float,
    pet: float,
    temp: float,
    uh1_ordinates: np.ndarray,
    uh2_ordinates: np.ndarray,
    layer_elevations: np.ndarray | None = None,
    layer_fractions: np.ndarray | None = None,
    glacier_fractions: np.ndarray | None = None,
    input_elevation: float | None = None,
    temp_gradient: float | None = None,
    precip_gradient: float | None = None,
    precip_gradient_type: str | None = None,
) -> tuple[State, dict[str, float]]:
    """Execute one GR6J+CemaNeige+glacier timestep."""
    snow_layer_states, snow_fluxes, _ = _snow_and_glacier_step(
        state,
        params,
        precip,
        temp,
        layer_elevations=layer_elevations,
        layer_fractions=layer_fractions,
        glacier_fractions=glacier_fractions,
        input_elevation=input_elevation,
        temp_gradient=temp_gradient,
        precip_gradient=precip_gradient,
        precip_gradient_type=precip_gradient_type,
    )

    gr6j_state = GR6JState(
        production_store=state.production_store,
        routing_store=state.routing_store,
        exponential_store=state.exponential_store,
        uh1_states=state.uh1_states.copy(),
        uh2_states=state.uh2_states.copy(),
    )
    liquid_input = snow_fluxes["snow_pliq_and_melt"] + snow_fluxes["glacier_melt"]
    new_gr6j_state, gr6j_fluxes = gr6j_step(gr6j_state, params.to_gr6j(), liquid_input, pet, uh1_ordinates, uh2_ordinates)

    new_state = State(
        production_store=new_gr6j_state.production_store,
        routing_store=new_gr6j_state.routing_store,
        exponential_store=new_gr6j_state.exponential_store,
        uh1_states=new_gr6j_state.uh1_states,
        uh2_states=new_gr6j_state.uh2_states,
        snow_layer_states=snow_layer_states,
    )
    fluxes = {
        "precip_raw": precip,
        "snow_pliq": snow_fluxes["snow_pliq"],
        "snow_psol": snow_fluxes["snow_psol"],
        "snow_pack": snow_fluxes["snow_pack"],
        "snow_thermal_state": snow_fluxes["snow_thermal_state"],
        "snow_gratio": snow_fluxes["snow_gratio"],
        "snow_pot_melt": snow_fluxes["snow_pot_melt"],
        "snow_melt": snow_fluxes["snow_melt"],
        "snow_pliq_and_melt": snow_fluxes["snow_pliq_and_melt"],
        "snow_temp": snow_fluxes["snow_temp"],
        "glacier_ice_melt": snow_fluxes["glacier_ice_melt"],
        "glacier_melt": snow_fluxes["glacier_melt"],
        **gr6j_fluxes,
    }
    return new_state, {key: float(value) for key, value in fluxes.items()}


def run(
    params: Parameters,
    forcing: Forcing,
    *,
    catchment: Catchment,
    initial_state: State | None = None,
) -> ModelOutput[GR6JCemaNeigeGlacierFluxes]:
    """Run the GR6J+CemaNeige+glacier model over a forcing timeseries."""
    if forcing.temp is None:
        raise ValueError("Temperature is required for GR6J+CemaNeige+glacier")

    state = State.initialize(params.to_cemaneige(), catchment) if initial_state is None else initial_state
    uh1_ordinates, uh2_ordinates = compute_uh_ordinates(params.x4)

    layer_elevations: np.ndarray | None = None
    layer_fractions: np.ndarray | None = None
    if catchment.n_layers > 1 and catchment.hypsometric_curve is not None:
        layer_elevations, layer_fractions = derive_layers(catchment.hypsometric_curve, catchment.n_layers)

    glacier_fractions = catchment.glacier_fractions
    if glacier_fractions is None:
        glacier_fractions = np.zeros(catchment.n_layers, dtype=np.float64)

    combined_rows: list[dict[str, float]] = []
    snow_rows: list[dict[str, float]] = []
    layer_pack: dict[str, list[np.ndarray]] = {
        "snow_pack": [],
        "snow_thermal_state": [],
        "snow_gratio": [],
        "snow_melt": [],
        "snow_pliq_and_melt": [],
        "layer_temp": [],
        "layer_precip": [],
    }

    for precip, pet, temp in zip(forcing.precip, forcing.pet, forcing.temp, strict=True):
        snow_layer_states, snow_fluxes, per_layer_fluxes = _snow_and_glacier_step(
            state,
            params,
            float(precip),
            float(temp),
            layer_elevations=layer_elevations,
            layer_fractions=layer_fractions,
            glacier_fractions=glacier_fractions,
            input_elevation=catchment.input_elevation,
            temp_gradient=catchment.temp_gradient,
            precip_gradient=catchment.precip_gradient,
            precip_gradient_type=catchment.precip_gradient_type.value,
        )

        gr6j_state = GR6JState(
            production_store=state.production_store,
            routing_store=state.routing_store,
            exponential_store=state.exponential_store,
            uh1_states=state.uh1_states.copy(),
            uh2_states=state.uh2_states.copy(),
        )
        liquid_input = snow_fluxes["snow_pliq_and_melt"] + snow_fluxes["glacier_melt"]
        new_gr6j_state, gr6j_fluxes = gr6j_step(
            gr6j_state,
            params.to_gr6j(),
            liquid_input,
            float(pet),
            uh1_ordinates,
            uh2_ordinates,
        )

        state = State(
            production_store=new_gr6j_state.production_store,
            routing_store=new_gr6j_state.routing_store,
            exponential_store=new_gr6j_state.exponential_store,
            uh1_states=new_gr6j_state.uh1_states,
            uh2_states=new_gr6j_state.uh2_states,
            snow_layer_states=snow_layer_states,
        )
        combined_fluxes = {
            "precip_raw": float(precip),
            "snow_pliq": snow_fluxes["snow_pliq"],
            "snow_psol": snow_fluxes["snow_psol"],
            "snow_pack": snow_fluxes["snow_pack"],
            "snow_thermal_state": snow_fluxes["snow_thermal_state"],
            "snow_gratio": snow_fluxes["snow_gratio"],
            "snow_pot_melt": snow_fluxes["snow_pot_melt"],
            "snow_melt": snow_fluxes["snow_melt"],
            "snow_pliq_and_melt": snow_fluxes["snow_pliq_and_melt"],
            "snow_temp": snow_fluxes["snow_temp"],
            "glacier_ice_melt": snow_fluxes["glacier_ice_melt"],
            "glacier_melt": snow_fluxes["glacier_melt"],
            **gr6j_fluxes,
        }
        combined_rows.append(combined_fluxes)
        snow_rows.append(
            {
                "precip_raw": float(precip),
                "snow_pliq": snow_fluxes["snow_pliq"],
                "snow_psol": snow_fluxes["snow_psol"],
                "snow_pack": snow_fluxes["snow_pack"],
                "snow_thermal_state": snow_fluxes["snow_thermal_state"],
                "snow_gratio": snow_fluxes["snow_gratio"],
                "snow_pot_melt": snow_fluxes["snow_pot_melt"],
                "snow_melt": snow_fluxes["snow_melt"],
                "snow_pliq_and_melt": snow_fluxes["snow_pliq_and_melt"],
                "snow_temp": snow_fluxes["snow_temp"],
                "snow_gthreshold": state.snow_layer_states[:, 2].mean(),
                "snow_glocalmax": state.snow_layer_states[:, 3].mean(),
            }
        )

        if per_layer_fluxes is not None:
            for key in layer_pack:
                layer_pack[key].append(np.array([layer_flux[key] for layer_flux in per_layer_fluxes], dtype=np.float64))

    flux_arrays = {
        key: np.array([row[key] for row in combined_rows], dtype=np.float64)
        for key in GR6JCemaNeigeGlacierFluxes.__dataclass_fields__
    }
    snow_arrays = {
        key: np.array([row[key] for row in snow_rows], dtype=np.float64)
        for key in SnowOutput.__dataclass_fields__
    }

    snow_layers = None
    if layer_elevations is not None and layer_fractions is not None:
        snow_layers = SnowLayerOutputs(
            layer_elevations=layer_elevations,
            layer_fractions=layer_fractions,
            snow_pack=np.vstack(layer_pack["snow_pack"]),
            snow_thermal_state=np.vstack(layer_pack["snow_thermal_state"]),
            snow_gratio=np.vstack(layer_pack["snow_gratio"]),
            snow_melt=np.vstack(layer_pack["snow_melt"]),
            snow_pliq_and_melt=np.vstack(layer_pack["snow_pliq_and_melt"]),
            layer_temp=np.vstack(layer_pack["layer_temp"]),
            layer_precip=np.vstack(layer_pack["layer_precip"]),
        )

    return ModelOutput(
        time=forcing.time,
        fluxes=GR6JCemaNeigeGlacierFluxes(**flux_arrays),
        snow=SnowOutput(**snow_arrays),
        snow_layers=snow_layers,
    )


__all__ = [
    "GR6JCemaNeigeGlacierFluxes",
    "PARAM_NAMES",
    "Parameters",
    "STATE_SIZE",
    "STATE_SIZE_BASE",
    "State",
    "compute_state_size",
    "run",
    "step",
]
