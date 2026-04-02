"""Coupled GR6J+CemaNeige model."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np

from wat_mod_giz.elevation import (
    ELEV_CAP_PRECIP,
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
from wat_mod_giz.outputs import ModelOutput, SnowLayerOutputs, SnowOutput
from wat_mod_giz.processes.cemaneige import (
    GTHRESHOLD_FACTOR,
    compute_actual_melt,
    compute_gratio,
    compute_potential_melt,
    compute_solid_fraction,
    partition_precipitation,
    update_thermal_state,
)
from wat_mod_giz.types import Catchment, PrecipGradientType
from wat_mod_giz.unit_hydrographs import compute_uh_ordinates

STATE_SIZE_BASE: int = 63
SNOW_LAYER_STATE_SIZE: int = 4
PARAM_NAMES: tuple[str, ...] = ("x1", "x2", "x3", "x4", "x5", "x6", "ctg", "kf")


def compute_state_size(n_layers: int) -> int:
    """Return the flattened state size for the chosen layer count."""
    return STATE_SIZE_BASE + n_layers * SNOW_LAYER_STATE_SIZE


@dataclass(frozen=True)
class Parameters:
    """Flat parameter set for GR6J+CemaNeige."""

    x1: float
    x2: float
    x3: float
    x4: float
    x5: float
    x6: float
    ctg: float
    kf: float

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        array = np.array([self.x1, self.x2, self.x3, self.x4, self.x5, self.x6, self.ctg, self.kf], dtype=np.float64)
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
        """Project coupled parameters onto the GR6J subset."""
        return GR6JParameters(self.x1, self.x2, self.x3, self.x4, self.x5, self.x6)


@dataclass
class State:
    """Combined GR6J stores and per-layer snow state."""

    production_store: float
    routing_store: float
    exponential_store: float
    uh1_states: np.ndarray
    uh2_states: np.ndarray
    snow_layer_states: np.ndarray

    @property
    def n_layers(self) -> int:
        """Return the number of snow layers."""
        return self.snow_layer_states.shape[0]

    @classmethod
    def initialize(cls, params: Parameters, catchment: Catchment) -> State:
        """Build the default coupled initial state."""
        base_gthreshold = GTHRESHOLD_FACTOR * catchment.mean_annual_solid_precip
        snow_layer_states = np.zeros((catchment.n_layers, SNOW_LAYER_STATE_SIZE), dtype=np.float64)

        if catchment.n_layers > 1 and catchment.hypsometric_curve is not None and catchment.input_elevation is not None:
            layer_elevations, _ = derive_layers(catchment.hypsometric_curve, catchment.n_layers)
            if catchment.precip_gradient is not None:
                gradient = catchment.precip_gradient
            elif catchment.precip_gradient_type == PrecipGradientType.linear:
                gradient = GRAD_P_LINEAR_DEFAULT
            else:
                gradient = GRAD_P_DEFAULT

            input_elevation = min(catchment.input_elevation, ELEV_CAP_PRECIP)
            for idx, layer_elevation in enumerate(layer_elevations):
                effective_layer_elevation = min(float(layer_elevation), ELEV_CAP_PRECIP)
                if catchment.precip_gradient_type == PrecipGradientType.linear:
                    ratio = max(0.0, 1.0 + gradient * (effective_layer_elevation - input_elevation))
                else:
                    ratio = math.exp(gradient * (effective_layer_elevation - input_elevation))
                gthreshold = base_gthreshold * ratio
                snow_layer_states[idx, 2] = gthreshold
                snow_layer_states[idx, 3] = gthreshold
        else:
            snow_layer_states[:, 2] = base_gthreshold
            snow_layer_states[:, 3] = base_gthreshold

        gr6j_state = GR6JState.initialize(params.to_gr6j())
        return cls(
            production_store=gr6j_state.production_store,
            routing_store=gr6j_state.routing_store,
            exponential_store=gr6j_state.exponential_store,
            uh1_states=gr6j_state.uh1_states,
            uh2_states=gr6j_state.uh2_states,
            snow_layer_states=snow_layer_states,
        )

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        array = np.empty(compute_state_size(self.n_layers), dtype=np.float64)
        array[0] = self.production_store
        array[1] = self.routing_store
        array[2] = self.exponential_store
        array[3:23] = self.uh1_states
        array[23:63] = self.uh2_states
        array[63:] = self.snow_layer_states.flatten()
        if dtype is not None:
            array = array.astype(dtype)
        return array

    @classmethod
    def from_array(cls, array: np.ndarray, n_layers: int) -> State:
        """Reconstruct state from array form."""
        expected_size = compute_state_size(n_layers)
        if len(array) != expected_size:
            raise ValueError(f"Expected array of length {expected_size}, got {len(array)}")
        return cls(
            production_store=float(array[0]),
            routing_store=float(array[1]),
            exponential_store=float(array[2]),
            uh1_states=array[3:23].copy(),
            uh2_states=array[23:63].copy(),
            snow_layer_states=array[63:].reshape(n_layers, SNOW_LAYER_STATE_SIZE).copy(),
        )


@dataclass(frozen=True)
class GR6JCemaNeigeFluxes:
    """Combined snow and GR6J flux outputs."""

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


def _single_layer_step(
    state: np.ndarray,
    params: Parameters,
    precip: float,
    temp: float,
) -> tuple[np.ndarray, dict[str, float]]:
    solid_fraction = compute_solid_fraction(temp)
    pliq, psol = partition_precipitation(precip, solid_fraction)
    snow_pack = state[0] + psol
    thermal_state = update_thermal_state(state[1], temp, params.ctg)
    potential_melt = compute_potential_melt(thermal_state, temp, params.kf, snow_pack)
    gratio_for_melt = compute_gratio(snow_pack, state[2])
    melt = compute_actual_melt(potential_melt, gratio_for_melt)
    snow_pack -= melt
    gratio = compute_gratio(snow_pack, state[2])
    liquid_input = pliq + melt

    new_state = np.array([snow_pack, thermal_state, state[2], state[3]], dtype=np.float64)
    fluxes = {
        "snow_pliq": pliq,
        "snow_psol": psol,
        "snow_pack": snow_pack,
        "snow_thermal_state": thermal_state,
        "snow_gratio": gratio,
        "snow_pot_melt": potential_melt,
        "snow_melt": melt,
        "snow_pliq_and_melt": liquid_input,
        "snow_temp": temp,
        "snow_gthreshold": state[2],
        "snow_glocalmax": state[3],
    }
    return new_state, fluxes


def _aggregate_fluxes(layer_fluxes: list[dict[str, float]], layer_fractions: np.ndarray) -> dict[str, float]:
    """Area-weight per-layer snow diagnostics."""
    keys = layer_fluxes[0].keys()
    return {
        key: sum(flux[key] * fraction for flux, fraction in zip(layer_fluxes, layer_fractions, strict=True))
        for key in keys
    }


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
    input_elevation: float | None = None,
    temp_gradient: float | None = None,
    precip_gradient: float | None = None,
    precip_gradient_type: str | None = None,
) -> tuple[State, dict[str, float], list[dict[str, float]] | None]:
    """Execute one coupled GR6J+CemaNeige timestep."""
    if state.n_layers == 1 or layer_elevations is None or layer_fractions is None or input_elevation is None:
        new_snow_state, snow_fluxes = _single_layer_step(state.snow_layer_states[0], params, precip, temp)
        snow_layer_states = np.array([new_snow_state], dtype=np.float64)
        aggregated_snow_fluxes = snow_fluxes
        per_layer_fluxes = None
    else:
        per_layer_fluxes = []
        snow_layer_states = np.zeros_like(state.snow_layer_states)
        layer_temps: list[float] = []
        layer_precips: list[float] = []

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

            new_layer_state, layer_fluxes = _single_layer_step(state.snow_layer_states[idx], params, layer_precip, layer_temp)
            snow_layer_states[idx] = new_layer_state
            layer_fluxes["layer_temp"] = layer_temp
            layer_fluxes["layer_precip"] = layer_precip
            per_layer_fluxes.append(layer_fluxes)
            layer_temps.append(layer_temp)
            layer_precips.append(layer_precip)

        aggregated_snow_fluxes = _aggregate_fluxes(per_layer_fluxes, layer_fractions)

    gr6j_state = GR6JState(
        production_store=state.production_store,
        routing_store=state.routing_store,
        exponential_store=state.exponential_store,
        uh1_states=state.uh1_states.copy(),
        uh2_states=state.uh2_states.copy(),
    )
    new_gr6j_state, gr6j_fluxes = gr6j_step(
        gr6j_state,
        params.to_gr6j(),
        aggregated_snow_fluxes["snow_pliq_and_melt"],
        pet,
        uh1_ordinates,
        uh2_ordinates,
    )

    new_state = State(
        production_store=new_gr6j_state.production_store,
        routing_store=new_gr6j_state.routing_store,
        exponential_store=new_gr6j_state.exponential_store,
        uh1_states=new_gr6j_state.uh1_states,
        uh2_states=new_gr6j_state.uh2_states,
        snow_layer_states=snow_layer_states,
    )

    combined_fluxes = {
        "precip_raw": precip,
        "snow_pliq": aggregated_snow_fluxes["snow_pliq"],
        "snow_psol": aggregated_snow_fluxes["snow_psol"],
        "snow_pack": aggregated_snow_fluxes["snow_pack"],
        "snow_thermal_state": aggregated_snow_fluxes["snow_thermal_state"],
        "snow_gratio": aggregated_snow_fluxes["snow_gratio"],
        "snow_pot_melt": aggregated_snow_fluxes["snow_pot_melt"],
        "snow_melt": aggregated_snow_fluxes["snow_melt"],
        "snow_pliq_and_melt": aggregated_snow_fluxes["snow_pliq_and_melt"],
        "snow_temp": aggregated_snow_fluxes["snow_temp"],
        **gr6j_fluxes,
    }
    return new_state, combined_fluxes, per_layer_fluxes


def run(
    params: Parameters,
    forcing: Forcing,
    *,
    catchment: Catchment,
    initial_state: State | None = None,
) -> ModelOutput[GR6JCemaNeigeFluxes]:
    """Run the coupled GR6J+CemaNeige model over a forcing timeseries."""
    if forcing.temp is None:
        raise ValueError("Temperature is required for GR6J+CemaNeige")

    state = State.initialize(params, catchment) if initial_state is None else initial_state
    uh1_ordinates, uh2_ordinates = compute_uh_ordinates(params.x4)

    layer_elevations: np.ndarray | None = None
    layer_fractions: np.ndarray | None = None
    if catchment.n_layers > 1 and catchment.hypsometric_curve is not None:
        layer_elevations, layer_fractions = derive_layers(catchment.hypsometric_curve, catchment.n_layers)

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
        state, combined_fluxes, per_layer_fluxes = step(
            state,
            params,
            float(precip),
            float(pet),
            float(temp),
            uh1_ordinates,
            uh2_ordinates,
            layer_elevations=layer_elevations,
            layer_fractions=layer_fractions,
            input_elevation=catchment.input_elevation,
            temp_gradient=catchment.temp_gradient,
            precip_gradient=catchment.precip_gradient,
            precip_gradient_type=catchment.precip_gradient_type.value,
        )
        combined_rows.append(combined_fluxes)
        snow_rows.append(
            {
                "precip_raw": float(precip),
                "snow_pliq": combined_fluxes["snow_pliq"],
                "snow_psol": combined_fluxes["snow_psol"],
                "snow_pack": combined_fluxes["snow_pack"],
                "snow_thermal_state": combined_fluxes["snow_thermal_state"],
                "snow_gratio": combined_fluxes["snow_gratio"],
                "snow_pot_melt": combined_fluxes["snow_pot_melt"],
                "snow_melt": combined_fluxes["snow_melt"],
                "snow_pliq_and_melt": combined_fluxes["snow_pliq_and_melt"],
                "snow_temp": combined_fluxes["snow_temp"],
                "snow_gthreshold": state.snow_layer_states[:, 2].mean(),
                "snow_glocalmax": state.snow_layer_states[:, 3].mean(),
            }
        )

        if per_layer_fluxes is not None:
            for key in layer_pack:
                layer_pack[key].append(np.array([layer_flux[key] for layer_flux in per_layer_fluxes], dtype=np.float64))

    flux_arrays = {
        key: np.array([row[key] for row in combined_rows], dtype=np.float64)
        for key in GR6JCemaNeigeFluxes.__dataclass_fields__
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
        fluxes=GR6JCemaNeigeFluxes(**flux_arrays),
        snow=SnowOutput(**snow_arrays),
        snow_layers=snow_layers,
    )


__all__ = ["GR6JCemaNeigeFluxes", "PARAM_NAMES", "Parameters", "State", "compute_state_size", "run", "step"]
