"""Tests for the standalone GR6J model."""

import numpy as np
import pytest

from wat_mod_giz.forcing import Forcing
from wat_mod_giz.models.gr6j import GR6JFluxes, Parameters, State, run, step
from wat_mod_giz.outputs import ModelOutput
from wat_mod_giz.unit_hydrographs import compute_uh_ordinates

EXPECTED_FLUX_KEYS = {
    "pet",
    "precip",
    "production_store",
    "net_rainfall",
    "storage_infiltration",
    "actual_et",
    "percolation",
    "effective_rainfall",
    "q9",
    "q1",
    "routing_store",
    "exchange",
    "actual_exchange_routing",
    "actual_exchange_direct",
    "actual_exchange_total",
    "qr",
    "qrexp",
    "exponential_store",
    "qd",
    "streamflow",
}


@pytest.fixture
def typical_params() -> Parameters:
    return Parameters(x1=350.0, x2=0.0, x3=90.0, x4=1.7, x5=0.0, x6=5.0)


@pytest.fixture
def simple_forcing() -> Forcing:
    return Forcing(
        time=np.arange("2020-01-01", "2020-01-06", dtype="datetime64[D]"),
        precip=np.array([10.0, 5.0, 0.0, 15.0, 2.0]),
        pet=np.array([3.0, 4.0, 5.0, 2.0, 3.5]),
    )


class TestParametersAndState:
    def test_parameters_roundtrip(self, typical_params: Parameters) -> None:
        restored = Parameters.from_array(np.asarray(typical_params))
        assert restored == typical_params

    def test_state_initialize_and_roundtrip(self, typical_params: Parameters) -> None:
        state = State.initialize(typical_params)
        restored = State.from_array(np.asarray(state))

        assert restored.production_store == pytest.approx(0.3 * typical_params.x1)
        np.testing.assert_array_equal(restored.uh1_states, state.uh1_states)
        np.testing.assert_array_equal(restored.uh2_states, state.uh2_states)


class TestStep:
    def test_returns_new_state_and_fluxes(self, typical_params: Parameters) -> None:
        state = State.initialize(typical_params)
        uh1, uh2 = compute_uh_ordinates(typical_params.x4)

        new_state, fluxes = step(state, typical_params, 10.0, 3.0, uh1, uh2)

        assert isinstance(new_state, State)
        assert set(fluxes) == EXPECTED_FLUX_KEYS

    def test_state_is_not_mutated(self, typical_params: Parameters) -> None:
        state = State.initialize(typical_params)
        original = np.asarray(state).copy()
        uh1, uh2 = compute_uh_ordinates(typical_params.x4)

        step(state, typical_params, 10.0, 3.0, uh1, uh2)

        np.testing.assert_array_equal(np.asarray(state), original)

    def test_streamflow_is_non_negative(self, typical_params: Parameters) -> None:
        state = State.initialize(typical_params)
        uh1, uh2 = compute_uh_ordinates(typical_params.x4)

        for precip, pet in [(10.0, 3.0), (0.0, 5.0), (50.0, 0.0), (0.0, 0.0)]:
            state, fluxes = step(state, typical_params, precip, pet, uh1, uh2)
            assert fluxes["streamflow"] >= 0.0


class TestRun:
    def test_returns_model_output(self, typical_params: Parameters, simple_forcing: Forcing) -> None:
        result = run(typical_params, simple_forcing)

        assert isinstance(result, ModelOutput)
        assert isinstance(result.fluxes, GR6JFluxes)
        assert set(result.fluxes.to_dict()) == EXPECTED_FLUX_KEYS

    def test_output_length_matches_input(self, typical_params: Parameters, simple_forcing: Forcing) -> None:
        result = run(typical_params, simple_forcing)
        assert len(result) == len(simple_forcing)

    def test_custom_initial_state_changes_result(self, typical_params: Parameters, simple_forcing: Forcing) -> None:
        custom_state = State(
            production_store=200.0,
            routing_store=60.0,
            exponential_store=10.0,
            uh1_states=np.zeros(20),
            uh2_states=np.zeros(40),
        )

        result_custom = run(typical_params, simple_forcing, initial_state=custom_state)
        result_default = run(typical_params, simple_forcing)

        assert result_custom.streamflow[0] != result_default.streamflow[0]
