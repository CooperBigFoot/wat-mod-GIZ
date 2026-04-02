"""Tests for the coupled GR6J+CemaNeige model."""

import numpy as np
import pytest

from wat_mod_giz.forcing import Forcing
from wat_mod_giz.models.gr6j_cemaneige import GR6JCemaNeigeFluxes, Parameters, State, run
from wat_mod_giz.outputs import ModelOutput, SnowLayerOutputs
from wat_mod_giz.types import Catchment


@pytest.fixture
def typical_params() -> Parameters:
    return Parameters(x1=350.0, x2=0.0, x3=90.0, x4=1.7, x5=0.0, x6=5.0, ctg=0.97, kf=2.5)


@pytest.fixture
def forcing_with_temp() -> Forcing:
    return Forcing(
        time=np.arange("2020-01-01", "2020-01-06", dtype="datetime64[D]"),
        precip=np.array([10.0, 5.0, 0.0, 15.0, 8.0]),
        pet=np.array([3.0, 4.0, 5.0, 3.5, 4.0]),
        temp=np.array([-5.0, 0.0, 5.0, -2.0, 8.0]),
    )


class TestRunWithSnow:
    def test_returns_model_output_with_snow(self, typical_params: Parameters, forcing_with_temp: Forcing) -> None:
        result = run(typical_params, forcing_with_temp, catchment=Catchment(mean_annual_solid_precip=150.0))

        assert isinstance(result, ModelOutput)
        assert isinstance(result.fluxes, GR6JCemaNeigeFluxes)
        assert result.snow is not None

    def test_requires_temperature(self, typical_params: Parameters) -> None:
        forcing = Forcing(
            time=np.arange("2020-01-01", "2020-01-04", dtype="datetime64[D]"),
            precip=np.array([10.0, 5.0, 8.0]),
            pet=np.array([3.0, 4.0, 4.0]),
        )
        with pytest.raises(ValueError, match="Temperature"):
            run(typical_params, forcing, catchment=Catchment(mean_annual_solid_precip=150.0))

    def test_cold_period_accumulates_snow(self, typical_params: Parameters) -> None:
        forcing = Forcing(
            time=np.arange("2020-01-01", "2020-01-04", dtype="datetime64[D]"),
            precip=np.array([10.0, 10.0, 10.0]),
            pet=np.array([1.0, 1.0, 1.0]),
            temp=np.array([-10.0, -10.0, -10.0]),
        )
        result = run(typical_params, forcing, catchment=Catchment(mean_annual_solid_precip=150.0))

        assert result.snow is not None
        assert result.snow.snow_pack[-1] > result.snow.snow_pack[0]
        np.testing.assert_allclose(result.snow.snow_psol, forcing.precip)

    def test_custom_initial_state_changes_snow_result(
        self, typical_params: Parameters, forcing_with_temp: Forcing
    ) -> None:
        catchment = Catchment(mean_annual_solid_precip=150.0)
        custom_state = State.initialize(typical_params, catchment)
        custom_state.snow_layer_states[0, 0] = 100.0

        result_custom = run(typical_params, forcing_with_temp, catchment=catchment, initial_state=custom_state)
        result_default = run(typical_params, forcing_with_temp, catchment=catchment)

        assert result_custom.snow is not None
        assert result_default.snow is not None
        assert result_custom.snow.snow_pack[0] != result_default.snow.snow_pack[0]


class TestRunWithMultiLayerSnow:
    def test_multi_layer_run_populates_snow_layers(
        self, typical_params: Parameters, forcing_with_temp: Forcing
    ) -> None:
        catchment = Catchment(
            mean_annual_solid_precip=150.0,
            n_layers=3,
            hypsometric_curve=np.linspace(200.0, 2000.0, 101),
            input_elevation=500.0,
        )
        result = run(typical_params, forcing_with_temp, catchment=catchment)

        assert isinstance(result.snow_layers, SnowLayerOutputs)
        assert result.snow_layers.snow_pack.shape == (len(forcing_with_temp), 3)
