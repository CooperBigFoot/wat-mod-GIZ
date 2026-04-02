"""Tests for the coupled GR6J+CemaNeige+glacier model."""

import numpy as np
import pytest

from wat_mod_giz.forcing import Forcing
from wat_mod_giz.models import gr6j_cemaneige
from wat_mod_giz.models.gr6j_cemaneige_glacier import GR6JCemaNeigeGlacierFluxes, Parameters, State, run, step
from wat_mod_giz.outputs import ModelOutput
from wat_mod_giz.types import Catchment
from wat_mod_giz.unit_hydrographs import compute_uh_ordinates


@pytest.fixture
def typical_params() -> Parameters:
    return Parameters(x1=350.0, x2=0.0, x3=90.0, x4=1.7, x5=0.0, x6=5.0, ctg=0.97, kf=2.5, fi=5.0, tm=0.0, swe_th=10.0)


@pytest.fixture
def glacier_catchment() -> Catchment:
    return Catchment(mean_annual_solid_precip=150.0, glacier_fractions=np.array([0.3]))


@pytest.fixture
def forcing_with_temp() -> Forcing:
    return Forcing(
        time=np.arange("2020-01-01", "2020-01-06", dtype="datetime64[D]"),
        precip=np.array([10.0, 5.0, 0.0, 15.0, 8.0]),
        pet=np.array([3.0, 4.0, 5.0, 3.5, 4.0]),
        temp=np.array([-5.0, 0.0, 5.0, -2.0, 8.0]),
    )


class TestRun:
    def test_returns_model_output(
        self, typical_params: Parameters, glacier_catchment: Catchment, forcing_with_temp: Forcing
    ) -> None:
        result = run(typical_params, forcing_with_temp, catchment=glacier_catchment)
        assert isinstance(result, ModelOutput)
        assert isinstance(result.fluxes, GR6JCemaNeigeGlacierFluxes)

    def test_glacier_melt_non_negative(
        self, typical_params: Parameters, glacier_catchment: Catchment, forcing_with_temp: Forcing
    ) -> None:
        result = run(typical_params, forcing_with_temp, catchment=glacier_catchment)
        assert np.all(result.fluxes.glacier_melt >= 0.0)

    def test_warm_temperatures_produce_glacier_melt(
        self, typical_params: Parameters, glacier_catchment: Catchment
    ) -> None:
        forcing = Forcing(
            time=np.arange("2020-07-01", "2020-07-06", dtype="datetime64[D]"),
            precip=np.zeros(5),
            pet=np.full(5, 5.0),
            temp=np.full(5, 10.0),
        )
        result = run(typical_params, forcing, catchment=glacier_catchment)
        assert result.fluxes.glacier_melt.sum() > 0.0

    def test_without_glacier_fractions_matches_cemaneige(
        self, typical_params: Parameters, forcing_with_temp: Forcing
    ) -> None:
        catchment = Catchment(mean_annual_solid_precip=150.0)
        glacier_result = run(typical_params, forcing_with_temp, catchment=catchment)
        base_params = gr6j_cemaneige.Parameters(
            x1=typical_params.x1,
            x2=typical_params.x2,
            x3=typical_params.x3,
            x4=typical_params.x4,
            x5=typical_params.x5,
            x6=typical_params.x6,
            ctg=typical_params.ctg,
            kf=typical_params.kf,
        )
        base_result = gr6j_cemaneige.run(base_params, forcing_with_temp, catchment=catchment)
        np.testing.assert_allclose(glacier_result.streamflow, base_result.streamflow)


class TestStep:
    def test_step_returns_glacier_keys(self, typical_params: Parameters, glacier_catchment: Catchment) -> None:
        state = State.initialize(
            gr6j_cemaneige.Parameters(
                x1=typical_params.x1,
                x2=typical_params.x2,
                x3=typical_params.x3,
                x4=typical_params.x4,
                x5=typical_params.x5,
                x6=typical_params.x6,
                ctg=typical_params.ctg,
                kf=typical_params.kf,
            ),
            glacier_catchment,
        )
        uh1, uh2 = compute_uh_ordinates(typical_params.x4)
        new_state, fluxes = step(state, typical_params, 10.0, 3.0, 5.0, uh1, uh2)

        assert isinstance(new_state, State)
        assert "glacier_melt" in fluxes
        assert "glacier_ice_melt" in fluxes
