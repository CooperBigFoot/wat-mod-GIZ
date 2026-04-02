"""Tests for coupled GR6J+CemaNeige state and parameter types."""

import numpy as np
import pytest

from wat_mod_giz.models.gr6j_cemaneige import Parameters, State
from wat_mod_giz.types import Catchment


class TestParameters:
    def test_parameter_roundtrip(self) -> None:
        params = Parameters(x1=350.0, x2=0.0, x3=90.0, x4=1.7, x5=0.0, x6=5.0, ctg=0.97, kf=2.5)
        restored = Parameters.from_array(np.asarray(params))
        assert restored == params


class TestState:
    def test_initialize_single_layer(self) -> None:
        params = Parameters(x1=350.0, x2=0.0, x3=90.0, x4=1.7, x5=0.0, x6=5.0, ctg=0.97, kf=2.5)
        state = State.initialize(params, Catchment(mean_annual_solid_precip=150.0))

        assert state.n_layers == 1
        assert state.snow_layer_states.shape == (1, 4)
        assert state.snow_layer_states[0, 2] == pytest.approx(135.0)

    def test_initialize_multi_layer_scales_thresholds(self) -> None:
        params = Parameters(x1=350.0, x2=0.0, x3=90.0, x4=1.7, x5=0.0, x6=5.0, ctg=0.97, kf=2.5)
        catchment = Catchment(
            mean_annual_solid_precip=150.0,
            n_layers=3,
            hypsometric_curve=np.linspace(200.0, 2000.0, 101),
            input_elevation=500.0,
        )
        state = State.initialize(params, catchment)

        assert state.n_layers == 3
        assert not np.allclose(state.snow_layer_states[:, 2], state.snow_layer_states[0, 2])
