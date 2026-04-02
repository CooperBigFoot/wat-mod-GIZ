"""Tests for shared elevation helper functions."""

import numpy as np
import pytest

from wat_mod_giz.elevation import (
    GRAD_P_DEFAULT,
    GRAD_T_DEFAULT,
    derive_layers,
    extrapolate_precipitation,
    extrapolate_precipitation_linear,
    extrapolate_temperature,
)


class TestDeriveLayers:
    """Tests for equal-area layer derivation."""

    def test_uniform_hypsometric_curve_produces_uniform_elevations(self) -> None:
        elevations, fractions = derive_layers(np.full(101, 500.0), n_layers=5)

        np.testing.assert_allclose(elevations, 500.0)
        np.testing.assert_allclose(fractions, 0.2)

    def test_fractions_sum_to_one(self) -> None:
        _, fractions = derive_layers(np.linspace(100.0, 2000.0, 101), n_layers=3)
        assert fractions.sum() == pytest.approx(1.0)

    def test_two_layers_split_at_midpoint(self) -> None:
        elevations, fractions = derive_layers(np.linspace(0.0, 2000.0, 101), n_layers=2)

        np.testing.assert_allclose(elevations, np.array([500.0, 1500.0]))
        np.testing.assert_allclose(fractions, np.array([0.5, 0.5]))


class TestExtrapolateTemperature:
    """Tests for lapse-rate temperature extrapolation."""

    def test_default_gradient_constant_matches_contract(self) -> None:
        assert GRAD_T_DEFAULT == 0.6

    def test_same_elevation_returns_input(self) -> None:
        assert extrapolate_temperature(15.0, 500.0, 500.0) == pytest.approx(15.0)

    def test_higher_elevation_is_colder(self) -> None:
        assert extrapolate_temperature(15.0, 500.0, 1500.0) == pytest.approx(9.0)


class TestExtrapolatePrecipitation:
    """Tests for precipitation extrapolation."""

    def test_default_exponential_gradient_constant_matches_contract(self) -> None:
        assert GRAD_P_DEFAULT == 0.00041

    def test_same_elevation_returns_input(self) -> None:
        assert extrapolate_precipitation(10.0, 500.0, 500.0) == pytest.approx(10.0)

    def test_higher_elevation_increases_precipitation(self) -> None:
        assert extrapolate_precipitation(10.0, 500.0, 1500.0) > 10.0

    def test_linear_gradient_clamps_to_zero(self) -> None:
        assert extrapolate_precipitation_linear(10.0, 4000.0, 0.0, gradient=0.001) == 0.0

    def test_linear_gradient_same_elevation_returns_input(self) -> None:
        assert extrapolate_precipitation_linear(10.0, 500.0, 500.0) == pytest.approx(10.0)
