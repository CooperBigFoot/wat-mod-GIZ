"""Tests for pure CemaNeige process equations."""

import pytest

from wat_mod_giz.processes.cemaneige import (
    compute_actual_melt,
    compute_gratio,
    compute_potential_melt,
    compute_solid_fraction,
    partition_precipitation,
    update_thermal_state,
)


class TestComputeSolidFraction:
    def test_all_snow_below_threshold(self) -> None:
        assert compute_solid_fraction(-5.0) == 1.0

    def test_all_rain_above_threshold(self) -> None:
        assert compute_solid_fraction(5.0) == 0.0

    def test_linear_transition(self) -> None:
        assert compute_solid_fraction(1.0) == pytest.approx(0.5)


class TestPartitionPrecipitation:
    def test_mass_is_conserved(self) -> None:
        pliq, psol = partition_precipitation(20.0, 0.25)
        assert pliq + psol == pytest.approx(20.0)


class TestUpdateThermalState:
    def test_value_is_capped_at_zero(self) -> None:
        assert update_thermal_state(0.0, 10.0, 0.5) == 0.0


class TestComputePotentialMelt:
    def test_requires_zero_thermal_state_and_positive_temperature(self) -> None:
        assert compute_potential_melt(-1.0, 5.0, 2.5, 100.0) == 0.0
        assert compute_potential_melt(0.0, 5.0, 2.5, 100.0) == pytest.approx(12.5)


class TestComputeGratio:
    def test_gratio_is_bounded(self) -> None:
        assert compute_gratio(0.0, 100.0) == 0.0
        assert compute_gratio(100.0, 100.0) == 1.0


class TestComputeActualMelt:
    def test_minimum_speed_applies(self) -> None:
        assert compute_actual_melt(10.0, 0.0) == pytest.approx(1.0)
