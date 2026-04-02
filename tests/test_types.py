"""Tests for shared forcing, streamflow, and catchment contracts."""

import numpy as np
import pytest

from wat_mod_giz.forcing import Forcing
from wat_mod_giz.streamflow import StreamflowSeries
from wat_mod_giz.types import Catchment, PrecipGradientType, Resolution


class TestForcing:
    """Tests for the array-first forcing container."""

    def test_valid_forcing_can_be_created(self) -> None:
        forcing = Forcing(
            time=np.arange("2020-01-01", "2020-01-06", dtype="datetime64[D]"),
            precip=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            pet=np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
        )

        assert len(forcing) == 5
        assert forcing.resolution == Resolution.daily
        assert forcing.time.dtype == np.dtype("datetime64[ns]")

    def test_temp_is_optional(self) -> None:
        forcing = Forcing(
            time=np.arange("2020-01-01", "2020-01-03", dtype="datetime64[D]"),
            precip=np.array([1.0, 2.0]),
            pet=np.array([0.5, 0.5]),
            temp=None,
        )

        assert forcing.temp is None

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="precip length"):
            Forcing(
                time=np.arange("2020-01-01", "2020-01-04", dtype="datetime64[D]"),
                precip=np.array([1.0, 2.0]),
                pet=np.array([0.5, 0.5, 0.5]),
            )

    def test_nan_values_raise(self) -> None:
        with pytest.raises(ValueError, match="precip array contains NaN values"):
            Forcing(
                time=np.arange("2020-01-01", "2020-01-03", dtype="datetime64[D]"),
                precip=np.array([1.0, np.nan]),
                pet=np.array([0.5, 0.5]),
            )

    def test_non_daily_spacing_raises(self) -> None:
        with pytest.raises(ValueError, match="time spacing"):
            Forcing(
                time=np.array(["2020-01-01", "2020-01-03"], dtype="datetime64[D]"),
                precip=np.array([1.0, 2.0]),
                pet=np.array([0.5, 0.5]),
            )


class TestStreamflowSeries:
    def test_valid_streamflow_series_can_be_created(self) -> None:
        observed = StreamflowSeries(
            time=np.arange("2020-01-01", "2020-01-06", dtype="datetime64[D]"),
            streamflow=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        )

        assert len(observed) == 5
        assert observed.resolution == Resolution.daily
        assert observed.time.dtype == np.dtype("datetime64[ns]")

    def test_streamflow_must_be_1d(self) -> None:
        with pytest.raises(ValueError, match="streamflow array must be 1D"):
            StreamflowSeries(
                time=np.arange("2020-01-01", "2020-01-03", dtype="datetime64[D]"),
                streamflow=np.array([[1.0, 2.0]]),
            )

    def test_time_must_be_1d(self) -> None:
        with pytest.raises(ValueError, match="time array must be 1D"):
            StreamflowSeries(
                time=np.array([["2020-01-01", "2020-01-02"]], dtype="datetime64[D]"),
                streamflow=np.array([1.0, 2.0]),
            )

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="streamflow length"):
            StreamflowSeries(
                time=np.arange("2020-01-01", "2020-01-04", dtype="datetime64[D]"),
                streamflow=np.array([1.0, 2.0]),
            )

    def test_nan_values_raise(self) -> None:
        with pytest.raises(ValueError, match="streamflow array contains NaN values"):
            StreamflowSeries(
                time=np.arange("2020-01-01", "2020-01-03", dtype="datetime64[D]"),
                streamflow=np.array([1.0, np.nan]),
            )

    def test_non_daily_spacing_raises(self) -> None:
        with pytest.raises(ValueError, match="time spacing"):
            StreamflowSeries(
                time=np.array(["2020-01-01", "2020-01-03"], dtype="datetime64[D]"),
                streamflow=np.array([1.0, 2.0]),
            )


class TestCatchment:
    """Tests for static catchment metadata."""

    def test_defaults_are_single_layer_and_exponential(self) -> None:
        catchment = Catchment(mean_annual_solid_precip=150.0)

        assert catchment.n_layers == 1
        assert catchment.precip_gradient_type == PrecipGradientType.exponential
        assert catchment.glacier_fractions is None

    def test_multi_layer_requires_hypsometric_curve(self) -> None:
        with pytest.raises(ValueError, match="hypsometric_curve"):
            Catchment(
                mean_annual_solid_precip=150.0,
                n_layers=3,
                input_elevation=500.0,
            )

    def test_multi_layer_requires_input_elevation(self) -> None:
        with pytest.raises(ValueError, match="input_elevation"):
            Catchment(
                mean_annual_solid_precip=150.0,
                n_layers=3,
                hypsometric_curve=np.linspace(100.0, 1000.0, 101),
            )

    def test_glacier_fractions_must_match_n_layers(self) -> None:
        with pytest.raises(ValueError, match="length must match n_layers"):
            Catchment(
                mean_annual_solid_precip=150.0,
                n_layers=2,
                hypsometric_curve=np.linspace(100.0, 1000.0, 101),
                input_elevation=500.0,
                glacier_fractions=np.array([0.1]),
            )

    def test_glacier_fractions_must_be_in_bounds(self) -> None:
        with pytest.raises(ValueError, match="values must be in \\[0, 1\\]"):
            Catchment(
                mean_annual_solid_precip=150.0,
                glacier_fractions=np.array([1.2]),
            )
