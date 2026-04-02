"""Tests for model-local calibration wrappers."""

import numpy as np
import pytest

from wat_mod_giz.calibration.engine import CalibrationSpec, _validate_bounds
from wat_mod_giz.calibration.metrics import nse
from wat_mod_giz.calibration.types import CalibrationResult
from wat_mod_giz.forcing import Forcing
from wat_mod_giz.models import gr6j, gr6j_cemaneige, gr6j_cemaneige_glacier
from wat_mod_giz.types import Catchment


@pytest.fixture
def gr6j_forcing() -> Forcing:
    return Forcing(
        time=np.arange("2020-01-01", "2020-01-19", dtype="datetime64[D]"),
        precip=np.array([12.0, 4.0, 0.0, 7.0, 2.0, 3.0, 11.0, 6.0, 0.0, 8.0, 5.0, 1.0, 9.0, 2.0, 0.0, 4.0, 10.0, 3.0]),
        pet=np.array([2.0, 2.5, 3.0, 2.2, 2.4, 2.7, 2.1, 2.3, 3.1, 2.2, 2.4, 2.8, 2.0, 2.5, 3.0, 2.7, 2.1, 2.3]),
    )


@pytest.fixture
def snow_forcing() -> Forcing:
    return Forcing(
        time=np.arange("2020-01-01", "2020-01-19", dtype="datetime64[D]"),
        precip=np.array([12.0, 4.0, 0.0, 7.0, 2.0, 3.0, 11.0, 6.0, 0.0, 8.0, 5.0, 1.0, 9.0, 2.0, 0.0, 4.0, 10.0, 3.0]),
        pet=np.array([2.0, 2.5, 3.0, 2.2, 2.4, 2.7, 2.1, 2.3, 3.1, 2.2, 2.4, 2.8, 2.0, 2.5, 3.0, 2.7, 2.1, 2.3]),
        temp=np.array([-3.0, -1.0, 0.0, 2.0, -2.0, 3.0, 1.0, 4.0, 5.0, 0.0, -1.0, 2.0, 4.0, 3.0, -2.0, 1.0, 5.0, 2.0]),
    )


@pytest.fixture
def snow_catchment() -> Catchment:
    return Catchment(mean_annual_solid_precip=150.0)


@pytest.fixture
def glacier_catchment() -> Catchment:
    return Catchment(mean_annual_solid_precip=150.0, glacier_fractions=np.array([0.25]))


def _synthetic_observed(output_streamflow: np.ndarray, warmup: int) -> np.ndarray:
    return output_streamflow[warmup:].copy()


class TestValidateBounds:
    def test_missing_bounds_raise(self) -> None:
        spec = CalibrationSpec(
            model_name="gr6j",
            parameter_names=gr6j.PARAM_NAMES,
            default_bounds=gr6j.DEFAULT_BOUNDS,
            parameters_type=gr6j.Parameters,
            run_model=gr6j.run,
        )
        with pytest.raises(ValueError, match="Missing bounds"):
            _validate_bounds({"x1": (1.0, 2.0)}, use_default_bounds=False, spec=spec)

    def test_default_bounds_are_used(self) -> None:
        spec = CalibrationSpec(
            model_name="gr6j",
            parameter_names=gr6j.PARAM_NAMES,
            default_bounds=gr6j.DEFAULT_BOUNDS,
            parameters_type=gr6j.Parameters,
            run_model=gr6j.run,
        )
        assert _validate_bounds(None, use_default_bounds=True, spec=spec) == gr6j.DEFAULT_BOUNDS


class TestGR6JCalibration:
    def test_returns_typed_result_and_respects_bounds(self, gr6j_forcing: Forcing) -> None:
        warmup = 5
        true_params = gr6j.Parameters(x1=350.0, x2=0.0, x3=90.0, x4=1.7, x5=0.0, x6=5.0)
        observed = _synthetic_observed(gr6j.run(true_params, gr6j_forcing).streamflow, warmup)

        result = gr6j.calibrate(
            forcing=gr6j_forcing,
            observed_streamflow=observed,
            warmup=warmup,
            population_size=10,
            generations=3,
            seed=42,
            progress=False,
            return_diagnostics=True,
        )

        assert isinstance(result, CalibrationResult)
        assert isinstance(result.parameters, gr6j.Parameters)
        assert result.objective == "nse"
        assert result.output.streamflow.shape == gr6j_forcing.precip.shape
        assert result.score["nse"] == pytest.approx(nse(observed, result.output.streamflow[warmup:]))
        params_array = np.asarray(result.parameters)
        lower = np.array([gr6j.DEFAULT_BOUNDS[name][0] for name in gr6j.PARAM_NAMES])
        upper = np.array([gr6j.DEFAULT_BOUNDS[name][1] for name in gr6j.PARAM_NAMES])
        assert np.all(params_array >= lower)
        assert np.all(params_array <= upper)
        assert result.diagnostics is not None
        assert len(result.diagnostics.best_score_history) == 3

    def test_same_seed_is_reproducible(self, gr6j_forcing: Forcing) -> None:
        warmup = 5
        true_params = gr6j.Parameters(x1=320.0, x2=0.2, x3=95.0, x4=1.9, x5=-0.2, x6=4.5)
        observed = _synthetic_observed(gr6j.run(true_params, gr6j_forcing).streamflow, warmup)

        kwargs = {
            "forcing": gr6j_forcing,
            "observed_streamflow": observed,
            "warmup": warmup,
            "population_size": 10,
            "generations": 3,
            "seed": 123,
            "progress": False,
        }
        result_a = gr6j.calibrate(**kwargs)
        result_b = gr6j.calibrate(**kwargs)

        np.testing.assert_allclose(np.asarray(result_a.parameters), np.asarray(result_b.parameters))
        assert result_a.score == result_b.score

    def test_progress_flag_does_not_change_result(self, gr6j_forcing: Forcing) -> None:
        warmup = 5
        true_params = gr6j.Parameters(x1=320.0, x2=0.2, x3=95.0, x4=1.9, x5=-0.2, x6=4.5)
        observed = _synthetic_observed(gr6j.run(true_params, gr6j_forcing).streamflow, warmup)

        result_with_progress = gr6j.calibrate(
            forcing=gr6j_forcing,
            observed_streamflow=observed,
            warmup=warmup,
            population_size=10,
            generations=3,
            seed=123,
            progress=True,
        )
        result_without_progress = gr6j.calibrate(
            forcing=gr6j_forcing,
            observed_streamflow=observed,
            warmup=warmup,
            population_size=10,
            generations=3,
            seed=123,
            progress=False,
        )

        np.testing.assert_allclose(
            np.asarray(result_with_progress.parameters), np.asarray(result_without_progress.parameters)
        )
        assert result_with_progress.score == result_without_progress.score


class TestSnowAndGlacierCalibration:
    def test_cemaneige_requires_temperature(self, gr6j_forcing: Forcing, snow_catchment: Catchment) -> None:
        observed = np.ones(len(gr6j_forcing) - 5, dtype=np.float64)
        with pytest.raises(ValueError, match="Temperature is required"):
            gr6j_cemaneige.calibrate(
                forcing=gr6j_forcing,
                observed_streamflow=observed,
                catchment=snow_catchment,
                warmup=5,
                population_size=10,
                generations=3,
                seed=42,
                progress=False,
            )

    def test_cemaneige_calibration_runs(self, snow_forcing: Forcing, snow_catchment: Catchment) -> None:
        warmup = 5
        true_params = gr6j_cemaneige.Parameters(x1=350.0, x2=0.0, x3=90.0, x4=1.7, x5=0.0, x6=5.0, ctg=0.97, kf=2.5)
        observed = _synthetic_observed(
            gr6j_cemaneige.run(true_params, snow_forcing, catchment=snow_catchment).streamflow,
            warmup,
        )

        result = gr6j_cemaneige.calibrate(
            forcing=snow_forcing,
            observed_streamflow=observed,
            catchment=snow_catchment,
            warmup=warmup,
            population_size=10,
            generations=3,
            seed=42,
            progress=False,
        )

        assert isinstance(result.parameters, gr6j_cemaneige.Parameters)
        assert "nse" in result.score

    def test_glacier_calibration_runs(self, snow_forcing: Forcing, glacier_catchment: Catchment) -> None:
        warmup = 5
        true_params = gr6j_cemaneige_glacier.Parameters(
            x1=350.0,
            x2=0.0,
            x3=90.0,
            x4=1.7,
            x5=0.0,
            x6=5.0,
            ctg=0.97,
            kf=2.5,
            fi=5.0,
            tm=0.0,
            swe_th=10.0,
        )
        observed = _synthetic_observed(
            gr6j_cemaneige_glacier.run(true_params, snow_forcing, catchment=glacier_catchment).streamflow,
            warmup,
        )

        result = gr6j_cemaneige_glacier.calibrate(
            forcing=snow_forcing,
            observed_streamflow=observed,
            catchment=glacier_catchment,
            warmup=warmup,
            population_size=10,
            generations=3,
            seed=42,
            progress=False,
        )

        assert isinstance(result.parameters, gr6j_cemaneige_glacier.Parameters)
        assert "nse" in result.score
