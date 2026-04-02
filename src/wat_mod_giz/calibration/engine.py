"""Shared calibration engine used by model-local wrapper functions."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from wat_mod_giz.calibration.metrics import get_metric
from wat_mod_giz.calibration.types import CalibrationDiagnostics, CalibrationResult, ObjectiveName
from wat_mod_giz.forcing import Forcing
from wat_mod_giz.streamflow import StreamflowSeries

_LARGE_PENALTY: float = 1.0e12


@dataclass(frozen=True)
class CalibrationSpec:
    """Model-specific hooks needed by the shared calibration engine."""

    model_name: str
    parameter_names: tuple[str, ...]
    default_bounds: dict[str, tuple[float, float]]
    parameters_type: type
    run_model: Callable[..., Any]


def _validate_bounds(
    bounds: dict[str, tuple[float, float]] | None,
    use_default_bounds: bool,
    spec: CalibrationSpec,
) -> dict[str, tuple[float, float]]:
    if bounds is None:
        if use_default_bounds:
            return spec.default_bounds
        raise ValueError("Must provide bounds or set use_default_bounds=True")

    missing = set(spec.parameter_names) - set(bounds)
    if missing:
        raise ValueError(f"Missing bounds for parameters: {sorted(missing)}")

    for name in spec.parameter_names:
        lower, upper = bounds[name]
        if lower >= upper:
            raise ValueError(f"Lower bound must be less than upper bound for '{name}': {lower} >= {upper}")
    return bounds


def _validate_observed(
    forcing: Forcing,
    observed: StreamflowSeries,
    warmup: int,
) -> np.ndarray:
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}")
    if warmup >= len(forcing):
        raise ValueError(f"warmup {warmup} must be smaller than forcing length {len(forcing)}")

    expected_length = len(forcing) - warmup
    if len(observed) != expected_length:
        raise ValueError(
            f"observed streamflow length {len(observed)} must equal forcing length {len(forcing)} minus warmup {warmup}"
        )

    expected_time = forcing.time[warmup:]
    if not np.array_equal(observed.time, expected_time):
        raise ValueError("observed time must match forcing time after warmup")

    return observed.streamflow


def _to_optimizer_value(score: float, direction: str) -> float:
    if not np.isfinite(score):
        return _LARGE_PENALTY
    if direction == "maximize":
        return -score
    return score


def _to_score_value(fitness: float, direction: str) -> float:
    if direction == "maximize":
        return -fitness
    return fitness


def calibrate_model(
    *,
    spec: CalibrationSpec,
    forcing: Forcing,
    observed: StreamflowSeries,
    objective: ObjectiveName,
    bounds: dict[str, tuple[float, float]] | None,
    use_default_bounds: bool,
    initial_state: Any | None,
    catchment: Any | None,
    warmup: int,
    population_size: int,
    generations: int,
    seed: int | None,
    progress: bool,
    return_diagnostics: bool,
    n_workers: int,
) -> CalibrationResult:
    """Run bounded single-objective calibration for one model."""
    try:
        from ctrl_freak import ga, polynomial_mutation, sbx_crossover
    except ImportError as exc:
        raise ImportError("Calibration requires ctrl-freak. Run `uv sync` to install local dependencies.") from exc

    observed_streamflow = _validate_observed(forcing, observed, warmup)
    resolved_bounds = _validate_bounds(bounds, use_default_bounds, spec)
    metric, direction = get_metric(objective)

    lower_bounds = np.array([resolved_bounds[name][0] for name in spec.parameter_names], dtype=np.float64)
    upper_bounds = np.array([resolved_bounds[name][1] for name in spec.parameter_names], dtype=np.float64)
    bound_arrays = (lower_bounds, upper_bounds)

    crossover = sbx_crossover(eta=15.0, bounds=bound_arrays, seed=seed)
    mutate = polynomial_mutation(eta=20.0, bounds=bound_arrays, seed=None if seed is None else seed + 1)

    best_fitness_history: list[float] = []
    best_score_history: list[float] = []
    should_render_progress = progress and sys.stderr.isatty()

    def init(rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(lower_bounds, upper_bounds)

    def evaluate(x: np.ndarray) -> float:
        params = spec.parameters_type.from_array(np.asarray(x, dtype=np.float64))
        run_kwargs: dict[str, Any] = {"params": params, "forcing": forcing, "initial_state": initial_state}
        if catchment is not None:
            run_kwargs["catchment"] = catchment
        output = spec.run_model(**run_kwargs)
        score = metric(observed_streamflow, output.streamflow[warmup:])
        return _to_optimizer_value(score, direction)

    def callback(result: Any, generation: int) -> bool:
        fitness = float(result.best[1])
        score = _to_score_value(fitness, direction)
        best_fitness_history.append(fitness)
        best_score_history.append(score)
        if should_render_progress:
            print(
                f"\rGeneration {generation + 1}/{generations}: best {objective}={score:.4f}",
                end="",
                file=sys.stderr,
                flush=True,
            )
        return False

    result = ga(
        init=init,
        evaluate=evaluate,
        crossover=crossover,
        mutate=mutate,
        pop_size=population_size,
        n_generations=generations,
        seed=seed,
        callback=callback,
        n_workers=n_workers,
    )

    if should_render_progress:
        print(file=sys.stderr)

    best_params = spec.parameters_type.from_array(np.asarray(result.best[0], dtype=np.float64))
    best_output_kwargs: dict[str, Any] = {"params": best_params, "forcing": forcing, "initial_state": initial_state}
    if catchment is not None:
        best_output_kwargs["catchment"] = catchment
    best_output = spec.run_model(**best_output_kwargs)
    best_score = metric(observed_streamflow, best_output.streamflow[warmup:])

    diagnostics = None
    if return_diagnostics:
        diagnostics = CalibrationDiagnostics(
            generations_completed=result.generations,
            evaluations=result.evaluations,
            best_fitness_history=np.asarray(best_fitness_history, dtype=np.float64),
            best_score_history=np.asarray(best_score_history, dtype=np.float64),
        )

    return CalibrationResult(
        model=spec.model_name,
        parameters=best_params,
        score={objective: float(best_score)},
        output=best_output,
        objective=objective,
        warmup=warmup,
        diagnostics=diagnostics,
    )
