"""Typed calibration results and shared calibration literals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

ObjectiveName = Literal["nse", "kge", "log_nse", "rmse", "mae", "pbias"]


@dataclass(frozen=True)
class CalibrationDiagnostics:
    """Optional optimizer diagnostics for teaching and debugging."""

    generations_completed: int
    evaluations: int
    best_fitness_history: np.ndarray
    best_score_history: np.ndarray


@dataclass(frozen=True)
class CalibrationResult:
    """Best-fit calibration result for one model and one objective."""

    model: str
    parameters: Any
    score: dict[str, float]
    output: Any
    objective: ObjectiveName
    warmup: int
    diagnostics: CalibrationDiagnostics | None = None
