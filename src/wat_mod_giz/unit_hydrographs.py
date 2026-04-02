"""Unit hydrograph utilities for GR6J routing."""

from __future__ import annotations

import numpy as np

NH: int = 20
D: float = 2.5


def compute_uh_ordinates(x4: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute UH1 and UH2 ordinates for a GR6J time constant."""
    uh1_ordinates = np.zeros(NH, dtype=np.float64)
    uh2_ordinates = np.zeros(2 * NH, dtype=np.float64)

    for i in range(1, NH + 1):
        uh1_ordinates[i - 1] = _ss1(i, x4) - _ss1(i - 1, x4)

    for i in range(1, 2 * NH + 1):
        uh2_ordinates[i - 1] = _ss2(i, x4) - _ss2(i - 1, x4)

    return uh1_ordinates, uh2_ordinates


def convolve_uh(uh_states: np.ndarray, pr_input: float, uh_ordinates: np.ndarray) -> tuple[np.ndarray, float]:
    """Perform one timestep of unit hydrograph convolution."""
    output = float(uh_states[0])
    new_states = np.zeros_like(uh_states)

    for k in range(len(uh_states) - 1):
        new_states[k] = uh_states[k + 1] + uh_ordinates[k] * pr_input

    new_states[-1] = uh_ordinates[-1] * pr_input
    return new_states, output


def _ss1(i: float, x4: float) -> float:
    """Compute the UH1 S-curve value at position ``i``."""
    if i <= 0:
        return 0.0
    if i < x4:
        return (i / x4) ** D
    return 1.0


def _ss2(i: float, x4: float) -> float:
    """Compute the UH2 S-curve value at position ``i``."""
    if i <= 0:
        return 0.0
    if i <= x4:
        return 0.5 * (i / x4) ** D
    if i < 2 * x4:
        return 1.0 - 0.5 * (2.0 - i / x4) ** D
    return 1.0
