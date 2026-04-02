"""Pure CemaNeige process equations."""

from __future__ import annotations

T_MELT: float = 0.0
MIN_SPEED: float = 0.1
GTHRESHOLD_FACTOR: float = 0.9


def compute_solid_fraction(temp: float) -> float:
    """Compute the solid precipitation fraction from air temperature."""
    if temp <= -1.0:
        return 1.0
    if temp >= 3.0:
        return 0.0
    return (3.0 - temp) / 4.0


def partition_precipitation(precip: float, solid_fraction: float) -> tuple[float, float]:
    """Partition precipitation into liquid and solid components."""
    pliq = (1.0 - solid_fraction) * precip
    psol = solid_fraction * precip
    return pliq, psol


def update_thermal_state(etg: float, temp: float, ctg: float) -> float:
    """Update snow thermal state with exponential smoothing capped at zero."""
    return min(ctg * etg + (1.0 - ctg) * temp, 0.0)


def compute_potential_melt(etg: float, temp: float, kf: float, snow_pack: float) -> float:
    """Compute potential snowmelt from degree-day forcing."""
    if etg == 0.0 and temp > T_MELT:
        return min(kf * temp, snow_pack)
    return 0.0


def compute_gratio(snow_pack: float, gthreshold: float) -> float:
    """Compute catchment snow cover fraction."""
    if gthreshold == 0.0:
        return 0.0
    if snow_pack >= gthreshold:
        return 1.0
    return snow_pack / gthreshold


def compute_actual_melt(potential_melt: float, gratio: float) -> float:
    """Scale potential melt by snow cover fraction with a minimum melt speed."""
    return ((1.0 - MIN_SPEED) * gratio + MIN_SPEED) * potential_melt
