"""Pure glacier melt process equations."""

from __future__ import annotations


def compute_ice_melt(swe: float, temp: float, fi: float, tm: float, swe_th: float) -> float:
    """Compute raw glacier ice melt before glacier-fraction scaling."""
    if swe <= swe_th and temp > tm:
        return fi * (temp - tm)
    return 0.0


def compute_layer_glacier_melt(ice_melt: float, glacier_fraction: float) -> float:
    """Scale raw ice melt by the glacier fraction of a layer."""
    return ice_melt * glacier_fraction
