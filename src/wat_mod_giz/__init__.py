"""Public package surface for wat_mod_giz."""

from wat_mod_giz.forcing import Forcing
from wat_mod_giz.outputs import ModelOutput
from wat_mod_giz.streamflow import StreamflowSeries
from wat_mod_giz.types import Catchment, PrecipGradientType, Resolution

__all__ = [
    "Catchment",
    "Forcing",
    "ModelOutput",
    "PrecipGradientType",
    "Resolution",
    "StreamflowSeries",
]

__version__ = "0.1.13"
