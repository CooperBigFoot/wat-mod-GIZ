"""Tests for pure glacier melt processes."""

import pytest

from wat_mod_giz.processes.glacier import compute_ice_melt, compute_layer_glacier_melt


class TestComputeIceMelt:
    def test_no_melt_when_snow_covers_glacier(self) -> None:
        assert compute_ice_melt(50.0, 10.0, 8.0, 0.0, 40.0) == 0.0

    def test_no_melt_when_cold(self) -> None:
        assert compute_ice_melt(0.0, -5.0, 8.0, 0.0, 10.0) == 0.0

    def test_melt_when_conditions_met(self) -> None:
        assert compute_ice_melt(0.0, 5.0, 8.0, 0.0, 10.0) == pytest.approx(40.0)


class TestComputeLayerGlacierMelt:
    def test_scales_by_glacier_fraction(self) -> None:
        assert compute_layer_glacier_melt(40.0, 0.3) == pytest.approx(12.0)
