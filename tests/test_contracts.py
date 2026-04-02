"""Phase 1 contract tests for the shared package surface."""

from wat_mod_giz import Catchment, Forcing, ModelOutput, PrecipGradientType, Resolution, StreamflowSeries


class TestPublicPackageSurface:
    """Tests for the top-level package exports frozen in phase 1."""

    def test_top_level_imports_resolve(self) -> None:
        """The shared public imports resolve from the package root."""
        assert Catchment.__name__ == "Catchment"
        assert Forcing.__name__ == "Forcing"
        assert ModelOutput.__name__ == "ModelOutput"
        assert PrecipGradientType.__name__ == "PrecipGradientType"
        assert Resolution.__name__ == "Resolution"
        assert StreamflowSeries.__name__ == "StreamflowSeries"

    def test_public_exports_include_shared_contract_types(self) -> None:
        """The package __all__ exposes the shared phase-1 contract surface."""
        from wat_mod_giz import __all__

        assert __all__ == [
            "Catchment",
            "Forcing",
            "ModelOutput",
            "PrecipGradientType",
            "Resolution",
            "StreamflowSeries",
        ]


class TestSharedEnums:
    """Tests for enum values frozen in phase 1."""

    def test_resolution_values_are_lowercase(self) -> None:
        """Resolution enum values remain lowercase contract strings."""
        assert {member.value for member in Resolution} == {"daily"}

    def test_precip_gradient_type_values_are_lowercase(self) -> None:
        """Precipitation gradient enum values remain lowercase contract strings."""
        assert {member.value for member in PrecipGradientType} == {
            "exponential",
            "linear",
        }
