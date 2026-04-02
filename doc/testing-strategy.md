# Testing Strategy

## Priority

The port is validated by behavioral parity with the Python-facing `pydrology`
contracts, using a simpler `bucket-model`-style test layout.

High-risk areas:

- parameter and state array packing order
- GR6J unit hydrograph state continuity
- CemaNeige multi-layer aggregation
- glacier exposure logic around `swe_th`
- coupled timestep ordering

## Planned Test Modules

- `tests/test_contracts.py`
  Confirms phase-1 package exports and shared enum values.
- `tests/test_types.py`
  Covers forcing and catchment contracts, validation rules, and shared type behavior.
- `tests/test_elevation.py`
  Covers layer derivation and temperature/precipitation extrapolation rules.
- `tests/test_gr6j_processes.py`
  Verifies pure GR6J equations against known behavioral expectations.
- `tests/test_gr6j_model.py`
  Verifies GR6J `step()` and `run()`, state continuity, and output field coverage.
- `tests/test_cemaneige_processes.py`
  Verifies snow partitioning, thermal state, melt, and aggregation rules.
- `tests/test_cemaneige_types.py`
  Verifies snow state initialization, array roundtrips, and multi-layer state handling.
- `tests/test_gr6j_cemaneige_model.py`
  Verifies snow-to-runoff coupling for single-layer and multi-layer runs.
- `tests/test_glacier_processes.py`
  Verifies glacier melt threshold logic and glacier-fraction scaling.
- `tests/test_gr6j_cemaneige_glacier_model.py`
  Verifies full coupled behavior and regression coverage for glacier-enabled runs.

## Test Style

- Prefer behavior-first tests over internal implementation checks.
- Reuse `pydrology` test scenarios where they define semantics clearly.
- Add explicit roundtrip tests for every array-backed parameter or state container.
- Keep process tests pure and focused.
- Keep model tests centered on `step()` and `run()` contracts.

## Verification Gates

Each phase should end with:

1. `uv run ruff check`
2. targeted `uv run pytest ...`
3. orchestrator review of `git diff`

The full suite becomes the release gate only after the coupled glacier model is in place.
