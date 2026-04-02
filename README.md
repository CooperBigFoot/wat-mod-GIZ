# wat-mod-GIZ

Python-native hydrological models ported from `../pydrology`, structured in a style similar to `../bucket-model`.

Current scope:

- GR6J
- GR6J + CemaNeige
- GR6J + CemaNeige + glacier
- Single-objective calibration via `ctrl-freak`

This repo does not port the Rust/PyO3 backend from `pydrology`. The implementation here is pure Python with explicit process modules, typed inputs, and thin model orchestration.

## Package Shape

Core package exports:

- `wat_mod_giz.Catchment`
- `wat_mod_giz.Forcing`
- `wat_mod_giz.ModelOutput`
- `wat_mod_giz.PrecipGradientType`
- `wat_mod_giz.Resolution`

Model entry points:

- `wat_mod_giz.models.gr6j`
- `wat_mod_giz.models.gr6j_cemaneige`
- `wat_mod_giz.models.gr6j_cemaneige_glacier`

Supporting modules:

- `wat_mod_giz.processes`
- `wat_mod_giz.elevation`
- `wat_mod_giz.unit_hydrographs`
- `wat_mod_giz.outputs`
- `wat_mod_giz.types`

Planning and contract notes are in [`doc/`](doc/).

## Example

```python
import numpy as np

from wat_mod_giz import Catchment, Forcing
from wat_mod_giz.models.gr6j_cemaneige_glacier import Parameters, run

forcing = Forcing(
    time=np.arange("2020-01-01", "2020-01-11", dtype="datetime64[D]"),
    precip=np.array([12.0, 8.0, 0.0, 15.0, 6.0, 2.0, 0.0, 11.0, 9.0, 4.0]),
    pet=np.array([1.5, 1.8, 2.0, 1.2, 1.7, 2.1, 2.3, 1.9, 1.6, 1.4]),
    temp=np.array([-4.0, -2.0, 1.0, 3.0, -1.0, 4.0, 6.0, 2.0, 0.0, 5.0]),
)

catchment = Catchment(
    mean_annual_solid_precip=180.0,
    n_layers=5,
    hypsometric_curve=np.linspace(600.0, 3200.0, 101),
    input_elevation=1200.0,
    glacier_fractions=np.array([0.0, 0.0, 0.1, 0.25, 0.4]),
)

params = Parameters(
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

result = run(params=params, forcing=forcing, catchment=catchment)

print(result.streamflow)
print(result.fluxes.glacier_melt)
print(result.snow)
```

## Calibration

```python
from wat_mod_giz.models.gr6j import calibrate

result = calibrate(
    forcing=forcing,
    observed_streamflow=observed_streamflow,
    warmup=365,
    use_default_bounds=True,
    objective="nse",
    seed=42,
)

print(result.parameters)
print(result.score)
print(result.output.streamflow)
```

## Development

Install and sync with `uv`:

```bash
uv sync
```

Run tests:

```bash
uv run pytest
```

Run lint:

```bash
uv run ruff check src tests
```

Format:

```bash
uv run ruff format
```

## Notes

- Time series are daily.
- Multi-layer snow and glacier runs use catchment hypsometry plus forcing input elevation.
- Calibration is model-local and array-first, with `ctrl-freak` kept behind thin wrappers.
