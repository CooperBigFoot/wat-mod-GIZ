# API Shape

## Top-Level Imports

The shared package surface should stay small:

```python
from wat_mod_giz import Catchment, Forcing, ModelOutput, PrecipGradientType, Resolution
from wat_mod_giz import StreamflowSeries
```

Model-specific imports should come from `wat_mod_giz.models`:

```python
from wat_mod_giz.models.gr6j import Parameters, State, run, step
from wat_mod_giz.models.gr6j_cemaneige import Parameters, State, run, step
from wat_mod_giz.models.gr6j_cemaneige_glacier import Parameters, State, run, step
```

## User-Facing Direction

- The API uses validated typed containers, not DataFrame-first inputs.
- Shared contracts live in `forcing.py`, `types.py`, and `outputs.py`.
- Shared time-series contracts live in `forcing.py` and `streamflow.py`.
- Pure process math lives under `processes/`.
- End users should mostly interact with `models/...py`.

## Intended Usage Style

The common flow is:

1. Build a `Forcing` object with aligned `time`, `precip`, `pet`, and optional `temp`.
2. Build a `Catchment` with only the metadata required by the chosen model.
3. Instantiate the model-specific `Parameters`.
4. Call `run(...)` for a full simulation or `step(...)` for a single timestep.
5. For calibration, build a `StreamflowSeries` aligned to the post-warmup period and pass it to `calibrate(...)`.

Illustrative shape:

```python
import numpy as np

from wat_mod_giz import Catchment, Forcing
from wat_mod_giz.models.gr6j_cemaneige_glacier import Parameters, run

forcing = Forcing(
    time=np.arange("2020-01-01", "2020-01-06", dtype="datetime64[D]"),
    precip=np.array([10.0, 5.0, 0.0, 7.0, 4.0]),
    pet=np.array([2.0, 2.5, 3.0, 2.0, 1.5]),
    temp=np.array([-3.0, 0.0, 4.0, 1.0, 5.0]),
)

catchment = Catchment(
    mean_annual_solid_precip=180.0,
    n_layers=3,
)

params = Parameters(...)
result = run(params=params, forcing=forcing, catchment=catchment)
```

## Stability Rules

- Preserve `pydrology` parameter names and ordering.
- Preserve `pydrology` state packing and initialization semantics.
- Keep the end-user surface smaller than `pydrology` by omitting registry and backend concerns.
