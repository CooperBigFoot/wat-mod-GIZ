# Model Contract

## API Direction

`wat-mod-GIZ` will expose an array-first API. It should feel simpler than
`pydrology`, but it should not switch to a DataFrame-first interface.

The stable user-facing concepts are:

- forcing containers for `time`, `precip`, `pet`, and optional `temp`
- catchment metadata needed for snow and glacier behavior
- parameter dataclasses with source-model parameter names
- mutable state dataclasses for stepwise execution
- typed output containers returned by `run()`

## Public Model Surface

Each model module under `src/wat_mod_giz/models/` must expose:

- `Parameters`
- `State`
- `run(...)`
- `step(...)`

The three supported model families are:

- `gr6j`
- `gr6j_cemaneige`
- `gr6j_cemaneige_glacier`

## Contract Preservation Rules

- Preserve `pydrology` parameter names and ordering.
- Preserve `pydrology` state layout and initialization rules.
- Preserve timestep ordering and coupling order.
- Preserve multi-layer snow semantics and glacier augmentation behavior.
- Do not preserve `pydrology` registry or backend integration details.

## Shared Types

Shared modules should define only the contracts needed by all three models:

- `forcing.py`: forcing container and resolution enum if needed
- `types.py`: catchment and shared enums
- `outputs.py`: common output wrapper and snow output containers
- `elevation.py`: extrapolation and layer derivation helpers
- `unit_hydrographs.py`: GR6J UH ordinates

## Coupling Order

The coupled model must follow this sequence:

1. read forcing for the timestep
2. run snow partitioning and melt
3. apply glacier melt after snow state is known
4. route resulting liquid input through GR6J

## Testing Standard

All public contracts must be backed by behavior-first tests. Roundtrip tests for
parameter and state containers are mandatory because array packing order is a
high-risk failure mode.
