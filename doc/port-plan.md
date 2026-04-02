# Port Plan

## Goal

Port the pure-Python semantics of the `pydrology` GR6J, GR6J+CemaNeige, and
GR6J+CemaNeige+glacier models into `wat-mod-GIZ` while using a cleaner package
shape inspired by `bucket-model`.

## Scope

The first implementation pass includes:

- shared forcing, catchment, state, and output contracts
- GR6J process equations and orchestration
- CemaNeige snow processes, layer handling, and elevation utilities
- glacier melt processes and coupled orchestration
- tests and implementation-facing docs in `doc/`

The first implementation pass excludes:

- Rust or PyO3 integration
- registry machinery
- calibration
- generic framework abstractions not needed by the three model variants

## Target Package Layout

```text
src/wat_mod_giz/
  __init__.py
  forcing.py
  types.py
  outputs.py
  elevation.py
  unit_hydrographs.py
  processes/
    gr6j.py
    cemaneige.py
    glacier.py
  models/
    gr6j.py
    gr6j_cemaneige.py
    gr6j_cemaneige_glacier.py
```

## Implementation Order

1. Freeze public contracts and docs.
2. Implement shared foundations used by every model.
3. Port GR6J and CemaNeige in parallel once shared contracts are stable.
4. Integrate GR6J+CemaNeige.
5. Add glacier augmentation.
6. Finish verification, docs, and release workflow.

## Source of Truth

Behavioral parity is defined by the Python-facing `pydrology` contracts and its
tests, not by its internal Rust boundary. When `pydrology` docs and tests
conflict, tests win for implementation details and docs win for naming and
intended semantics.
