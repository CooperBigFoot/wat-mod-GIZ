# Python Basics Course Design

## Purpose

This notebook sequence is a pre-hydrology Python course for master's students.
Its purpose is to build enough Python fluency that students can later
implement simple hydrological models in a structured way.

The sequence should prepare students for the later hydrology course, where they
will work with real catchment data and more complete models.

## Agreed Direction

- Start with notebooks.
- Delay package-style `.py` work until the students have basic confidence.
- Do not require formal tests in the basics sequence.
- Keep examples hydrology-flavored, but do not overload notebook 01 with real
  data processing.
- Use a modelling progression that starts simple:
  - runoff coefficient model first
  - simple tank model second
  - richer improvements can be left to later notebooks or student extensions

## Framework To Teach

The goal is not just to teach Python syntax. The goal is to teach a modelling
framework that matches the structure used in this repo and related projects.

Core ideas:

- data lives in explicit variables and containers
- calculations belong in functions
- model parameters should be grouped together
- model state should be explicit
- `step()` simulates one timestep
- `run()` simulates a full timeseries

The teaching sequence should move toward that structure gradually rather than
introducing it all at once.

## Planned Topic Sequence

1. basic data structures
2. functions
3. basic typing
4. conditional logic
5. loops
6. classes or dataclasses
7. transition from notebook code to `.py` modules

Important teaching decision:

- type annotations begin in notebook 02, not notebook 01

## Notebook Format Rule

Every teaching notebook should mix explanation and action.

Preferred rhythm:

1. markdown explaining one concept
2. small code example
3. short exercise
4. check or discussion

Avoid:

- long uninterrupted markdown sections
- long uninterrupted code sections
- teaching syntax without a hydrology-flavored example

## Notebook 01 Decision

- filename: `01_variables_and_types.ipynb`
- audience: true beginners
- expected length: one 90-minute lecture
- focus: pure Python basics first
- libraries: stdlib only
- exercises: short and interleaved with the teaching cells

Notebook 01 covers:

- variables
- `int` and `float`
- `str` and `bool`
- `list`
- `dict`
- `type(...)`
- notebook execution order and reassignment

Notebook 01 does not cover:

- loops
- conditionals as a full topic
- functions
- classes
- pandas
- numpy
- type annotations
- real catchment data workflows

## Open Decisions For Later

- total number of notebooks in the basics block
- exact scope of notebook 02
- when pandas and numpy first appear
- when the runoff coefficient model is first introduced explicitly
- when the first package-style `.py` file is introduced
