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
2. conditional logic
3. loops and comprehensions
4. functions and basic typing
5. classes or dataclasses
6. transition from notebook code to `.py` modules

Important teaching decision:

- notebook 02 focuses on conditional logic
- notebook 03 focuses on `for` loops first, then list comprehension, with only
  a light preview of dictionary comprehension
- notebook 04 introduces functions first, then basic built-in type hints
- notebook 04 ends with a guided data-cleaning challenge using simple negative
  bad-data rules
- classes stay for later notebooks

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

## Notebook 02 Decision

- filename: `02_conditional_logic.ipynb`
- audience: true beginners
- expected length: one 90-minute lecture
- focus: comparisons, booleans, and conditional branches
- libraries: stdlib only
- exercises: short and interleaved with the teaching cells

Notebook 02 covers:

- boolean results from comparisons
- `>`
- `<`
- `>=`
- `<=`
- `==`
- `if`
- `elif`
- `else`
- the difference between assignment `=` and comparison `==`

Notebook 02 does not cover:

- loops
- functions
- classes
- pandas
- numpy
- type annotations

## Notebook 03 Decision

- filename: `03_loops_and_comprehensions.ipynb`
- audience: true beginners
- expected length: one 90-minute lecture
- focus: looping over lists, counting, accumulation, and a first introduction
  to comprehensions
- libraries: stdlib only
- exercises: short and interleaved with the teaching cells

Notebook 03 covers:

- `for` loops over lists
- running totals
- counting with `if` inside a loop
- building a list with `append()`
- simple list comprehensions
- a very light introduction to dictionary comprehensions

Notebook 03 does not cover:

- nested loops
- `while` loops
- functions
- classes
- pandas
- numpy
- type annotations

## Notebook 04 Decision

- filename: `04_functions_and_typing.ipynb`
- audience: true beginners
- expected length: one 90-minute lecture
- focus: reusable functions, basic built-in type hints, and a guided
  multi-function mini-workflow
- libraries: stdlib only
- exercises: short and interleaved with the teaching cells

Notebook 04 covers:

- `def`
- parameters
- `return`
- functions with one input and one output
- functions with two inputs
- functions with conditionals inside
- functions with loops inside
- basic type hints with built-in types such as `float`, `int`,
  `list[float]`, and `dict[str, float]`
- a guided challenge that cleans simple bad data and computes precipitation and
  runoff summaries

Notebook 04 does not cover:

- classes
- `typing` imports
- advanced type syntax
- pandas
- numpy
- package-style `.py` modules

## Open Decisions For Later

- total number of notebooks in the basics block
- exact scope of notebook 05
- when pandas and numpy first appear
- when the runoff coefficient model is first introduced explicitly
- when the first package-style `.py` file is introduced
