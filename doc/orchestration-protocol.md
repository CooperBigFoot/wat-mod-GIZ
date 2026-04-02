# Orchestration Protocol

This repository is being implemented under an orchestrated sub-agent workflow.

## Hard Rules

- The Orchestrator must never write code directly.
- Sub-agents must never commit or push code.
- Only the Orchestrator may handle version bumping, commit, tag, push, and
  `clog log`.

## Responsibilities

### Explore

Before each phase, the Orchestrator re-reads the relevant local source and
target files so sub-agent prompts are based on current repository state.

### Plan

Work is split into dependency-aware phases. Parallelism is allowed only when
agents work on orthogonal files or on code behind a stable shared contract.

### Delegate

All implementation work is assigned to general-purpose sub-agents. Each prompt
must explicitly forbid commit and push.

### Review

After each delivery, the Orchestrator reviews `git diff`, checks alignment with
the frozen contracts in `doc/`, and runs the relevant tests.

### Iterate

If tests fail or review finds drift, fixes are delegated back out to sub-agents.
The Orchestrator does not patch the code inline.

### Release

Only after a full phase is green does the Orchestrator perform version bumping,
commit, tagging, push, and `clog` logging.
