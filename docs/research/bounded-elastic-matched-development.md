# Bounded elastic matched development

Issue #1562 compares the existing bounded structure-off, bounded growth, bounded elastic, and
fixed-capacity CBP arms under one peak-memory and final-size budget. This is an ASI fixed-shape
IPMNIST adaptation of `arXiv:2608.01475v1`; the paper discloses no official code repository, and
the protocol records the architecture, task-length, input-scaling, boundary, and pruning-sample
differences. It is not a paper reproduction.

The prospective plan binds the canonical materialized OpenML MNIST training-array digests,
current source/runtime identity, exact `pyproject.toml` and `uv.lock` bytes, an exact 8-task by
5,000-example configuration, all four arms, and five globally searched campaign seeds. Tests use
a separate test-only capability and seed roster, so they never execute the campaign schedule.
Each result retains observations, updates, data and environment steps, model queries, persistent
bytes, peak budget, active final size, structural events, and telemetry-only timing. One aggregate
256 MiB numeric envelope covers retained dataset, schedule, and peak persistent bytes and is
checked before dataset copying, schedule construction, parameter initialization, or execution.

The preregistered primary comparison is paired whole-stream mean online accuracy for
`bounded_growth` and `bounded_elastic` against `bounded_fixed_cbp`. A candidate is supported only
when all five paired deltas are strictly positive, rejected only when all five are nonpositive,
and otherwise inconclusive. The campaign is supported when either candidate is supported,
rejected when both are rejected, and otherwise inconclusive. This conservative sign rule is a
development-selection outcome only; it is not a significance test or scientific evidence.

Execution, strict reexecution, and publication are hard-disabled until both a separately reviewed
source transition and runtime authorization become exact `true`. The future output path is NEW.
Before any dataset load or runner dispatch, the campaign publisher must reserve that path through
per-segment no-follow directory descriptors. Publication is create-only, fsynced, uniquely linked,
bounded-reread, duplicate-key rejected, and strictly revalidated. Every result is development-only,
permanently nonpromoting, and retains negative or inconclusive outcomes.

The read-only catalog is available without loading MNIST:

```bash
.venv/bin/python -m alberta_framework.evaluation.bounded_elastic_matched_runner --catalog
```

No campaign run or result artifact is included with this plan.
