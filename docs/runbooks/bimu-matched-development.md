# BiMU bounded matched development campaign

This campaign compares the current binary Bayesian learner against its exact
mechanism-off reduction. It is a five-task, 256-train/256-test, width-32
development slice. It is permanently nonpromoting, is not paper-comparable,
and cannot support a scientific, SOTA, or paper-reproduction claim.

The literal plan fixes OpenML `mnist_784` version 1 through the canonical
60,000-row loader, the first 256 scaled rows as training data, the last 256 as
a disjoint development test, seeds 157001–157003, and arms `memory_off` and
`bimu`. The two arm configurations differ only in `memory_window` (`None`
versus `128`). Every other configuration, schedule, initial state, counter,
and numeric resource field must match within each pair.

Seeds 157001–157003 are publicly exposed by this literal development plan and
are therefore consumed for every promotion purpose. They have not produced a
retained matched result. Any future scientific protocol needs a new path and
untouched seeds; changing the execution gate cannot make this roster eligible.

The preregistered primary outcome uses the final-model mean accuracy over the
five task permutations. It is `supported` only when all three candidate-minus-
control deltas are strictly positive, `rejected` when all three are
nonpositive, and `inconclusive` otherwise. The whole-stream online metric is
secondary and cannot change that classification. Timing is telemetry only.

## Review and execution gate

`EXECUTION_AUTHORIZED` and `AUTHORIZATION_TRANSITION_APPROVED` are frozen to
`False` in
`alberta_framework/evaluation/bimu_matched_nonpromoting.py`. The `run-shard`
command and every publisher fail before plan, data, replay, or execution work
unless both literals are exact `True`. A separate reviewed authorization change
must open both gates; both values are bound into the plan, identity, policy,
completed and failed shards, and status. That source change also
changes the plan's source identity, so the plan must then be published from
the authorized revision before any shard starts. That review must also update
the literal `FROZEN_PLAN_SHA256`; an authorization flip without the matching
reviewed plan digest fails closed.

Inspect the currently unauthorized plan in a disposable preview namespace:

```bash
campaign_preview=$(mktemp -d)
.venv/bin/asi-bimu-matched-development plan --root "$campaign_preview"
.venv/bin/asi-bimu-matched-development validate --root "$campaign_preview"
```

Disposable plan publication and read-only validation cannot dispatch a shard
or produce an aggregate. `run-shard` and `summarize` accept only the registered
repository root. Do not invoke `plan --root .` until a reviewed authorization
transition is ready to publish the source-bound plan in the new immutable
namespace.

If any file already exists in `development.v1`, the authorization change must
advance the namespace rather than replace that file.

After separate authorization, launch each of the six commands in its own fresh
Linux process. Substitute each frozen arm and seed exactly once:

```bash
.venv/bin/asi-bimu-matched-development run-shard \
  --root . --arm memory_off --seed 157001
```

Once all six fixed shard paths exist, summarize and revalidate:

```bash
.venv/bin/asi-bimu-matched-development summarize --root .
.venv/bin/asi-bimu-matched-development validate --root .
```

All files publish without replacement under the registered repository's exact
`outputs/bimu_matched/development.v1/` namespace. Plan and aggregate work is
reserved before validation, data loading, or strict replay; reservation cleanup
compares held and live inode identities. Keep supported, rejected, and
inconclusive aggregates. An ordinary `Exception` during shard execution or
strict validation publishes one generic failed-attempt receipt atomically at
that shard's canonical path and returns nonzero. The receipt retains no
exception type, message, or representation, cannot enter aggregation, and
forbids retry in this namespace. `BaseException`, process death, and failure to
publish the failure receipt remain outside this retention guarantee. Source,
runtime, dependency, process, dataset, resource, and telemetry digests are
consistency bindings, not authenticated execution attestation.
