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

`EXECUTION_AUTHORIZED` is frozen to `False` in
`alberta_framework/evaluation/bimu_matched_nonpromoting.py`. The `run-shard`
command fails before loading MNIST while that literal is false. A separate
reviewed authorization change must open the gate; that source change also
changes the plan's source identity, so the plan must then be published from
the authorized revision before any shard starts. That review must also update
the literal `FROZEN_PLAN_SHA256`; an authorization flip without the matching
reviewed plan digest fails closed.

Inspect the currently unauthorized plan in a disposable root. Do not publish it
under the repository's immutable output namespace:

```bash
campaign_root=$(mktemp -d)
.venv/bin/asi-bimu-matched-development plan --root "$campaign_root"
.venv/bin/asi-bimu-matched-development validate --root "$campaign_root"
```

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

All files publish without replacement under
`outputs/bimu_matched/development.v1/`. Keep supported, rejected, and
inconclusive aggregates. Source, runtime, dependency, process, dataset,
resource, and telemetry digests are consistency bindings, not authenticated
execution attestation.

Each shard's final path is reserved before plan admission, dataset load, or
execution. A crash or execution failure intentionally leaves a zero-byte,
mode-000 non-result reservation at that path: it consumes that attempt and is
not a valid shard or permission to retry. Successful publication fills the
same held inode, makes it read-only, rereads bounded bytes without following
links, and strictly validates them before reporting completion.
