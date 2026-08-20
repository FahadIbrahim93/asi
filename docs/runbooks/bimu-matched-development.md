# BiMU matched development run

This lane is a permanently nonpromoting, bounded comparison between the
current binary Bayesian learner with its memory disabled and the same learner
with the frozen 128-example memory window. It is not a reproduction of the
paper's 1,000-task result and cannot support an external SOTA claim.

The prospective plan is
`alberta_framework.evaluation.bimu_matched_nonpromoting.FROZEN_BIMU_MATCHED_PLAN`.
It fixes three consumed development seeds, the two-arm roster, five tasks,
256 training and 256 test examples, width 32, the exact OpenML-derived dataset
digest, counters, resource bounds, and comparison scope. The executable and
strict aggregate validator are in
`alberta_framework.evaluation.bimu_matched_runner`.

An operator must first reconstruct the exact float32/int32 arrays described by
the plan and independently verify their digest. Then call
`run_bimu_matched_development(train_x, train_y, test_x, test_y)` and retain the
returned result with `write_bimu_matched_result(...)`. The writer admits no
caller-selected destination. Its sole registered namespace is
`outputs/bimu_matched/development.v1/result.<content-sha256>.json` under that
exact root. Publication is durable and no-replace; supported, rejected, and
inconclusive outcomes are all retained.

Before treating a retained record as a development result, require:

- all six seed-arm rows and both arms for every seed;
- exact dataset, task schedule, initial state, observations, label queries,
  optimizer-seen count, model queries, and numeric resource matching;
- `validate_bimu_matched_result` against the same exact input arrays;
- current source and runtime identity equality; and
- permanent `development_only`, no-promotion, no-SOTA, and negative-retention
  policy fields.

The aggregate reports paired late-five and whole-stream online deltas. Timing
is telemetry only. Consistency hashes are not authenticated execution proof.
Paper comparison remains closed until the official data order, five-run
aggregate, paper-scale 1,000-task stream, and other recorded protocol gaps are
reproduced under a separately reviewed protocol.

The public runner and publisher are hard-disabled. A later, independently
reviewed authorization change must flip both the literal execution flag and
the runner transition gate; public execution requires both to be exact `True`,
and reports bind both values. Tests exercise only the private bounded executor.
Publication reserves a deterministic create-only name through a pinned
no-follow directory descriptor before strict reexecution, then uses a
no-replace link, fsync, bounded reread, and strict reload validation.
