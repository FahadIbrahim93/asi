# Intentional Updates matched development

Issue #1561 is split into two compatible-within-family subprotocols: the existing batch-size-one
IPMNIST adapter, and a recurring two-state continuing-MDP consumer for linear TD(0), TD(lambda),
and Q(lambda). Metrics are paired only inside each family; the report does not rank supervised
accuracy against prediction error or control reward.

The mechanism is pinned to `arXiv:2604.19033v1` and author code revision
`sharifnassab/Intentional_RL@e86e26fd8613ac212e9a52c3fed8a01d0a31f685`. The control consumer
specializes the author optimizer's entrywise RMS statistic, eligibility trace, global normalizer,
adaptive error clipping, and update to a float32 linear learner. It is not the authors' neural
benchmark and is not publication-equivalent. The supervised adapter remains a documented
lambda-zero protocol extension rather than an RL reproduction.

The literal plan reserves four globally searched campaign seeds that no test executes, binds the
canonical materialized OpenML
MNIST training-array digests already retained by ASI, requires equal observation and update
counts within each pair, and asks four two-sided paired questions with a Bonferroni-adjusted
98.75% interval (`t(3) = 5.391949071934058`). Timing is retained as telemetry only. Persistent
bytes, data/environment steps, updates, model/action queries, RNG operations, and optimizer solves
are explicit in every child record.

Prediction pairs share their prescribed transition schedule. Q(lambda) pairs share the
environment, seed, and exogenous exploration variates; their learned policies may produce
different actions and transitions.

The earlier draft roster was consumed by contract tests and is explicitly quarantined. Contract
tests use a distinct test-only capability and roster. Execution and public publication are
deliberately disabled in the initial merge; both a literal reviewed-transition flag and a separate
runtime authorization flag must become exact `true` before reservation, dataset loading, or a
consumer call. Inspecting the catalog is read-only:

```bash
.venv/bin/asi-intentional-updates-matched-development --catalog
```

A separate independently reviewed change must authorize execution. An authorized run may use only
the frozen input identity and reserved NEW output path. It acquires an exclusive reservation before
dataset loading, pins every directory segment without following symlinks, publishes by no-replace
hard link, fsyncs, and strictly rereads and revalidates the report. Every supported, rejected, or
inconclusive completed comparison remains development-only and permanently nonpromoting. A failure
after the first consumer dispatch leaves the immutable reservation as a consumed-without-result
tombstone, so a partially consumed schedule cannot be retried or represented as a retained result.
