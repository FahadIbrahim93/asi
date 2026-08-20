# V5 append-only amendment and audit

This file amends, and does not replace, the retained V5 JSON, runner, or
report. Their hashes are bound in `V5_model_side_amendment.v1.json`.

The original execution is not a literal execution of its preregistration. Both
model-side controls failed, so the runner was required to abort before online
checkpoint scoring. It instead produced 216 cells. Those observations remain
available as raw history, but zero online cells are valid under the registered
protocol and the model-side rows are void. The raw runner's data-dependent
`promotion.promoted` calculation is archived behavior and is ignored; the
maintained audit has no promotion route and permanently forbids scientific
promotion.

The F5c value near 2,000 samples is retained only as a same-stack descriptive
consistency check. It reused MNIST, schedule construction, and seeds and used a
stronger hybrid batch estimator, so it is neither an independent replication
nor confirmation of a general floor.

The structural interpretation is also narrower than the original report. For
a novel permutation, improving a pre-shift model reference does not remove the
need for post-shift identifying information. This does not cover recurrence:
a repeating permutation can be recognized using stored whole-input behavior
without identifying pixels independently. V5 therefore leaves the model-side
family and ledger entry 15 open.

The original cache bytes, complete dependency lock, and exact invocation were
not recorded. The companion JSON binds the historical source and reported
runtime plus a canonical OpenML cache reference, while explicitly preserving
those provenance limitations. Its status is
`invalid-preregistered-execution`, not evidence or performance.
