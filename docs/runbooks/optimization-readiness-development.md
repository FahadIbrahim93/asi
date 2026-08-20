# Optimization Readiness development execution

ASI's prospective Optimization Readiness slice is implemented in
`alberta_framework/evaluation/optimization_readiness_executor.py`. It is a
permanently nonpromoting execution protocol, not a completed experiment or an
external-comparator claim.

The current slice deliberately uses one bounded model whose entire computation
can be independently repeated: float64 linear regression without a bias term,
mean squared loss, and a supplied 10,000-example task/checkpoint. Before reading
array elements, it rejects more than 64 parameters, more than 500 million
preflight work units, or more than 256 MiB of conservatively estimated total
live memory. A preflight work unit is a safety-only parameter/data-incidence
proxy used to bound this fixed implementation. It is not a floating-point
operation count or a resource-comparison metric.

For each execution it derives rather than accepts:

- the full-validation loss and gradient;
- 128 size-four gradients sampled independently with replacement from an
  explicit JAX Threefry root;
- Optimization Readiness strength, reliability, and combined score;
- parameter and gradient norms;
- 99% energy ranks for the input representation and exact squared-loss
  curvature; and
- mean relative validation-loss reduction across 128 independent SGD rollouts
  at each frozen horizon: 1, 10, and 100 steps.

The artifact binds package-source hashes, Python/NumPy/JAX/runtime facts, the
RNG implementation and schedule digest, exact dataset and checkpoint content,
seed, task/checkpoint labels, and deterministic resource counters. Call
`validate_optimization_readiness_execution(...)` with the original arrays and
checkpoint. The validator checks their identities and repeats every schedule,
gradient, update, diagnostic, metric, and counter; it does not trust artifact
claims.

Wall time is deliberately recorded as unmeasured (`0.0`, with
`timing_measured=false`) because this protocol has not qualified a timing
environment. It is never an acceptance input.

The additive `optimization_readiness_entk.py` adapter closes one important
model gap without rewriting the linear Appendix C.1 slice. It accepts an exact
caller-owned float64 two-layer ReLU regression checkpoint, analytically builds
its empirical neural-tangent feature matrix, measures hidden-representation and
eNTK 99% energy ranks, and derives the loss gain from one exact full-gradient
trajectory at each independent 1/10/100-step horizon. Its strict validator binds
and replays the dataset, checkpoint, current source, NumPy build/runtime,
metrics, and conservative memory/work receipt. One model query is one example
evaluation, including a differentiated evaluation only once; the receipt
charges the initial diagnostic, all 111 optimizer updates, three terminal
evaluations, and both representation and Jacobian SVDs.
It bounds observations at 256, input dimension at 32, hidden width at 64,
parameters at 4,096, SVD work at 100 million units, rollout work at 200 million
units, and total numeric memory at 256 MiB before copying arrays or invoking
linear algebra. This is a real
nonlinear checkpoint diagnostic, but not the paper's deep-network eNTK panel.

No campaign output is checked in. A comparison campaign still needs a
predeclared, matched panel of real continual-learning model checkpoints and
tasks, the paper's exact architectures/representations and eNTK baselines,
retained failures, and fresh execution after its protocol is reviewed and
frozen. Any such run must write to a new append-only output path and remains
development-only.

The additive checkpoint-panel adapter in
`alberta_framework/evaluation/optimization_readiness_panel.py` supplies the
bounded orchestration and retention boundary for that next step. It accepts
three to sixteen caller-owned cases with distinct dataset/checkpoint content
identities (not merely distinct labels), runs the exact
model-bound executor for every case, derives Spearman association between
Optimization Readiness and the 1/10/100-step future-gain measurements, sums
logical work receipts, and validates by re-executing the complete roster. Its
status is only a development association (`supported`, `rejected`, or
`inconclusive`); it cannot promote a claim. Retention is create-only under
`outputs/optimization_readiness/development.v1/` and requires the original
arrays for strict validation. No case roster or result is supplied by the
repository: selecting real continual checkpoints, predeclaring their task
panel, reviewing the protocol, and retaining all outcomes remain open.
