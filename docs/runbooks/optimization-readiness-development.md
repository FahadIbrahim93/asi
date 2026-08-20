# Optimization Readiness prospective development panel

This is an unexecuted, permanently nonpromoting plan for issue #1568. It pins
Wang et al., *Predicting Plasticity in Deep Continual Learning: A Theoretical
Perspective*, `arXiv:2605.09044v1` (2026-05-09). The paper cited no official
code revision as of the 2026-08-20 freeze.

The private model adapter is exact float64 linear regression without a bias,
with squared loss and one caller-owned 10,000-example validation task. For each
checkpoint it derives the full-validation loss and gradient, 128 independently
sampled size-four diagnostic gradients, parameter and gradient norms, input
representation and exact Hessian 99% energy ranks, and mean relative loss
reduction over 128 independent size-four SGD rollouts at 1, 10, and 100 steps.
The learning rate is `1e-3`; every RNG root is JAX Threefry. The strict replay
binds the source files, Python/NumPy/JAX runtime and dependency identities,
dataset and checkpoint bytes, schedule, counters, resources, and both
authorization literals.

The six exact seeds are `2684771901` through `2684771906`. Before this file was
written, that range had no exact match in fetched branch history or workspace
content. This public plan now exposes them, so they are unexecuted but already
permanently consumed for every promotion purpose. The exact roster, seed
status, decision rule,
paper identity, authorization state, and parity gaps are returned by
`frozen_optimization_readiness_plan()` without reading data or writing output.

## Authorization and transaction boundary

`AUTHORIZATION_TRANSITION_APPROVED` and `EXECUTION_AUTHORIZED` are separate
literal `False` values. Both must be changed to exact `True` in a separately
reviewed source transition, and the frozen plan's authorization identity must
change in the same review. The public runner fails before roster inspection,
directory creation, dataset hashing, scheduling, replay, execution, or
publication while either gate or the bound plan disagrees.

After authorization, the only public execution path reserves the deterministic
registered destination before inspecting the caller's cases. It holds the
reservation across all six executions, aggregation, no-replace publication,
bounded `O_NOFOLLOW` reread, and complete strict reexecution from the caller's
original arrays. Publication is create-only at the exact repository namespace
`outputs/optimization_readiness/prospective.v1/result.json`. Directory walking
uses pinned dirfds from the registered repository root; reservation cleanup and
post-link failure cleanup compare inode identities.

No result exists and no command should be run before independent audit. Timing
is currently unmeasured telemetry (`0.0`) and is never an outcome input.

## Deliberate paper-parity limits

This bounded local adapter is not the paper's Slowly-Changing Regression or
Permuted-MNIST experiment. Its input representation and exact linear Hessian
are checkpoint-invariant for a shared task, so their rank correlations may be
undefined. It does not implement the paper's nonlinear architectures, eNTK,
effective rank, active-neuron fraction, or P-MNIST 128-example gradient-Gram
curvature approximation. It evaluates only six checkpoints on one supplied
task, not the paper's multi-run/multi-task panels.

Those gaps are part of the immutable plan and prevent a paper-reproduction,
SOTA, scientific-evidence, or promotion claim. The result status is only
`supported`, `rejected`, or `inconclusive` for this bounded development panel.
It is supported only if Optimization Readiness has positive finite Spearman
correlation and strictly exceeds its gradient-strength-only mechanism-off
reduction, gradient norm, representation rank, curvature rank, and parameter
norm at every frozen horizon. Ties and undefined
correlations are inconclusive. Every outcome, including negative results, must
be retained; #1568 remains open until an authorized retained result and all
named paper-parity requirements are separately satisfied.
