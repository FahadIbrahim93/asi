# Continual World qualification boundary

This lane prepares an isolated, permanently nonpromoting CW20 environment smoke
test. It does not claim a Continual World result or robotics readiness.

The audit pins the final paper revision, `arXiv:2105.10919v3`, official repository
`https://github.com/awarelab/continual_world.git` at commit
`73f63bb4fa0b5d00bda973e20dfb783bfcf1b8aa`, and its Meta-World dependency
`0875192baaa91c43523708f55866d98eaf3facaf`. The exact CW10 ordering is copied
from the pinned official `continualworld/tasks.py`; CW20 repeats it once. The
official runtime is legacy: Ubuntu 18.04, Python 3.6, TensorFlow 2+, Gym,
`mujoco-py>=2.0,<2.1`, MuJoCo 2.0, and Meta-World v1. The recipe does not pin most
transitive versions or downloaded archive bytes, so an ASI run must first build a
separate image and bind its immutable image digest, MuJoCo archive SHA-256, and
resolved package/runtime versions in `IsolatedRuntimeIdentity`. The resulting
receipt also binds the current ASI qualification source, dependency sources,
the authoritative issue-1580 external-qualification catalog, host runtime,
JAX/NumPy versions, and the frozen paper/workload registries. A
receipt from a different current tree or host runtime fails closed.

The qualification smoke traverses all 20 official tasks for two steps each with
the fixed float32 zero action. The evaluator may record the task index, boundary,
and name; no learner exists and no task or boundary information reaches a learner.
The receipt binds observation, reward, success, and task-index traces; charges the
fixed action's 16 persistent bytes and the externally measured simulator numeric
array state (the sum of reported numeric-array `nbytes`, not process RSS, native
heap, or renderer allocations); records exact logical action, observation,
reward, success, and task-index trace
bytes, 40 environment steps, zero updates/data steps/model queries, and
telemetry-only timing; and retains supported, rejected, and inconclusive outcomes.
Those outcomes are scoped only to completion of the fixed-action official-runtime
trace. In particular, `supported` is not a learning, paper-comparison, performance,
robotics-readiness, or scientific outcome. The 16 mechanism bytes describe the
canonical float32 action payload and exclude Python-object overhead.

Before any development comparison, the real legacy image must execute this smoke
and its receipt must validate. A learning lane must then add matched frozen seeds,
the official SAC fine-tuning control, reset and multihead ablations, exact replay
and optimizer state accounting, and full failure retention. Paper comparability
additionally requires 1M steps per task, 20 seeds, randomized object positions,
separate task heads, single-task reference curves, binary-success evaluation,
final performance, forgetting, forward transfer, and 90% bootstrap intervals.
The paper reports roughly 100 hours for one CW20 run, so none of that campaign is
executed as part of tests or this setup change.
