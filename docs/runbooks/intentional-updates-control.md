# Intentional Updates control development lane

This lane closes the implementation gap between ASI's equation-level
Intentional Updates utilities and an actual temporal consumer. It runs linear
TD(0), TD(λ), and Q-learning on a bounded recurring two-state continuing
MDP. It is additive: the supervised IPMNIST extension remains unchanged.

The six public arms are three matched pairs:

| Consumer | Fixed-step control | Candidate |
|---|---|---|
| State-value TD(0) | `fixed_td0` | `intentional_td0` |
| State-value TD(λ) | `fixed_trace` | `intentional_trace` |
| Q-learning control | `fixed_q_learning` | `intentional_q_learning` |

Private `*_off` aliases execute the exact fixed arm and exist only to test the
mechanism-off reduction. Candidate arms use the paper's pinned TD(0) or
conservative trace step-size kernel. Q-learning uses the TD(0) construction on the
selected action value. This linear two-state adaptation is not the paper's
reported benchmark suite and is explicitly marked `publication_equivalent:
false`.

## Run one bounded shard

Use the project environment. Omit `--output` to print strict JSON:

```bash
.venv/bin/python -m alberta_framework.benchmarks.intentional_updates_control \
  --arm intentional_trace --seed 0 --horizon 512 --phase-length 64
```

To retain a development result, choose a new path. Publication is create-only;
an existing file is never overwritten:

```bash
.venv/bin/python -m alberta_framework.benchmarks.intentional_updates_control \
  --arm intentional_trace --seed 0 \
  --output scratch/intentional-updates-trace-seed-0.json
```

Do not write exploratory records into a frozen evidence root. A retained
negative result belongs in the repository's negative-results process; do not
delete it merely because the candidate loses.

## What the record proves

The validator re-executes the complete stream and learner and requires an exact
record match. Each record binds:

- current bytes of the lane and shared equation-kernel modules;
- Python, JAX, jaxlib, NumPy, backend, and platform identity;
- the control agent's explicit Threefry RNG implementation;
- the complete configuration and a canonical workload digest;
- environment steps, rewards, updates, model/action queries, Threefry split and
  fold-in calls, trajectory length, persistent numeric bytes, diagonal-statistic
  updates, trace updates, and intentional step-size solves;
- the complete trajectory, final state, and summary metrics.

Timing is deliberately absent: no timing protocol has been qualified. The
candidate performs extra step-size work, which its receipt exposes rather than
calling compute matched. Passing validation remains development infrastructure,
not a measured win or scientific evidence.

## Before a campaign

Freeze a paired seed schedule and candidate selection rule in a separate plan,
remeasure all fixed controls on the same source identity, and retain every
completed result—including losses. A publication claim additionally requires
paper-faithful environments and architectures, untouched held-out seeds,
preregistration, a versioned evidence schema, and review. This runbook does not
authorize such a campaign.
