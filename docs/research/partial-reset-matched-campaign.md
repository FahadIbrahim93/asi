# Partial-reset matched development campaign

This lane is a code-qualified, permanently nonpromoting comparison of the five
calibrated-partial-reset arms already registered in `ipmnist_screening.py`:
`cpr_ipmnist`, `cpr_hard_reset`, `cpr_l2_init`, `cpr_utility_free`, and the
`cpr_off` conditioning control. It is not a completed campaign, a result, a
scientific evaluation, or evidence of parity with the cited CPR paper.

The frozen plan consumes five development seeds, 200 tasks of 5,000 online
updates per cell, and 25 cells (25,000,000 observations and 50,000,000 model
queries). Every cell uses an explicitly selected JAX Threefry root. Arms within
a seed bind the same example/permutation schedule, initial parameters, initial
learner state, and declared resource envelope. The report also binds the exact
materialized canonical IPMNIST arrays, current implementation and lockfile
bytes, runtime/JAX configuration, terminal state hashes, and the exact
seed-major/arm-minor roster.

`build_partial_reset_campaign` runs the complete plan in one process and
returns a strict JSON-domain report. `validate_partial_reset_campaign` replays
the plan, source/runtime/dataset identities, schedules, initial states, each
cell receipt, paired arithmetic, and resource counters without rerunning the
learners. `write_partial_reset_campaign_new` validates before create-only,
read-only publication. No command in tests launches the full campaign, and no
output path is reserved or populated by this implementation.

There is deliberately no winner rule. All reports have decision status
`inconclusive` with reason `no_registered_selection_rule`; observed deltas
cannot select or promote an arm after the fact.

## Remaining gates

- Preregister a selection rule, minimum effect, multiplicity treatment, and
  resource acceptance rule before executing the five development seeds.
- Obtain an explicit compute authorization and choose a new append-only output
  destination; then retain every outcome, including failures and negative
  comparisons.
- Audit the per-parameter, retained-initialization implementation against the
  paper's per-neuron/fresh-reset mechanism and released code. These protocol
  differences remain material, so this lane cannot claim paper replication.
- If development results justify it, freeze a separate scientific protocol
  with untouched seeds and its own versioned schema and validator. Development
  seeds and reports from this lane can never be promoted.
- Compare any selected mechanism in recurrence/retention and downstream
  control settings, and account for measured peak memory and qualified timing;
  the current receipt makes no physical-RSS or latency claim.
