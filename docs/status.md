# Alberta Plan research status

This document is the compact, human-readable status map for the Alberta
Framework. It separates available mechanisms, stored scientific outcomes, and
the evidence still required for a complete continual agent.

**Verdict: in progress.** The package exposes research surfaces related to all
twelve steps of the Alberta Plan, but no step satisfies this repository's full
completion rule. There is no accepted end-to-end Alberta Plan completion claim.

This page deliberately does not record a dated live evidence-registry result,
module count, test count, campaign ranking, or session chronology. Those facts
change. Run the relevant command or read the primary artifact instead.

## Sources of authority

Use each record only for the question it owns:

- This page records the stable evidence levels, current Step 1–12 gaps, and
  completion gates.
- [Evidence methodology](evidence/methodology.md) owns promotion rules,
  property-level evidence, artifact contracts, and scientific limitations.
- The [negative-results ledger](evidence/negative-results.md) owns rejected,
  bounded, consumed, and closed development results.
- [`evidence_manifest.py`](../alberta_framework/evaluation/evidence_manifest.py)
  owns the live five-claim registry, its exact source closures, and exit-code
  semantics.
- Versioned JSON artifacts own frozen scientific outcomes. Validators, not
  narrative summaries, decide whether those artifacts match current sources.
- IPMNIST summaries and reports under
  [`outputs/ipmnist_screening/`](../outputs/ipmnist_screening/) own that moving
  development campaign's measurements.

Implementation, tests, smoke runs, development experiments, and scientific
evidence are different kinds of progress. A public class or passing test proves
that a mechanism exists; it does not by itself prove benefit, retention,
resource parity, or integration.

## Evidence interpretation

The stable evidence scale is L0 for mechanism contracts, L1 for learning in a
controlled toy problem, L2 for preregistered matched-resource comparison, and
L3 for integration across one uninterrupted agent life. The
[evidence methodology](evidence/methodology.md) defines these levels,
promotion rules, artifact contracts, registered frozen outcomes, and the two
narrow historical compatibility routes in detail.

For this repository, a Step is complete only when its defining outcome reaches
L2 and its required links to earlier Steps are exercised at L3. Missing
promoting evidence must fail closed; it must not be treated as a skipped test
or inferred from adjacent results. All twelve public Step modules contain
mechanism or smoke surfaces. Only the Step 1 and Step 2 smoke probes are
console scripts; smoke execution is L0 and structurally nonpromoting.

## Live evidence registry

The registry is intentionally limited to five narrow claims. Its exact claim
inventory, source closures, and exit-code semantics live in
[`evidence_manifest.py`](../alberta_framework/evaluation/evidence_manifest.py),
while versioned artifacts and their strict validators own frozen outcomes. Run
the live check from a repository checkout:

```bash
.venv/bin/alberta-evidence-status
```

An accepted registry does not certify the package, complete a Step, or
establish Alberta Plan completion. Source mismatches and invalid artifacts fail
closed; pinned artifacts must not be edited or repaired. Normal wheel and
sdist installations exclude `outputs/`, so their registry checks ordinarily
report artifacts as missing.

## Alberta Steps 1–12

### Step 1 — Nonstationary prediction

**Required outcome.** Track nonstationary affine prediction online with
normalization, relevance-sensitive step sizes, robustness, and bounded work.

**Available surface.** The package includes drifting Step 1 streams, LMS,
IDBD, Autostep and comparison optimizers, online normalizers, update bounding,
and a deterministic smoke kernel.

**Open gate.** No frozen matched multi-seed Step 1 comparison and no L3 link to
the later learned-state/control agent satisfy the completion rule.

### Step 2 — Feature construction and replacement

**Required outcome.** Generate useful nonlinear features, estimate future
utility, and replace features within a fixed budget while retaining recurring
critical structure.

**Available evidence.** Pair construction, bounded feature banks, utility and
lifecycle mechanisms exist. The recurring-pair and scale-robust-pair artifacts
record two accepted but deliberately narrow L2 outcomes.

**Open gate.** The stored protocols begin from constrained pair-product
families and do not establish autonomous question discovery, general
selective retention, or an uninterrupted downstream control benefit. Separate
development failures and bounds remain nonpromoting and are recorded in the
[negative-results ledger](evidence/negative-results.md).

### Step 3 — Many continuing predictions

**Required outcome.** Learn many continuing, potentially off-policy GVFs with
history and feature finding.

**Available surface.** Horde, mixed and independent demons, TD/GTD variants,
traces, normalization, and a causal Step 2 feature handoff have mechanism and
small learning coverage.

**Open gate.** There is no promoted matched comparison showing useful learned
questions/features across recurrence, nor an L3 connection to the final
control loop.

### Step 4 — Control I

**Required outcome.** Progress from bandit and contextual control to
sequential actor-critic or action-value control with learned features.

**Available surface.** SARSA, discrete and continuous actor-critic,
average-reward and off-policy variants, bounded updates, and small online
control diagnostics are implemented.

**Open gate.** No frozen resource-matched continual-control result establishes
retention and recovery while learned state changes, and no complete L3 link to
prediction and feature lifecycles exists.

### Step 5 — Continuing prediction II

**Required outcome.** Learn differential average-reward predictions together
with conventional value and expected-duration predictions needed by options.

**Available surface.** Differential TD/GTD/Horde learners and bounded
return/duration model components have mechanism and toy-learning coverage.

**Open gate.** There is no promoted comparison or integrated option-control
result demonstrating calibrated multi-timescale predictions.

### Step 6 — Continuing control II

**Required outcome.** Demonstrate reproducible continuing average-reward
control across the intended suite, including RiverSwim, access control,
Jellybean, GARNET, and continuing conversions.

**Available surface.** Differential SARSA and actor-critic mechanisms,
closed-loop micro-environments, benchmark adapters, and Forager campaign
tooling are present.

**Open gate.** The named suite has no single promoted result, and no completed
paper-length, matched-resource Alberta-versus-baseline Forager comparison
exists.

### Step 7 — Incremental planning

**Required outcome.** Validate bounded incremental average-reward planning,
then planning with function approximation and adaptive features.

**Available surface.** World-model updates, bounded dreaming, and option
search-control are available as mechanism surfaces.

**Open gate.** Model support and uncertainty are not yet externally calibrated,
and there is no frozen matched-budget result showing reliable planning benefit
under changing dynamics and representations.

### Step 8 — Learned world and representation loop

**Required outcome.** Close the perception → world model → feature ranking →
feature replacement → model-feedback loop.

**Available evidence.** One-step, action-conditioned, shallow, ensemble,
recurrent, and latent model components exist. The FTL artifact records a
narrow historical L2 decision-fidelity acceptance for one fixed-shape model
and protocol.

**Open gate.** The accepted FTL scope does not establish a calibrated general
world model, learned-target quality, retained planning benefit, partner
modeling, or the complete feedback loop under one owner and lifetime.

### Step 9 — Exploration and search control

**Required outcome.** Improve exploration and planning order under matched
real-transition, model-query, and backup budgets without exploiting noisy or
irrelevant novelty.

**Available surface.** Surprise, priority, utility, guarded dreaming, and
bounded search-control mechanisms exist.

**Open gate.** Development diagnostics do not provide a preregistered
matched-budget benefit, calibrated causal score production, or an integrated
held-out exploration result.

### Step 10 — Subtasks, options, models, and planning

**Required outcome.** Discover reward-respecting subtasks, learn options and
option models, and consume those models in planning.

**Available surface.** STOMP supports specified subtasks, temporally extended
actions, option learning, outcome models, and bounded option-model backups.

**Open gate.** The default path does not autonomously discover and repeatedly
maintain useful subtasks under one continual owner, and no held-out matched
benefit result closes the loop.

### Step 11 — Causal utility and OaK

**Required outcome.** Track causal utility, safely replace features, subtasks,
options, and models, and compose behaviours through an option keyboard.

**Available surface.** OaK utility tracking, option curation, keyboard
mechanics, and feature-lifecycle transactions are implemented.

**Open gate.** Autonomous go/no-go authority, repeated safe replacement,
automatic keyboard consumption, and causal outcome evidence remain absent.
Mechanism-level lifecycle receipts do not grant deployment or promotion.

### Step 12 — Intelligence amplification

**Required outcome.** Measurably increase another learning agent's capability
through a closed, continuing interaction loop.

**Available evidence.** Prediction augmentation, recommendation protocols,
partner/world models, and two-agent streams exist. The recurring-multiagent
artifact records one narrow accepted L2 coadaptation result. The frozen IA
artifact is a valid rejection.

**Open gate.** Coadaptation is not the same as causal amplification. The IA
intervention gate remains failed, and there is no L3 partner-benefit result
under changing reliability, communication cost, retained skills, and bounded
resources.

No Step currently satisfies the repository completion rule.

## Active development campaigns

### IPMNIST screening and confirmation

IPMNIST is the headline optimization/plasticity lane, but it is
**development-grade and permanently nonpromoting**. Screening, confirmation,
and publication-run records may support descriptive development conclusions;
they do not become scientific evidence through replication, more seeds, or
better performance.

Use the primary records rather than copying rankings or means here:

- [theory and forward hypotheses](research/ipmnist-theory.md);
- [campaign runbook](../outputs/ipmnist_screening/RUNBOOK.md);
- [stored development report](../outputs/ipmnist_screening/FINAL_REPORT.md);
- [artifact and reproducibility audit](../outputs/ipmnist_screening/AUDIT.md);
- [publication-run record](../outputs/ipmnist_screening/publication_runs/RESULTS.md);
  and
- `outputs/ipmnist_screening/summary_*.json` for the latest stored summaries.

Remeasure the intended baseline under the current development protocol before
any A/B comparison. Do not infer registry acceptance, a promoted
state-of-the-art claim, or an Alberta Step completion from this lane. Seeds
used for development or selection cannot later serve as untouched promotion
seeds.
