# Gradual-transition development family

ASI has two deliberately separate adapters for the transition definitions in
[Liu and Mou, arXiv:2602.09234v2](https://arxiv.org/abs/2602.09234v2):

- `ipmnist_gradual.py` owns the retained abrupt-versus-input-interpolation
  single-pass result and its correction chain. Nothing in the new family
  reinterprets or promotes that record.
- `ipmnist_gradual_family.py` adds bounded, caller-fed, end-to-end
  output-interpolation and task-sampling arms beside an abrupt control.

The additive runner uses `K + 1` complete micro-phases at coefficients
`0/K, ..., K/K`. Every arm receives the same number of training observations,
updates, and post-phase new-task evaluation observations. Output interpolation
uses soft-target cross entropy for the paper's old-one-hot to uniform to
new-one-hot path. That arm requires the old and new task arrays to contain
identical, row-aligned inputs, and uses each shared row's corresponding old
and new labels. Task sampling selects exactly
`floor(alpha * M)` new-task examples and therefore
`ceil((1 - alpha) * M)` old-task examples from each `M`-example phase. Its
realized positions and old/new sample orders come from explicit Threefry keys.
The abrupt control switches from the old paired data to the new paired data at
the midpoint.

This is not a paper reproduction. In particular, it uses bounded paired
micro-phases rather than the paper's full datasets and horizon. The output arm
presents paired new-task inputs while its target moves from the paired old
label through the uniform distribution to the paired new label. Callers supply
the arrays; the adapter does not establish an official dataset acquisition
protocol.

Each result binds both materialized datasets, the realized schedule, the
current adapter, learner, optimizer-safety, and seed-validation source files,
Python/JAX/NumPy/backend identity,
exact counters, and persistent numeric bytes. Validation recomputes the data
and schedule identities. Timing is telemetry only. The runner does not write
artifacts, authenticate execution, or authorize promotion. All results are
development-only, and negative results must be retained if a future reviewed
campaign chooses to persist them.

## Five-seed matched campaign

`gradual_micro_phase_campaign.py` composes the additive runner into a frozen,
permanently nonpromoting five-seed development campaign. Its bounded plan uses
10 transition intervals, 5,000 paired rows, the 784-300-150-10 AdamW network,
and exactly one row-aligned old/new phase dataset. The resulting 15 arm cells
consume 825,000 updates and 2,475,000 model queries. This is intentionally not
the paper's complete dataset or training horizon. Features must be exact
float32 arrays in `[-1, 1]`; labels must be exact int32 arrays in the configured
class range.

The campaign binds exact caller-fed dataset bytes (without claiming an official
acquisition protocol), current relevant sources and lockfile, runtime/JAX
configuration, explicit Threefry roots, realized schedules, initial parameters
and learner states, exact counters, and numeric resource receipts. Validation
reexecutes all five three-arm runs and exact-compares the normalized report;
unqualified timing is discarded. The writer validates before create-only,
read-only publication. No output is created automatically.

The paired intervals are descriptive only. Every report records an
`inconclusive` decision because no selection rule is registered. Before any
campaign execution, contributors must preregister a selection rule, minimum
effect, multiplicity and resource gates, obtain compute approval, and choose a
new append-only destination. Paper-scale replication, official dataset
acquisition, untouched scientific seeds, downstream recurrence/control tests,
measured peak RSS, and qualified timing remain separate open gates.

Run the bounded contract checks with:

```bash
.venv/bin/python -m pytest tests/test_ipmnist_gradual_family.py -q
```

The public campaign and publisher fail closed while both the frozen plan and
separate transition gate are unauthorized. Tests use only the private bounded
executor. Publication reserves a deterministic name before strict reexecution
and uses pinned no-follow directory operations, no-replace linking, fsync,
bounded reread, and strict reload validation.
