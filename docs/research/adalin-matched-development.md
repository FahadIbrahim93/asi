# AdaLin matched development campaign

This lane compares the AdaLin ReLU mechanism with its exact alpha-zero,
mechanism-disabled control over five consumed development seeds. It is permanently
nonpromoting: its only valid campaign decision is `inconclusive`, including when every
paired metric favors one arm. No result has been produced by adding this machinery.

Run `asi-adalin-matched-development --help` for the create-only CLI contract. The caller
must supply one NPZ file with exact `train_inputs`, `train_labels`, `test_inputs`, and
`test_labels` arrays. The runner binds the complete dataset bytes, current execution
sources, Python/JAX/dependency/runtime configuration, the explicit Threefry schedule,
each arm's initial state, and the exact seed-by-arm roster. Its strict validator reruns
all ten shards and compares every result except wall-clock telemetry. Additive work is
summed across shards; persistent state is reported as the maximum per-shard allocation.

The bounded profile uses eight tasks, 64 training examples per task, batch size one,
and a 300-by-150 MLP. These settings intentionally keep a development campaign finite;
they do not reproduce the paper's 400 tasks, 10,000 examples per task, batch size 16,
or 100-by-100 MLP. The learner receives neither task identity nor boundary events, while
the evaluator necessarily uses task boundaries for permutation and test evaluation.

The following gates remain open before any paper-level or scientific conclusion:

- The pinned official repository commit contains only a README and supplies no runnable
  implementation or experiment configuration to compare against.
- Official MNIST bytes, sampled indices, permutation seeds, example orders, and exact
  dataloader behavior are not specified or bound. Caller-supplied data is not a substitute.
- The paper-scale architecture, batching, 400-task horizon, per-task reporting, and its
  three final evaluation seeds have not been reproduced.
- A separate preregistered protocol would require untouched seeds, frozen thresholds and
  artifacts, independent validation, and explicit promotion authorization. These five
  development seeds can never be reused for that purpose.

Consistency hashes detect drift but are not authenticated execution attestation. Timing
remains telemetry-only, and the lane makes no performance, scientific, or SOTA claim.
