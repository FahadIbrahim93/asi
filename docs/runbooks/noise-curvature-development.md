# Noise-curvature matched development campaign

The maintained campaign adapter is
`alberta_framework/evaluation/noise_curvature_campaign.py`. It executes the
four registered scheduler arms for the frozen development seeds 0–4 through
the current exact-step IPMNIST runner. This is a permanently nonpromoting
development screen, not a paper reproduction, scientific evaluation, or SOTA
claim.

The transaction binds:

- the frozen OpenML `mnist_784` v1 training selection/materialization and its
  exact float32 input and int32 label bytes;
- the complete IPMNIST configuration;
- an explicit Threefry root, initialization, permutation schedule, example
  schedule, and noise root for every seed;
- the directly executed package-source closure and Python/NumPy/JAX runtime;
- the complete 20-row seed/arm roster and the existing strict per-run receipts;
- exact logical query, observation, schedule, and persistent-numeric-resource
  totals; and
- uncorrected paired Student-t development intervals for combined minus fixed,
  gradient-only, and volatility-only scheduling.

Validation repeats every run from the supplied arrays. The current runner's
unqualified wall-time telemetry is discarded and canonicalized to zero;
compiler temporaries and aggregate peak memory are not claimed. A source or
runtime change during execution fails the transaction.

## Operator sequence

Do not run this from pytest. Select a new absolute run root and retain every
outcome without replacement:

```python
from pathlib import Path

from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig, load_mnist_train
from alberta_framework.evaluation.noise_curvature_campaign import (
    retain_noise_curvature_campaign,
    run_noise_curvature_campaign,
)

data_x, data_y = load_mnist_train()
config = IPMNISTConfig()
result = run_noise_curvature_campaign(data_x, data_y, config=config)
destination = retain_noise_curvature_campaign(
    result,
    data_x,
    data_y,
    config=config,
    repository_root=Path("/absolute/new/asi-run-root"),
)
print(destination)
```

The campaign rejects arrays that do not match the current screening lane's
60,000-row OpenML MNIST selection, `[-1, 1]` float32 materialization, and int32
labels. The domain-separated materialization hashes are
`b8078cd833f53d89828a5e28d728517be9add34076f13fe973399f1f16381313`
for features and
`4f1dd9551f104f8153409e0add59f0a71568f7bad5a5f8e2274480c186fe219a`
for labels; these are the same values retained in existing strict IPMNIST
records. Supplying arbitrary shape-compatible arrays is not a dataset identity.

The writer creates
`outputs/noise_curvature/development.v1/result.<sha256>.json` below that new
root and refuses an existing content name or a symlinked namespace. Do not copy
the result into a pinned output root without a separately reviewed retention
decision.

## Remaining gates

The four-arm result cannot evaluate the live-incumbent hillclimb gate because
the current RLS control has a separate campaign protocol and is deliberately
excluded. Even a positive five-seed screen still requires a current-source
live-control comparison, 200-task development confirmation, recurrence and
control transfer, honest peak-memory qualification, and a separately frozen
fresh-seed protocol before any scientific claim. The implementation remains a
paper adaptation: it uses input-permuted rather than random-label MNIST, a
different network and online update protocol, recent-example diagnostics,
approximated curvature statistics, and no identified official code.
