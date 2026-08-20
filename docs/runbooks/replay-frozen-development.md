# Replay/frozen matched development campaign

The maintained adapter is
`alberta_framework/evaluation/replay_frozen_campaign.py`. It executes five
consumed development seeds across the exact eight-arm replay/frozen roster.
Every retained outcome is permanently `inconclusive`: this campaign verifies
execution and accounting but neither ranks arms nor advances a candidate.

The 40-shard transaction binds the frozen OpenML `mnist_784` v1 bytes, full
IPMNIST configuration, explicit Threefry roots, seed schedules and initial
parameters, every seed/arm initial learner state, current package/config
sources, runtime/dependencies, exact per-run receipts, and aggregate logical
resource counters. Validation recomputes every shard. Unqualified runner timing
is discarded and canonicalized to zero; peak working set and scalar FLOPs are
not claimed.

## Operator sequence

Do not execute this campaign from pytest. Use a new absolute run root and
retain every outcome without replacement:

```python
from pathlib import Path

from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig, load_mnist_train
from alberta_framework.evaluation.replay_frozen_campaign import (
    retain_replay_frozen_campaign,
    run_replay_frozen_campaign,
)

x, y = load_mnist_train()
config = IPMNISTConfig()
result = run_replay_frozen_campaign(x, y, config=config)
destination = retain_replay_frozen_campaign(
    result,
    x,
    y,
    config=config,
    repository_root=Path("/absolute/new/asi-run-root"),
)
print(destination)
```

The writer creates
`outputs/replay_frozen/development.v1/result.<sha256>.json` below the new root.
It refuses an existing content name or symlinked namespace. Do not copy a
result into a pinned output root without a separately reviewed retention
decision.

## Interpretation boundary

These arms are protocol-extended local controls, not paper reproductions or
pretrained ceilings. RanDumb uses raw IPMNIST pixels. The RanPAC arm uses a
random raw-pixel projection and recursive ridge readout, not a pretrained ViT
and task-end Gram recomputation. The PROL arms are architecture-only
prompt/affine proxies without the pretrained model and adapt class-incremental
semantics to task-free streaming. The replay Transformer is represented only
by bounded label attention and raw-example replay.

All local arms charge zero pretraining examples, updates, queries, and bytes;
therefore none can support a pretrained-feature ceiling claim. A useful result
would still need faithful upstream implementations and checkpoints, exact
pretraining/extractor data and compute provenance, matched current controls,
development selection criteria, transfer/retention tests, qualified memory and
timing, and a separately frozen fresh-seed scientific protocol.

## External feature qualification boundary

`asi-pretrained-feature-qualification` prints a static, fail-closed blocker
manifest for the genuine RanDumb, RanPAC, and PROL qualification lane. It pins
the official repositories and commits but performs no clone, download, model
execution, or output write. The accompanying Python provider protocol requires
a future isolated adapter to receive a caller-bound checkpoint/random
initialization, preprocessing, evaluation and pretraining dataset identities,
runtime image and lock identities, and exact pretraining and extraction costs.
Its bounded smoke receipt is permanently nonpromoting and cannot turn the
existing proxy arms into ceilings.

The issue remains open. Official artifacts and licenses must be acquired and
verified; RanPAC/PROL pretraining datasets and costs must be attested; the
RanDumb random initialization artifact must be reconstructed exactly; isolated
official adapters need parity traces; and replay plus all three ceiling arms
must run as one matched IPMNIST campaign with charged storage, queries,
compute, memory, and qualified timing before any ceiling comparison exists.
