# CLEAR qualification lane

This is a setup and accounting lane for CLEAR real-world continual imagery. It
does not download the dataset, train a model, report a score, authorize a run,
or create scientific evidence. Every result is development-only, permanently
nonpromoting, and retained when negative.

## Pinned research surface

- Paper: `arXiv:2201.06289v3` (9 June 2022), which supersedes v1 and v2.
- Official curation code: `linzhiqiu/continual-learning` at
  `620cab4a7d99921fde73b67b53879470533cb39a`.
- Authors' classification reference: `ElvishElvis/CLEAR-Continual_Learning_Benchmark`
  at `75d5d2e7d412a787e0decf0417a4868c56691252`.
- Avalanche adapter and metric implementation: `ContinualAI/avalanche` at
  `eb075be393e1f458b2c352514ff6c17b5a2c0f4e`.

CLEAR derives natural temporal buckets from YFCC100M imagery spanning
2004–2014. Bucket 0 is the optional unlabeled/pretraining bucket; the selected
CLEAR-100 supervised lane uses labeled buckets 1–10. The selected protocol is
streaming: the accuracy matrix is used to measure the near future, including
the superdiagonal `next_domain` metric. It is not the alternate within-bucket
70:30 IID protocol.

The project site labels its material CC BY, and the two codebases have their
own repository licenses. That is not sufficient evidence that every underlying
Flickr/YFCC asset is redistributable or still available. The current Avalanche
adapter downloads archives from a mutable Hugging Face `main` URL with
`checksum=None`; the authors' download script instead names S3 archives. No
reviewed provider revision or provider-published archive SHA-256 was found.
Those are explicit blockers, not values to infer. A local qualification receipt
therefore records caller-computed archive SHA-256 values while preserving
`provider_archive_checksums_published: false`.

## Frozen development comparison

The adapter records the official Avalanche example's ResNet-18-from-scratch
control: 224-pixel crops, ImageNet normalization, SGD at 0.01 with momentum
0.9 and weight decay 1e-5, batch 256, 100 epochs per bucket, and a step
scheduler every 30 epochs with gamma 0.1. Seeds 0–4 are ASI training RNG roots;
they are not CLEAR IID split seeds. Each control axis is paired with an exact
mechanism-off reduction. There is deliberately no mechanism-on implementation
in this issue.

The plan computes training observations, optimizer updates, model queries,
archive bytes, and zero environment steps from the verified manifest. A future
runner must additionally receipt exact persistent parameter/optimizer/replay
bytes; timing stays telemetry-only. Metrics are the five official matrix
summaries: accuracy, in-domain, next-domain, forward transfer, and backward
transfer.

## Local manifest and CLI

Create a JSON file with this exact shape and locally computed byte identities:

```json
{
  "schema_version": "asi.clear.qualification.v1",
  "dataset": "clear100",
  "protocol": "streaming-near-future",
  "buckets": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  "years": [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014],
  "samples_per_bucket": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  "archives": [{
    "role": "locally-acquired-clear100",
    "path": "clear100.zip",
    "size_bytes": 123,
    "sha256": "<64 lowercase hex characters>"
  }],
  "provider_archive_checksums_published": false
}
```

The sample counts above are placeholders and must be replaced from the local
prepared metadata. Run only after the data-use/storage review:

```bash
.venv/bin/asi-clear-qualification manifest.json --dataset-root /approved/clear
```

The command reads and hashes local regular files below the root and prints a
plan to stdout. It rejects symlinks, traversal, extra fields, malformed counts,
hash drift, and oversized manifests. It never extracts or writes data.

## Remaining comparability gates

- Resolve a content-addressed provider snapshot and published checksums, or
  archive an independently reviewed acquisition receipt.
- Review YFCC/Flickr asset terms, takedown behavior, and approved storage.
- Parse the prepared metadata and independently verify class and bucket counts;
  archive-byte identity alone does not prove semantic split parity.
- Implement a runner outside the #1578 native-suite adapter, then add image,
  label, transform, metric, JIT/parity, and end-to-end tests.
- Receipt exact model, optimizer, accelerator, preprocessing, and optional
  bucket-0 pretraining costs. The selected control excludes bucket-0
  pretraining; any use needs a separately matched no-pretraining ablation.
- Freeze fresh scientific seeds only after development selection. Nothing in
  this lane can promote a claim or establish SOTA.
