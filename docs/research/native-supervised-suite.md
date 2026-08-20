# Native supervised continual-learning suite

Issue #1578 pins Avalanche at
`ContinualAI/avalanche@eb075be393e1f458b2c352514ff6c17b5a2c0f4e` (MIT; audited
18 August 2026) and its paper at arXiv `2302.01766`. Dataset authorities are the original MNIST
site (60,000 train and 10,000 test 28×28 images) and the University of Toronto CIFAR-100 release
(100 fine classes, 600 images each). ASI's existing IPMNIST anchor remains Elsayed and Mahmood,
ICLR 2024, arXiv `2404.00781`, and its separately audited UPGD code revision; this suite does not
replace the full `upgd_ipmnist` runner.

The additive `asi.native_supervised_cl_development.v1` runner accepts caller-supplied exact
float32 images and int32 labels; it never downloads or writes data. It deterministically constructs
Split MNIST (five ascending class pairs), Rotated MNIST (fixed 0/45/90/135/180 degree rotations),
Split CIFAR-100 (twenty ascending five-class experiences), and IPMNIST (seeded per-task pixel
permutations). Four consumed development seeds bind example order and transformations. Task IDs and
boundaries are never passed to learners. Every arm sees identical task arrays in identical order.

The matched controls are online multinomial SGD, bounded replay SGD, an online running-centroid
classifier, and a literal frozen/no-learning reduction. The SGD arms have the same optimizer-call
budget: replay uses a retained example while its control repeats the current example. Predictions
precede updates. Receipts count
data examples and bytes, model queries, parameter updates, replay inserts/samples/peak bytes,
persistent numeric bytes, exact logical calls, and telemetry-only elapsed nanoseconds. Results are
permanently nonpromoting and negative outcomes must be retained. `asi-native-supervised-catalog
--catalog` reports setup metadata only; dataset execution requires explicit arrays through the API.

## Held-out supplied-array qualification

`native_supervised_qualification.py` adds the
`asi.native_supervised_cl_qualification.v2` development contract without changing the v1 runner.
The caller must provide distinct train and held-out test arrays plus a bounded `DatasetClaims`
record. Inputs are finite exact float32 values in `[0, 1]`; labels are exact int32 values. The
runner snapshots the arrays, hashes the exact dtype, shape, and canonical bytes of each partition,
binds the caller's asset-manifest digest and authority URL, and hashes the exact train and test
schedules. The asset manifest is explicitly marked `caller_asserted_not_verified`: ASI does not
authenticate the claimed upstream assets or infer that arbitrary supplied arrays are canonical
MNIST or CIFAR-100.

Every learner uses one shared output head. It receives neither task IDs nor boundary events. A
separate evaluation harness knows the task partition, pauses training, and evaluates every held-out
task before training and after each task. For `T` tasks the retained matrix therefore has shape
`(T + 1, T)`: row zero is the untrained baseline and row `i + 1` follows task `i`. The summary is
defined exactly as follows:

- final average accuracy is the mean of the final matrix row;
- average forgetting is the task mean of the best post-learning accuracy minus final accuracy;
- forward transfer is the mean, over tasks 1 through `T - 1`, of accuracy immediately before that
  task minus its row-zero accuracy. Task zero is excluded because no prior learning exists.

The forgetting statistic above is the common peak-to-final matrix definition. It is deliberately
not presented as Avalanche's `ExperienceForgetting`, which uses the first post-training value as
its reference at the audited revision. A parity runner must retain and name that external metric
separately rather than silently treating the two definitions as interchangeable.

Train and test sampling use partition-specific JAX Threefry keys. Rotated MNIST retains one fixed
example slice across all five rotations. Task transforms use shared task-specific keys, so IPMNIST
applies the same pixel permutation to both partitions even when their lengths differ. The result
records the RNG and learner/evaluator information contracts, current
module/runtime identities, array and schedule digests, complete train/evaluation query and byte
counts, updates, replay traffic, persistent and peak numeric bytes, score-vector elements, logical
calls, and telemetry-only timing. The peak numeric field is explicitly scoped to simultaneously
retained canonical arrays; it is not process RSS and excludes JAX/NumPy allocator overhead and
transient workspaces. `validate_supplied_array_qualification` snapshots the caller-held arrays
again and reruns the complete shard; every non-timing field must match exactly.

Preflight limits each partition to 100,000 examples, all supplied arrays together to 128 MiB,
flattened inputs to 4,096 values, each task slice to 64 train and 64 test examples, and each arm to
500 million logit multiply-accumulates. Missing claims, absent arrays, unsupported dtypes, unsafe
work, malformed labels, nonfinite or out-of-range inputs, value-identical train/test partitions,
source/runtime drift, or replay mismatch fail closed. This cheap identity check does not prove
sample-level
disjointness; the split remains a bounded caller assertion until canonical loaders and checksums
exist.
`benchmark_qualification_payload` checks whether supplied arrays fit this bounded native slice
without running a learner; it is not launch authorization or external qualification.

This qualification is not Avalanche parity or a scientific benchmark. Avalanche permits shuffled
class orders and task labels; RotatedMNIST uses torchvision/PIL `RandomRotation`, whereas the native
lane freezes an explicit NumPy nearest-neighbor transform. Avalanche Split CIFAR-100 defaults to
random crop and horizontal flip, which this deterministic lane excludes. The bounded runner uses a
linear model, one online pass, small development slices, and no deep external strategies. The v2
matrix closes the native metric-instrumentation gap, but it does not establish official asset
identity, Avalanche metric/transform parity, competitive baseline parity, or a benchmark result.
IPMNIST qualification slices do not match its 200×5,000 full campaign budget.

Before comparison: pin dataset byte checksums and train/test split loaders; reproduce Avalanche task
membership, pixel transforms, normalization and augmentation at the pinned revision; freeze whether
task IDs/boundaries, replay, pretrained features, multi-epoch training, and dynamic heads are allowed;
independently confirm the held-out metric definitions against the selected external protocol; port
strong deep replay and regularization baselines; qualify accelerator timing and compute accounting;
rerun the full existing IPMNIST controls; and preregister untouched scientific seeds. No result or
SOTA claim exists here.

## Canonical asset gate

`native_supervised_canonical.py` adds the additive
`asi.native_supervised_cl_canonical.v3` gate. The caller supplies an absolute local directory; the
runner does not download, extract to disk, or write results. Every path component is opened relative
to an anchored directory descriptor without following symlinks. Asset files must be unique regular
files with one link and must match the frozen official-release size, SHA-256, and MD5 identities.

For MNIST, the loader accepts only the four canonical gzip IDX files, bounds decompression to the
exact header-derived size, verifies magic/count/28×28 shape, and verifies the known 60,000/10,000
class histograms. For CIFAR-100, it accepts only the official 169,001,437-byte Python archive,
checks the exact archive digest before decoding, verifies the upstream train/test member MD5s,
loads fine labels into exact uint8 NCHW arrays, and verifies 500/100 examples for every class. The
pickle decoder is unreachable for any archive that misses the official cryptographic identity.

The loader hashes its exact decoded arrays and selects a deterministic class-balanced bounded slice
before invoking v2, so the full CIFAR payload never violates v2's 128 MiB execution bound. The v3
receipt charges all hashed asset bytes, decoded IDX or tar-member payload bytes, canonical arrays,
retained adapter slices, opened files, and their simultaneous retained loader payload. Its validator
rereads the official files and reexecutes the
complete v2 qualification; only timing telemetry is ignored.

The result separately retains final stream accuracy, prior-task first-post-training forgetting,
prior-task BWT, FWT against the untrained row, and ASI's peak-to-final forgetting. The forgetting
and BWT stream means exclude the last task because Avalanche does not emit either value for an
experience until it has both an initial and a later evaluation. These definitions align
the retained matrix with the declared Avalanche concepts, but no Avalanche code was executed.
Rotated MNIST still uses ASI's deterministic nearest-neighbor rotation rather than torchvision/PIL;
Split CIFAR-100 still omits Avalanche's crop/flip augmentation; the bounded linear arms are not
competitive deep baselines; and IPMNIST is not the full 200×5,000 `[-1, 1]` campaign. Accordingly,
the blocker manifest keeps external transform parity, external metric-implementation parity,
competitive baseline parity, full-horizon IPMNIST parity, accelerator timing, and scientific
promotion false.
