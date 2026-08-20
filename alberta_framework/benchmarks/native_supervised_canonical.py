"""Canonical-asset gate for the bounded native supervised CL adapter.

The v1 runner and v2 supplied-array qualification remain unchanged.  This
module adds the missing trust boundary in front of v2: caller-held official
MNIST or CIFAR-100 bytes must match frozen release digests, decode under an
exact loader contract, and reproduce canonical split/cardinality invariants.
No download, extraction to disk, output write, or scientific promotion occurs.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
import operator
import os
import pickle
import platform
import stat
import struct
import tarfile
from pathlib import Path, PurePath
from typing import SupportsIndex, cast

import jax
import numpy as np

from alberta_framework.benchmarks import native_supervised_qualification as supplied
from alberta_framework.benchmarks import native_supervised_suite as native

SCHEMA = "asi.native_supervised_cl_canonical.v3"
BLOCKER_SCHEMA = "asi.native_supervised_cl_canonical.blockers.v1"
MAX_ASSET_BYTES = 200_000_000
MAX_TOTAL_ASSET_BYTES = 200_000_000
ASSET_CONTRACT = "official_release_sha256_md5_size_and_exact_loader_invariants.v1"
METRIC_CONTRACT = (
    "heldout_matrix:final_stream_accuracy;prior_task_first_post_training_forgetting;"
    "prior_task_backward_transfer;forward_transfer;peak_to_final_forgetting.v2"
)
TRANSFORM_PARITY = "native_deterministic_transform_not_audited_external_execution"
METRIC_PARITY = "definitions_encoded_external_implementation_not_executed"


class CanonicalQualificationError(ValueError):
    """A canonical asset or loader invariant failed closed."""


def _exact_int(value: object, name: str, low: int, high: int) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact integer")
    result = operator.index(cast(SupportsIndex, value))
    if not low <= result <= high:
        raise ValueError(f"{name} must lie in [{low}, {high}]")
    return result


def _digest(value: object, name: str, length: int) -> str:
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal digest")
    return value


def _bounded_string(value: object, name: str, limit: int = 512) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty exact string")
    if len(value.encode("utf-8", errors="strict")) > limit:
        raise ValueError(f"{name} exceeds its UTF-8 byte bound")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class AssetSpec:
    name: str
    size_bytes: int
    sha256: str
    md5: str

    def __post_init__(self) -> None:
        name = _bounded_string(self.name, "asset name", 128)
        if PurePath(name).name != name or name in (".", ".."):
            raise ValueError("asset name must be one plain path component")
        _exact_int(self.size_bytes, "asset size", 1, MAX_ASSET_BYTES)
        _digest(self.sha256, "asset sha256", 64)
        _digest(self.md5, "asset md5", 32)


@dataclasses.dataclass(frozen=True, slots=True)
class CanonicalDefinition:
    dataset_id: str
    authority_uri: str
    files: tuple[AssetSpec, ...]
    train_examples: int
    test_examples: int
    image_shape: tuple[int, ...]
    n_classes: int
    train_class_histogram: tuple[int, ...]
    test_class_histogram: tuple[int, ...]
    loader_contract: str

    def __post_init__(self) -> None:
        if type(self.dataset_id) is not str or self.dataset_id not in ("mnist", "cifar100"):
            raise ValueError("dataset_id must identify MNIST or CIFAR-100")
        uri = _bounded_string(self.authority_uri, "authority_uri")
        if not uri.startswith("https://"):
            raise ValueError("authority_uri must be HTTPS")
        if type(self.files) is not tuple or not self.files:
            raise ValueError("files must be a non-empty exact tuple")
        if any(type(value) is not AssetSpec for value in self.files):
            raise ValueError("files must contain exact AssetSpec values")
        for value in self.files:
            AssetSpec.__post_init__(value)
        names = tuple(value.name for value in self.files)
        if len(set(names)) != len(names):
            raise ValueError("asset names must be unique")
        if sum(value.size_bytes for value in self.files) > MAX_TOTAL_ASSET_BYTES:
            raise ValueError("asset set exceeds the aggregate byte bound")
        _exact_int(self.train_examples, "train_examples", 1, 100_000)
        _exact_int(self.test_examples, "test_examples", 1, 100_000)
        if (
            type(self.image_shape) is not tuple
            or not self.image_shape
            or any(type(value) is not int or not 1 <= value <= 4096 for value in self.image_shape)
            or math.prod(self.image_shape) > native.MAX_INPUT_DIM
        ):
            raise ValueError("image_shape must be a bounded exact integer tuple")
        _exact_int(self.n_classes, "n_classes", 2, 100)
        for name, histogram, total in (
            ("train histogram", self.train_class_histogram, self.train_examples),
            ("test histogram", self.test_class_histogram, self.test_examples),
        ):
            if (
                type(histogram) is not tuple
                or len(histogram) != self.n_classes
                or any(type(value) is not int or value <= 0 for value in histogram)
                or sum(histogram) != total
            ):
                raise ValueError(f"{name} differs from the canonical split")
        _bounded_string(self.loader_contract, "loader_contract")


_MNIST_FILES = (
    AssetSpec(
        "train-images-idx3-ubyte.gz",
        9_912_422,
        "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
        "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    ),
    AssetSpec(
        "train-labels-idx1-ubyte.gz",
        28_881,
        "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
        "d53e105ee54ea40749a09fcbcd1e9432",
    ),
    AssetSpec(
        "t10k-images-idx3-ubyte.gz",
        1_648_877,
        "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
        "9fb629c4189551a2d022fa330f9573f3",
    ),
    AssetSpec(
        "t10k-labels-idx1-ubyte.gz",
        4_542,
        "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
        "ec29112dd5afa0611ce80d1b7f02629c",
    ),
)
_CIFAR_FILES = (
    AssetSpec(
        "cifar-100-python.tar.gz",
        169_001_437,
        "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7",
        "eb9058c3a382ffc7106e4002c42a8d85",
    ),
)
_MNIST_TRAIN_HISTOGRAM = (5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949)
_MNIST_TEST_HISTOGRAM = (980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009)


def canonical_definition(benchmark_id: object) -> CanonicalDefinition:
    spec = native.benchmark_spec(benchmark_id)
    if spec.benchmark_id == "split_cifar100":
        return CanonicalDefinition(
            dataset_id="cifar100",
            authority_uri="https://www.cs.toronto.edu/~kriz/cifar.html",
            files=_CIFAR_FILES,
            train_examples=50_000,
            test_examples=10_000,
            image_shape=(3, 32, 32),
            n_classes=100,
            train_class_histogram=(500,) * 100,
            test_class_histogram=(100,) * 100,
            loader_contract=(
                "official Python archive; exact train/test member MD5; uint8 NCHW; "
                "fine_labels; raw/255 float32"
            ),
        )
    return CanonicalDefinition(
        dataset_id="mnist",
        authority_uri="https://yann.lecun.com/exdb/mnist/",
        files=_MNIST_FILES,
        train_examples=60_000,
        test_examples=10_000,
        image_shape=(28, 28),
        n_classes=10,
        train_class_histogram=_MNIST_TRAIN_HISTOGRAM,
        test_class_histogram=_MNIST_TEST_HISTOGRAM,
        loader_contract="official gzip IDX; uint8 NHW; raw/255 float32",
    )


@dataclasses.dataclass(frozen=True, slots=True)
class AssetReceipt:
    name: str
    size_bytes: int
    sha256: str
    md5: str

    def __post_init__(self) -> None:
        AssetSpec(self.name, self.size_bytes, self.sha256, self.md5)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns)


def _directory_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mode)


def _open_anchored_root(root: Path) -> tuple[int, tuple[int, int, int]]:
    raw = os.fspath(root)
    if not os.path.isabs(raw):
        raise CanonicalQualificationError("asset root must be absolute")
    parts = Path(raw).parts
    if any(part in (".", "..") for part in parts[1:]):
        raise CanonicalQualificationError("asset root must be lexically normalized")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    current = os.open("/", flags)
    try:
        for part in parts[1:]:
            before = os.stat(part, dir_fd=current, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode):
                raise CanonicalQualificationError(
                    "asset root has a symlink or non-directory ancestor"
                )
            following = os.open(part, flags | getattr(os, "O_NOFOLLOW", 0), dir_fd=current)
            opened = os.fstat(following)
            if _directory_identity(before) != _directory_identity(opened):
                os.close(following)
                raise CanonicalQualificationError("asset root ancestor changed while opening")
            os.close(current)
            current = following
        metadata = os.fstat(current)
        return current, _directory_identity(metadata)
    except BaseException:
        os.close(current)
        raise


def _read_asset(root: Path, spec: AssetSpec) -> tuple[AssetReceipt, bytes]:
    AssetSpec.__post_init__(spec)
    descriptor, root_identity = _open_anchored_root(root)
    try:
        before = os.stat(spec.name, dir_fd=descriptor, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise CanonicalQualificationError("asset must be a regular non-symlink file")
        if before.st_nlink != 1:
            raise CanonicalQualificationError("asset hardlink count must equal one")
        if before.st_size != spec.size_bytes:
            raise CanonicalQualificationError("asset size differs from the frozen release")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            file_descriptor = os.open(spec.name, flags, dir_fd=descriptor)
        except OSError as exc:
            raise CanonicalQualificationError(
                "asset open rejected a symlink or unsafe entry"
            ) from exc
        try:
            opened = os.fstat(file_descriptor)
            if _stat_identity(before) != _stat_identity(opened) or opened.st_nlink != 1:
                raise CanonicalQualificationError("asset identity changed while opening")
            chunks: list[bytes] = []
            observed = 0
            while True:
                chunk = os.read(file_descriptor, min(1 << 20, spec.size_bytes + 1 - observed))
                if not chunk:
                    break
                observed += len(chunk)
                if observed > spec.size_bytes:
                    raise CanonicalQualificationError("asset grew beyond its frozen size")
                chunks.append(chunk)
            after = os.fstat(file_descriptor)
            current = os.stat(spec.name, dir_fd=descriptor, follow_symlinks=False)
            if (
                observed != spec.size_bytes
                or _stat_identity(opened) != _stat_identity(after)
                or _stat_identity(opened) != _stat_identity(current)
                or _directory_identity(os.fstat(descriptor)) != root_identity
            ):
                raise CanonicalQualificationError(
                    "asset or containing directory changed while reading"
                )
        finally:
            os.close(file_descriptor)
    except OSError as exc:
        raise CanonicalQualificationError("asset filesystem inspection failed closed") from exc
    finally:
        os.close(descriptor)
    final_descriptor, final_root_identity = _open_anchored_root(root)
    os.close(final_descriptor)
    if final_root_identity != root_identity:
        raise CanonicalQualificationError("asset root directory entry changed while reading")
    raw = b"".join(chunks)
    sha256 = hashlib.sha256(raw).hexdigest()
    md5 = hashlib.md5(raw, usedforsecurity=False).hexdigest()  # noqa: S324
    if sha256 != spec.sha256 or md5 != spec.md5:
        raise CanonicalQualificationError("asset digest differs from the frozen release")
    return AssetReceipt(spec.name, len(raw), sha256, md5), raw


def _decompress_exact(raw: bytes, expected_bytes: int, name: str) -> bytes:
    _exact_int(expected_bytes, "expected decoded bytes", 1, 100_000_000)
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as stream:
            decoded = stream.read(expected_bytes + 1)
    except (EOFError, OSError) as exc:
        raise CanonicalQualificationError(f"{name} gzip stream is malformed") from exc
    if len(decoded) != expected_bytes:
        raise CanonicalQualificationError(f"{name} decoded length differs from the IDX contract")
    return decoded


def _decode_idx_images(raw: bytes, count: int, shape: tuple[int, int]) -> np.ndarray:
    expected = 16 + count * math.prod(shape)
    if len(raw) != expected:
        raise CanonicalQualificationError("IDX image payload length mismatch")
    magic, observed_count, rows, columns = struct.unpack(">IIII", raw[:16])
    if magic != 2051:
        raise CanonicalQualificationError("IDX image magic mismatch")
    if (observed_count, rows, columns) != (count, *shape):
        raise CanonicalQualificationError("IDX image shape/count mismatch")
    return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape((count, *shape)).copy()


def _decode_idx_labels(raw: bytes, count: int) -> np.ndarray:
    if len(raw) != 8 + count:
        raise CanonicalQualificationError("IDX label payload length mismatch")
    magic, observed_count = struct.unpack(">II", raw[:8])
    if magic != 2049:
        raise CanonicalQualificationError("IDX label magic mismatch")
    if observed_count != count:
        raise CanonicalQualificationError("IDX label count mismatch")
    return np.frombuffer(raw, dtype=np.uint8, offset=8).copy()


def _load_mnist(
    root: Path, definition: CanonicalDefinition
) -> tuple[tuple[AssetReceipt, ...], int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loaded = tuple(_read_asset(root, spec) for spec in definition.files)
    by_name = {receipt.name: raw for receipt, raw in loaded}
    required = {
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    }
    if set(by_name) != required:
        raise CanonicalQualificationError(
            "MNIST asset roster differs from the exact loader contract"
        )
    if len(definition.image_shape) != 2:
        raise CanonicalQualificationError("MNIST image shape must have rank two")
    shape = definition.image_shape
    train_images_raw = _decompress_exact(
        by_name["train-images-idx3-ubyte.gz"],
        16 + definition.train_examples * math.prod(shape),
        "train images",
    )
    train_labels_raw = _decompress_exact(
        by_name["train-labels-idx1-ubyte.gz"], 8 + definition.train_examples, "train labels"
    )
    test_images_raw = _decompress_exact(
        by_name["t10k-images-idx3-ubyte.gz"],
        16 + definition.test_examples * math.prod(shape),
        "test images",
    )
    test_labels_raw = _decompress_exact(
        by_name["t10k-labels-idx1-ubyte.gz"], 8 + definition.test_examples, "test labels"
    )
    return (
        tuple(receipt for receipt, _ in loaded),
        sum(
            len(value)
            for value in (
                train_images_raw,
                train_labels_raw,
                test_images_raw,
                test_labels_raw,
            )
        ),
        _decode_idx_images(train_images_raw, definition.train_examples, shape),
        _decode_idx_labels(train_labels_raw, definition.train_examples),
        _decode_idx_images(test_images_raw, definition.test_examples, shape),
        _decode_idx_labels(test_labels_raw, definition.test_examples),
    )


def _tar_member_bytes(archive: tarfile.TarFile, name: str, expected_md5: str) -> bytes:
    try:
        member = archive.getmember(name)
    except KeyError as exc:
        raise CanonicalQualificationError(f"CIFAR archive lacks {name}") from exc
    if (
        not member.isfile()
        or member.issym()
        or member.islnk()
        or not 1 <= member.size <= 200_000_000
    ):
        raise CanonicalQualificationError("CIFAR member is not a bounded regular file")
    extracted = archive.extractfile(member)
    if extracted is None:
        raise CanonicalQualificationError("CIFAR member could not be read")
    raw = extracted.read(member.size + 1)
    if len(raw) != member.size:
        raise CanonicalQualificationError("CIFAR member length mismatch")
    if hashlib.md5(raw, usedforsecurity=False).hexdigest() != expected_md5:  # noqa: S324
        raise CanonicalQualificationError("CIFAR member digest mismatch")
    return raw


def _decode_cifar_pickle(raw: bytes, count: int) -> tuple[np.ndarray, np.ndarray]:
    # Pickle is admitted only after the enclosing archive and member match exact
    # official release digests. Arbitrary caller bytes never reach this decoder.
    try:
        value = pickle.loads(raw, encoding="bytes")  # noqa: S301
    except (pickle.PickleError, EOFError, AttributeError, ValueError) as exc:
        raise CanonicalQualificationError("official CIFAR pickle failed to decode") from exc
    if type(value) is not dict or b"data" not in value or b"fine_labels" not in value:
        raise CanonicalQualificationError("CIFAR pickle lacks exact data/fine-label fields")
    data = value[b"data"]
    labels = value[b"fine_labels"]
    if (
        type(data) is not np.ndarray
        or data.dtype != np.uint8
        or data.shape != (count, 3072)
        or type(labels) is not list
        or len(labels) != count
        or any(type(label) is not int for label in labels)
    ):
        raise CanonicalQualificationError("CIFAR pickle differs from the canonical array contract")
    return data.reshape((count, 3, 32, 32)).copy(), np.asarray(labels, dtype=np.uint8)


def _load_cifar(
    root: Path, definition: CanonicalDefinition
) -> tuple[tuple[AssetReceipt, ...], int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(definition.files) != 1:
        raise CanonicalQualificationError("CIFAR requires exactly one official archive")
    receipt, raw = _read_asset(root, definition.files[0])
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
            train_raw = _tar_member_bytes(
                archive, "cifar-100-python/train", "16019d7e3df5f24257cddd939b257f8d"
            )
            test_raw = _tar_member_bytes(
                archive, "cifar-100-python/test", "f0ef6b0ae62326f3e7ffdfab6717acfc"
            )
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise CanonicalQualificationError("official CIFAR archive failed to decode") from exc
    train_images, train_labels = _decode_cifar_pickle(train_raw, definition.train_examples)
    test_images, test_labels = _decode_cifar_pickle(test_raw, definition.test_examples)
    return (
        (receipt,),
        len(train_raw) + len(test_raw),
        train_images,
        train_labels,
        test_images,
        test_labels,
    )


def _array_digest(value: np.ndarray, label: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"asi-canonical-loader-array-v3\0")
    digest.update(label.encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _histogram(labels: np.ndarray, n_classes: int) -> tuple[int, ...]:
    if labels.dtype != np.uint8 or np.any(labels >= n_classes):
        raise CanonicalQualificationError("labels lie outside the canonical class range")
    return tuple(int(value) for value in np.bincount(labels, minlength=n_classes))


@dataclasses.dataclass(frozen=True, slots=True)
class CanonicalAssetBinding:
    dataset_id: str
    authority_uri: str
    asset_contract: str
    assets: tuple[AssetReceipt, ...]
    train_examples: int
    test_examples: int
    image_shape: tuple[int, ...]
    train_class_histogram: tuple[int, ...]
    test_class_histogram: tuple[int, ...]
    train_loader_output_sha256: str
    test_loader_output_sha256: str
    official_asset_digests_verified: bool = True
    canonical_loader_invariants_verified: bool = True

    def __post_init__(self) -> None:
        if type(self.dataset_id) is not str or self.dataset_id not in ("mnist", "cifar100"):
            raise ValueError("unknown canonical dataset identity")
        _bounded_string(self.authority_uri, "authority_uri")
        if type(self.asset_contract) is not str or self.asset_contract != ASSET_CONTRACT:
            raise ValueError("asset verification contract drift")
        if type(self.assets) is not tuple or not self.assets:
            raise ValueError("asset receipts must be a non-empty tuple")
        for receipt in self.assets:
            if type(receipt) is not AssetReceipt:
                raise ValueError("asset receipts must contain exact AssetReceipt values")
            AssetReceipt.__post_init__(receipt)
        _exact_int(self.train_examples, "train_examples", 1, 100_000)
        _exact_int(self.test_examples, "test_examples", 1, 100_000)
        if type(self.image_shape) is not tuple or not self.image_shape:
            raise ValueError("image_shape must be an exact non-empty tuple")
        expected_classes = 10 if self.dataset_id == "mnist" else 100
        for histogram, total in (
            (self.train_class_histogram, self.train_examples),
            (self.test_class_histogram, self.test_examples),
        ):
            if (
                type(histogram) is not tuple
                or len(histogram) != expected_classes
                or any(type(count) is not int or count <= 0 for count in histogram)
                or sum(histogram) != total
            ):
                raise ValueError("canonical histogram total mismatch")
        _digest(self.train_loader_output_sha256, "train loader output", 64)
        _digest(self.test_loader_output_sha256, "test loader output", 64)
        flags = (self.official_asset_digests_verified, self.canonical_loader_invariants_verified)
        if any(type(flag) is not bool or not flag for flag in flags):
            raise ValueError("canonical assets and loader invariants must be verified")


@dataclasses.dataclass(frozen=True, slots=True)
class CanonicalResourceReceipt:
    asset_bytes_hashed: int
    decoded_payload_bytes: int
    canonical_array_bytes: int
    adapter_slice_bytes: int
    peak_loader_payload_bytes: int
    asset_files_opened: int

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            _exact_int(getattr(self, field.name), field.name, 1, 2**63 - 1)
        if self.peak_loader_payload_bytes < max(
            self.asset_bytes_hashed + self.decoded_payload_bytes + self.canonical_array_bytes,
            self.canonical_array_bytes + self.adapter_slice_bytes,
        ):
            raise ValueError("peak loader payload omits simultaneously retained bytes")


def _expected_mnist_decoded_payload_bytes(definition: CanonicalDefinition) -> int:
    if definition.dataset_id == "mnist":
        pixels = math.prod(definition.image_shape)
        return (
            16
            + definition.train_examples * pixels
            + 8
            + definition.train_examples
            + 16
            + definition.test_examples * pixels
            + 8
            + definition.test_examples
        )
    raise ValueError("decoded IDX accounting only applies to MNIST")


def _expected_resources(
    definition: CanonicalDefinition,
    train_examples_per_task: int,
    test_examples_per_task: int,
    decoded_payload_bytes: int,
) -> CanonicalResourceReceipt:
    asset_bytes = sum(asset.size_bytes for asset in definition.files)
    pixels = math.prod(definition.image_shape)
    canonical_bytes = (definition.train_examples + definition.test_examples) * (pixels + 1)
    adapter_bytes = (
        definition.n_classes
        * (train_examples_per_task + test_examples_per_task)
        * (pixels * np.dtype(np.float32).itemsize + np.dtype(np.int32).itemsize)
    )
    return CanonicalResourceReceipt(
        asset_bytes_hashed=asset_bytes,
        decoded_payload_bytes=decoded_payload_bytes,
        canonical_array_bytes=canonical_bytes,
        adapter_slice_bytes=adapter_bytes,
        peak_loader_payload_bytes=max(
            asset_bytes + decoded_payload_bytes + canonical_bytes,
            canonical_bytes + adapter_bytes,
        ),
        asset_files_opened=len(definition.files),
    )


def _probability(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite exact probability")
    return value


def _signed_probability(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite exact signed probability")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class AvalancheMatrixMetrics:
    final_stream_accuracy: float
    first_post_training_forgetting: float
    backward_transfer: float
    forward_transfer: float
    peak_to_final_forgetting: float

    def __post_init__(self) -> None:
        _probability(self.final_stream_accuracy, "final_stream_accuracy")
        _signed_probability(self.first_post_training_forgetting, "first_post_training_forgetting")
        _signed_probability(self.backward_transfer, "backward_transfer")
        _signed_probability(self.forward_transfer, "forward_transfer")
        _probability(self.peak_to_final_forgetting, "peak_to_final_forgetting")


def avalanche_matrix_metrics(matrix: object) -> AvalancheMatrixMetrics:
    if type(matrix) is not tuple or not matrix:
        raise ValueError("accuracy matrix must be an exact non-empty tuple")
    tasks = len(matrix[0])
    if tasks == 0 or len(matrix) != tasks + 1:
        raise ValueError("accuracy matrix must have shape (tasks + 1, tasks)")
    for row in matrix:
        if type(row) is not tuple or len(row) != tasks:
            raise ValueError("accuracy matrix rows have inconsistent shape")
        for value in row:
            _probability(value, "accuracy matrix entry")
    final = float(sum(matrix[-1]) / tasks)
    forgetting = float(
        sum(matrix[task + 1][task] - matrix[-1][task] for task in range(tasks - 1))
        / max(tasks - 1, 1)
    )
    backward = float(
        sum(matrix[-1][task] - matrix[task + 1][task] for task in range(tasks - 1))
        / max(tasks - 1, 1)
    )
    forward = float(
        sum(matrix[task][task] - matrix[0][task] for task in range(1, tasks)) / max(tasks - 1, 1)
    )
    peak = float(
        sum(
            max(row[task] for row in matrix[task + 1 :]) - matrix[-1][task] for task in range(tasks)
        )
        / tasks
    )
    return AvalancheMatrixMetrics(final, forgetting, backward, forward, peak)


def _module_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_LOADED_SOURCE_IDENTITY = (
    ("alberta_framework/benchmarks/native_supervised_canonical.py", _module_sha256(Path(__file__))),
    (
        "alberta_framework/benchmarks/native_supervised_qualification.py",
        _module_sha256(Path(supplied.__file__)),
    ),
    (
        "alberta_framework/benchmarks/native_supervised_suite.py",
        _module_sha256(Path(native.__file__)),
    ),
)


def _source_identity() -> tuple[tuple[str, str], ...]:
    current = (
        (
            "alberta_framework/benchmarks/native_supervised_canonical.py",
            _module_sha256(Path(__file__)),
        ),
        (
            "alberta_framework/benchmarks/native_supervised_qualification.py",
            _module_sha256(Path(supplied.__file__)),
        ),
        (
            "alberta_framework/benchmarks/native_supervised_suite.py",
            _module_sha256(Path(native.__file__)),
        ),
    )
    if current != _LOADED_SOURCE_IDENTITY:
        raise ValueError("canonical qualification source changed after import")
    return current


def _runtime_identity() -> tuple[tuple[str, str], ...]:
    devices = jax.devices()
    if not devices:
        raise ValueError("JAX exposes no devices")
    return (
        ("python", platform.python_version()),
        ("jax", jax.__version__),
        ("jaxlib", importlib.metadata.version("jaxlib")),
        ("numpy", np.__version__),
        ("backend", jax.default_backend()),
        ("device_kind", devices[0].device_kind),
        ("jax_enable_x64", str(bool(jax.config.jax_enable_x64)).lower()),
        ("jax_default_matmul_precision", str(jax.config.jax_default_matmul_precision)),
        ("jax_numpy_dtype_promotion", str(jax.config.jax_numpy_dtype_promotion)),
        ("operating_system", platform.system()),
        ("machine", platform.machine()),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class CanonicalQualification:
    schema: str
    benchmark_id: str
    seed: int
    train_examples_per_task: int
    test_examples_per_task: int
    replay_capacity: int
    comparison_reference: str
    asset_binding: CanonicalAssetBinding
    resources: CanonicalResourceReceipt
    qualification: supplied.SuppliedArrayQualification
    avalanche_metrics: tuple[AvalancheMatrixMetrics, ...]
    metric_contract: str
    transform_parity_contract: str
    metric_parity_contract: str
    source_identity: tuple[tuple[str, str], ...]
    runtime_identity: tuple[tuple[str, str], ...]
    external_transform_parity: bool = False
    external_metric_implementation_parity: bool = False
    development_only: bool = True
    scientific_promotion_allowed: bool = False
    negative_results_must_be_retained: bool = True

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != SCHEMA:
            raise ValueError("canonical qualification schema mismatch")
        native.benchmark_spec(self.benchmark_id)
        definition = canonical_definition(self.benchmark_id)
        CanonicalDefinition.__post_init__(definition)
        _exact_int(self.seed, "seed", 0, 2**32 - 1)
        _exact_int(self.train_examples_per_task, "train_examples_per_task", 1, 64)
        _exact_int(self.test_examples_per_task, "test_examples_per_task", 1, 64)
        _exact_int(self.replay_capacity, "replay_capacity", 1, 64)
        if self.comparison_reference != native.AVALANCHE_REVISION:
            raise ValueError("Avalanche comparison revision drift")
        if type(self.asset_binding) is not CanonicalAssetBinding:
            raise ValueError("asset_binding must be exact")
        CanonicalAssetBinding.__post_init__(self.asset_binding)
        expected_assets = tuple(
            AssetReceipt(asset.name, asset.size_bytes, asset.sha256, asset.md5)
            for asset in definition.files
        )
        if (
            self.asset_binding.dataset_id != definition.dataset_id
            or self.asset_binding.authority_uri != definition.authority_uri
            or self.asset_binding.assets != expected_assets
            or self.asset_binding.train_examples != definition.train_examples
            or self.asset_binding.test_examples != definition.test_examples
            or self.asset_binding.image_shape != definition.image_shape
            or self.asset_binding.train_class_histogram != definition.train_class_histogram
            or self.asset_binding.test_class_histogram != definition.test_class_histogram
        ):
            raise ValueError("asset binding differs from the canonical definition")
        if type(self.resources) is not CanonicalResourceReceipt:
            raise ValueError("resources must be exact")
        CanonicalResourceReceipt.__post_init__(self.resources)
        decoded_bytes = self.resources.decoded_payload_bytes
        if definition.dataset_id == "mnist":
            decoded_bytes = _expected_mnist_decoded_payload_bytes(definition)
        elif decoded_bytes < (
            definition.train_examples + definition.test_examples
        ) * (math.prod(definition.image_shape) + 1):
            raise ValueError("decoded CIFAR payload omits its canonical numeric arrays")
        if self.resources != _expected_resources(
            definition,
            self.train_examples_per_task,
            self.test_examples_per_task,
            decoded_bytes,
        ):
            raise ValueError("resource receipt differs from the exact canonical payload")
        if type(self.qualification) is not supplied.SuppliedArrayQualification:
            raise ValueError("nested qualification must be exact")
        supplied.SuppliedArrayQualification.__post_init__(self.qualification)
        if (
            self.qualification.benchmark_id != self.benchmark_id
            or self.qualification.seed != self.seed
        ):
            raise ValueError("nested qualification identity mismatch")
        if type(self.avalanche_metrics) is not tuple or len(self.avalanche_metrics) != len(
            native.ARM_IDS
        ):
            raise ValueError("Avalanche metric roster mismatch")
        for metric in self.avalanche_metrics:
            if type(metric) is not AvalancheMatrixMetrics:
                raise ValueError("Avalanche metrics must be exact")
            AvalancheMatrixMetrics.__post_init__(metric)
        expected_metrics = tuple(
            avalanche_matrix_metrics(arm.accuracy_matrix) for arm in self.qualification.arms
        )
        if self.avalanche_metrics != expected_metrics:
            raise ValueError("Avalanche metrics do not replay from the held-out matrix")
        if self.metric_contract != METRIC_CONTRACT:
            raise ValueError("metric contract drift")
        if (
            self.transform_parity_contract != TRANSFORM_PARITY
            or self.metric_parity_contract != METRIC_PARITY
        ):
            raise ValueError("external parity contract drift")
        if self.source_identity != _source_identity():
            raise ValueError("current canonical qualification source drift")
        if self.runtime_identity != _runtime_identity():
            raise ValueError("current canonical qualification runtime drift")
        flags = (
            not self.external_transform_parity,
            not self.external_metric_implementation_parity,
            self.development_only,
            not self.scientific_promotion_allowed,
            self.negative_results_must_be_retained,
        )
        if any(type(flag) is not bool or not flag for flag in flags):
            raise ValueError("canonical qualification must remain nonpromoting and parity-honest")


def _manifest_sha256(definition: CanonicalDefinition) -> str:
    value = {
        "dataset_id": definition.dataset_id,
        "authority_uri": definition.authority_uri,
        "files": [dataclasses.asdict(asset) for asset in definition.files],
        "loader_contract": definition.loader_contract,
        "train_examples": definition.train_examples,
        "test_examples": definition.test_examples,
        "image_shape": list(definition.image_shape),
        "train_histogram": list(definition.train_class_histogram),
        "test_histogram": list(definition.test_class_histogram),
    }
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _load_verified(
    benchmark_id: str, root: Path
) -> tuple[
    CanonicalDefinition, CanonicalAssetBinding, CanonicalResourceReceipt, tuple[np.ndarray, ...]
]:
    definition = canonical_definition(benchmark_id)
    CanonicalDefinition.__post_init__(definition)
    if definition.dataset_id == "mnist":
        assets, decoded_bytes, train_images, train_labels, test_images, test_labels = _load_mnist(
            root, definition
        )
    else:
        assets, decoded_bytes, train_images, train_labels, test_images, test_labels = _load_cifar(
            root, definition
        )
    if train_images.shape != (definition.train_examples, *definition.image_shape):
        raise CanonicalQualificationError("train image shape differs from the canonical split")
    if test_images.shape != (definition.test_examples, *definition.image_shape):
        raise CanonicalQualificationError("test image shape differs from the canonical split")
    train_histogram = _histogram(train_labels, definition.n_classes)
    test_histogram = _histogram(test_labels, definition.n_classes)
    if train_histogram != definition.train_class_histogram:
        raise CanonicalQualificationError("train class histogram differs from the canonical split")
    if test_histogram != definition.test_class_histogram:
        raise CanonicalQualificationError("test class histogram differs from the canonical split")
    train_sha = _array_digest(train_images, "train_images")
    test_sha = _array_digest(test_images, "test_images")
    if train_sha == test_sha:
        raise CanonicalQualificationError("canonical train and test loader outputs are identical")
    binding = CanonicalAssetBinding(
        dataset_id=definition.dataset_id,
        authority_uri=definition.authority_uri,
        asset_contract=ASSET_CONTRACT,
        assets=assets,
        train_examples=definition.train_examples,
        test_examples=definition.test_examples,
        image_shape=definition.image_shape,
        train_class_histogram=train_histogram,
        test_class_histogram=test_histogram,
        train_loader_output_sha256=train_sha,
        test_loader_output_sha256=test_sha,
    )
    canonical_bytes = sum(
        array.nbytes for array in (train_images, train_labels, test_images, test_labels)
    )
    asset_bytes = sum(asset.size_bytes for asset in assets)
    placeholder = CanonicalResourceReceipt(
        asset_bytes_hashed=asset_bytes,
        decoded_payload_bytes=decoded_bytes,
        canonical_array_bytes=canonical_bytes,
        adapter_slice_bytes=1,
        peak_loader_payload_bytes=max(
            asset_bytes + decoded_bytes + canonical_bytes, canonical_bytes + 1
        ),
        asset_files_opened=len(assets),
    )
    return definition, binding, placeholder, (train_images, train_labels, test_images, test_labels)


def _bounded_class_slice(
    images: np.ndarray, labels: np.ndarray, n_classes: int, count: int
) -> tuple[np.ndarray, np.ndarray]:
    indices: list[int] = []
    for class_id in range(n_classes):
        eligible = np.flatnonzero(labels == class_id)
        if eligible.size < count:
            raise CanonicalQualificationError(
                "canonical split lacks a bounded per-class adapter slice"
            )
        indices.extend(int(value) for value in eligible[:count])
    selected = np.asarray(indices, dtype=np.int64)
    return (
        np.ascontiguousarray(images[selected].astype(np.float32) / np.float32(255.0)),
        np.ascontiguousarray(labels[selected].astype(np.int32)),
    )


def run_canonical_asset_qualification(
    benchmark_id: object,
    asset_root: object,
    *,
    seed: object,
    train_examples_per_task: object = 8,
    test_examples_per_task: object = 1,
    replay_capacity: object = 16,
) -> CanonicalQualification:
    spec = native.benchmark_spec(benchmark_id)
    if type(asset_root) is not type(Path()):
        raise ValueError("asset_root must be an exact pathlib.Path")
    host_seed = _exact_int(seed, "seed", 0, 2**32 - 1)
    train_count = _exact_int(train_examples_per_task, "train_examples_per_task", 1, 64)
    test_count = _exact_int(test_examples_per_task, "test_examples_per_task", 1, 64)
    capacity = _exact_int(replay_capacity, "replay_capacity", 1, 64)
    definition, binding, base_resources, arrays = _load_verified(spec.benchmark_id, asset_root)
    train_images, train_labels = _bounded_class_slice(
        arrays[0], arrays[1], definition.n_classes, train_count
    )
    test_images, test_labels = _bounded_class_slice(
        arrays[2], arrays[3], definition.n_classes, test_count
    )
    adapter_bytes = sum(
        array.nbytes for array in (train_images, train_labels, test_images, test_labels)
    )
    resources = dataclasses.replace(
        base_resources,
        adapter_slice_bytes=adapter_bytes,
        peak_loader_payload_bytes=max(
            base_resources.asset_bytes_hashed
            + base_resources.decoded_payload_bytes
            + base_resources.canonical_array_bytes,
            base_resources.canonical_array_bytes + adapter_bytes,
        ),
    )
    claims = supplied.DatasetClaims(
        benchmark_id=spec.benchmark_id,
        authority_uri=definition.authority_uri,
        asset_manifest_sha256=_manifest_sha256(definition),
        split_contract=(
            "v3 verified official release and exact loader; deterministic class-balanced bounded "
            "slice feeds v2; full external transform parity remains open"
        ),
    )
    nested = supplied.run_supplied_array_qualification(
        spec.benchmark_id,
        train_images,
        train_labels,
        test_images,
        test_labels,
        claims=claims,
        seed=host_seed,
        train_examples_per_task=train_count,
        test_examples_per_task=test_count,
        replay_capacity=capacity,
    )
    result = CanonicalQualification(
        schema=SCHEMA,
        benchmark_id=spec.benchmark_id,
        seed=host_seed,
        train_examples_per_task=train_count,
        test_examples_per_task=test_count,
        replay_capacity=capacity,
        comparison_reference=native.AVALANCHE_REVISION,
        asset_binding=binding,
        resources=resources,
        qualification=nested,
        avalanche_metrics=tuple(
            avalanche_matrix_metrics(arm.accuracy_matrix) for arm in nested.arms
        ),
        metric_contract=METRIC_CONTRACT,
        transform_parity_contract=TRANSFORM_PARITY,
        metric_parity_contract=METRIC_PARITY,
        source_identity=_source_identity(),
        runtime_identity=_runtime_identity(),
    )
    CanonicalQualification.__post_init__(result)
    return result


def _without_timing(value: CanonicalQualification) -> CanonicalQualification:
    arms = tuple(
        dataclasses.replace(arm, receipt=dataclasses.replace(arm.receipt, elapsed_ns=0))
        for arm in value.qualification.arms
    )
    return dataclasses.replace(
        value, qualification=dataclasses.replace(value.qualification, arms=arms)
    )


def validate_canonical_asset_qualification(
    value: object, asset_root: object
) -> CanonicalQualification:
    if type(value) is not CanonicalQualification:
        raise ValueError("result must be an exact CanonicalQualification")
    CanonicalQualification.__post_init__(value)
    replay = run_canonical_asset_qualification(
        value.benchmark_id,
        asset_root,
        seed=value.seed,
        train_examples_per_task=value.train_examples_per_task,
        test_examples_per_task=value.test_examples_per_task,
        replay_capacity=value.replay_capacity,
    )
    if _without_timing(value) != _without_timing(replay):
        raise ValueError("canonical qualification replay differs from the retained result")
    return value


def qualification_blocker_manifest() -> dict[str, object]:
    return {
        "schema": BLOCKER_SCHEMA,
        "avalanche_revision": native.AVALANCHE_REVISION,
        "official_assets_verified_by_runner": True,
        "canonical_loader_cardinalities_and_histograms_verified": True,
        "heldout_evaluation_matrix_retained": True,
        "avalanche_style_metric_definitions_retained": True,
        "external_transform_parity": False,
        "external_metric_implementation_parity": False,
        "competitive_deep_baseline_parity": False,
        "full_ipmnist_horizon_parity": False,
        "accelerator_timing_qualified": False,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify and run the canonical supervised CL slice")
    parser.add_argument("--blockers", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--benchmark", choices=native.BENCHMARK_IDS)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--seed", type=int, default=native.FROZEN_SEEDS[0])
    parser.add_argument("--train-examples-per-task", type=int, default=8)
    parser.add_argument("--test-examples-per-task", type=int, default=1)
    parser.add_argument("--replay-capacity", type=int, default=16)
    args = parser.parse_args(argv)
    if args.blockers and not args.execute:
        print(json.dumps(qualification_blocker_manifest(), sort_keys=True, separators=(",", ":")))
        return 0
    if not args.execute or args.benchmark is None or args.asset_root is None:
        parser.error("use --blockers or provide --execute, --benchmark, and --asset-root")
    result = run_canonical_asset_qualification(
        args.benchmark,
        args.asset_root,
        seed=args.seed,
        train_examples_per_task=args.train_examples_per_task,
        test_examples_per_task=args.test_examples_per_task,
        replay_capacity=args.replay_capacity,
    )
    print(json.dumps(dataclasses.asdict(result), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ASSET_CONTRACT",
    "BLOCKER_SCHEMA",
    "METRIC_CONTRACT",
    "SCHEMA",
    "AssetReceipt",
    "AssetSpec",
    "AvalancheMatrixMetrics",
    "CanonicalAssetBinding",
    "CanonicalDefinition",
    "CanonicalQualification",
    "CanonicalQualificationError",
    "CanonicalResourceReceipt",
    "avalanche_matrix_metrics",
    "canonical_definition",
    "main",
    "qualification_blocker_manifest",
    "run_canonical_asset_qualification",
    "validate_canonical_asset_qualification",
]
