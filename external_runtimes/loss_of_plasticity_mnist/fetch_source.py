"""Fetch and safely unpack the exact official loss-of-plasticity source archive."""

from __future__ import annotations

import hashlib
import os
import stat
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

SOURCE_COMMIT = "a6b79580d85f3025bdb601566d3627c5f489f13b"
SOURCE_ARCHIVE_SHA256 = "fe8e7973eda2201865b4d2f76e4aa6a1d68959da1dbd4e50fc29f4d3dad8e5b7"
SOURCE_URL = (
    "https://github.com/shibhansh/loss-of-plasticity/archive/"
    f"{SOURCE_COMMIT}.tar.gz"
)
SOURCE_DIRECTORY = Path(f"/opt/loss-of-plasticity-{SOURCE_COMMIT}")
_ARCHIVE_PATH = Path("/tmp/loss-of-plasticity.tar.gz")
_MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
_MAX_EXPANDED_BYTES = 64 * 1024 * 1024
_MAX_MEMBERS = 512


def _download() -> None:
    digest = hashlib.sha256()
    total = 0
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "asi-qualification/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        with _ARCHIVE_PATH.open("xb") as output:
            final_url = response.geturl()
            if not final_url.startswith(
                "https://codeload.github.com/shibhansh/loss-of-plasticity/"
            ):
                raise ValueError(
                    "source archive redirected outside the exact official codeload origin"
                )
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_ARCHIVE_BYTES:
                    raise ValueError("source archive exceeds its byte limit")
                digest.update(chunk)
                output.write(chunk)
    if digest.hexdigest() != SOURCE_ARCHIVE_SHA256:
        raise ValueError("source archive differs from the pinned official bytes")


def _extract() -> None:
    expected_root = f"loss-of-plasticity-{SOURCE_COMMIT}"
    expanded_bytes = 0
    with tarfile.open(_ARCHIVE_PATH, mode="r:gz") as archive:
        members = archive.getmembers()
        if not 1 <= len(members) <= _MAX_MEMBERS:
            raise ValueError("source archive member count differs from its bounded contract")
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or not path.parts
                or path.parts[0] != expected_root
                or ".." in path.parts
                or not (member.isdir() or member.isreg())
            ):
                raise ValueError("source archive contains an inadmissible member")
            if member.size < 0:
                raise ValueError("source archive contains a negative member size")
            expanded_bytes += member.size
            if expanded_bytes > _MAX_EXPANDED_BYTES:
                raise ValueError("source archive exceeds its expanded byte limit")
        archive.extractall("/opt")
    if not SOURCE_DIRECTORY.is_dir():
        raise ValueError("source archive did not create the exact pinned root")


def main() -> int:
    if SOURCE_DIRECTORY.exists() or _ARCHIVE_PATH.exists():
        raise ValueError("source destination must start absent")
    try:
        _download()
        _extract()
    finally:
        if _ARCHIVE_PATH.exists():
            _ARCHIVE_PATH.unlink()
    for root, directories, files in os.walk(SOURCE_DIRECTORY):
        for name in (*directories, *files):
            path = Path(root, name)
            mode = path.stat().st_mode
            executable = bool(mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
            path.chmod(0o555 if path.is_dir() or executable else 0o444)
    SOURCE_DIRECTORY.chmod(0o555)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
