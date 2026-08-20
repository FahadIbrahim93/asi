"""Pinned-directory, create-only publication for prospective development reports."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable, Sequence
from pathlib import Path, PosixPath


def open_directory_chain(root: Path, segments: Sequence[str]) -> int:
    """Open/create a directory chain without following its named components."""
    if type(root) is not PosixPath or not root.is_absolute():
        raise ValueError("publication root must be an exact absolute POSIX Path")
    if (type(segments) is not tuple and type(segments) is not list) or any(
        type(segment) is not str
        or not segment
        or segment in {".", ".."}
        or "/" in segment
        or "\x00" in segment
        for segment in segments
    ):
        raise ValueError("publication directory segments are invalid")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    directory = os.open(root, flags)
    try:
        for segment in segments:
            try:
                os.mkdir(segment, mode=0o755, dir_fd=directory)
            except FileExistsError:
                pass
            child = os.open(segment, flags, dir_fd=directory)
            os.close(directory)
            directory = child
        return directory
    except BaseException:
        os.close(directory)
        raise


def publish_prepared_json_at(
    directory: int,
    name: str,
    *,
    prepare: Callable[[], bytes],
    validate_loaded: Callable[[object], None],
    max_bytes: int,
) -> None:
    """Reserve, prepare, link without replacement, reread, and strictly validate JSON."""
    if type(directory) is not int or directory < 0:
        raise ValueError("directory must be one pinned directory descriptor")
    if (
        type(name) is not str
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\x00" in name
        or len(name.encode("utf-8")) > 240
    ):
        raise ValueError("publication name is invalid")
    if type(max_bytes) is not int or not 0 < max_bytes <= 64 * 1024 * 1024:
        raise ValueError("publication byte bound is invalid")
    reserve = f".{name}.reserve"
    temporary = f".{name}.tmp"
    create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    reserve_descriptor = os.open(reserve, create_flags, 0o400, dir_fd=directory)
    os.close(reserve_descriptor)
    temporary_created = False
    published = False
    try:
        encoded = prepare()
        if type(encoded) is not bytes or not 0 < len(encoded) <= max_bytes:
            raise ValueError("prepared report exceeds its exact byte bound")
        descriptor = os.open(temporary, create_flags, 0o400, dir_fd=directory)
        temporary_created = True
        try:
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                if written <= 0:
                    raise OSError("publication write made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(
            temporary,
            name,
            src_dir_fd=directory,
            dst_dir_fd=directory,
            follow_symlinks=False,
        )
        published = True
        read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, read_flags, dir_fd=directory)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size != len(encoded):
                raise RuntimeError("published report is not the prepared regular file")
            loaded = bytearray()
            while len(loaded) <= max_bytes:
                chunk = os.read(descriptor, min(64 * 1024, max_bytes + 1 - len(loaded)))
                if not chunk:
                    break
                loaded.extend(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        if bytes(loaded) != encoded or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise RuntimeError("published report changed during its bounded reread")
        try:
            decoded = json.loads(loaded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("published report is not exact JSON") from exc
        validate_loaded(decoded)
        os.fsync(directory)
    except BaseException:
        if published:
            os.unlink(name, dir_fd=directory)
        raise
    finally:
        if temporary_created:
            os.unlink(temporary, dir_fd=directory)
        os.unlink(reserve, dir_fd=directory)


__all__ = ["open_directory_chain", "publish_prepared_json_at"]
