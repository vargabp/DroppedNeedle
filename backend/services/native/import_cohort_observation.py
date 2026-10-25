"""Unwired Linux read-only byte provider for the native cohort candidate.

Observations are short-lived, not publication, ownership or cleanup permission.
The caller must still hold the native filesystem coordinator and policy guard.
"""
from __future__ import annotations

import hashlib
import math
import os
import stat
import time
from contextlib import ExitStack
from pathlib import PurePosixPath
from typing import Mapping

from services.native.file_revision import revision_from_stat


def _parts(value: str, *, absolute: bool) -> tuple[str, ...]:
    path = PurePosixPath(value)
    if (not value or path.is_absolute() != absolute or path.as_posix() != value
            or value in (".", "/") or ".." in path.parts or "\\" in value
            or value.startswith("//") or any(ord(c) < 32 for c in value)):
        raise ValueError("Noncanonical cohort path")
    return path.parts[1:] if absolute else path.parts


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def observe_cohort_bytes(
    *, root_path: str, members: Mapping[str, tuple[str, str]],
    max_total_bytes: int, timeout_seconds: float,
) -> dict[str, str]:
    """Hash complete ID -> (relative path, native exact stat revision) input.

    Fail the entire observation on drift, nonregular files, symlinks, budget
    exhaustion or unsupported descriptor-relative operations. Never falls back
    to path.resolve/open, which could follow a swapped ancestor. Requires the
    caller to supply the complete cohort, not a scan page. Completeness against
    a canonical recording map is a separate publisher/store responsibility.
    Byte budget bounds cohort size; verification reads each file twice. The
    monotonic deadline is cooperative between syscalls, not a hard timeout for
    blocked kernel/storage I/O. Root identity must come from native configuration.
    This cannot prevent writes after return or replace coordinator locking.
    """
    if (type(max_total_bytes) is not int or max_total_bytes < 0
            or not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0 or not 1 <= len(members) <= 500):
        raise ValueError("Invalid cohort observation bounds")
    if not hasattr(os, "O_NOFOLLOW") or os.open not in os.supports_dir_fd:
        raise ValueError("Descriptor-relative no-follow filesystem required")
    root_parts = _parts(root_path, absolute=True)
    paths = {key: _parts(value[0], absolute=False) for key, value in members.items()}
    if any(not isinstance(key, str) or not key for key in members) or len(set(paths.values())) != len(paths):
        raise ValueError("Unique cohort IDs and paths required")
    deadline = time.monotonic() + timeout_seconds
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC

    def bounded() -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError("Cohort observation deadline exceeded")

    def open_relative(base: int, parts: tuple[str, ...], *, directory: bool) -> int:
        with ExitStack() as stack:
            current = base
            for part in parts[:-1]:
                bounded()
                current = os.open(part, directory_flags, dir_fd=current)
                stack.callback(os.close, current)
            bounded()
            flags = directory_flags if directory else (
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
            return os.open(parts[-1], flags, dir_fd=current)

    with ExitStack() as stack:
        slash = os.open("/", directory_flags)
        stack.callback(os.close, slash)
        root = open_relative(slash, root_parts, directory=True)
        stack.callback(os.close, root)
        root_identity = _identity(os.fstat(root))
        held = {}
        total = 0
        results = {}
        for key, parts in paths.items():
            descriptor = open_relative(root, parts, directory=False)
            stack.callback(os.close, descriptor)
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or revision_from_stat(before) != members[key][1]:
                raise ValueError("Cohort file type or exact stat revision changed")
            total += before.st_size
            if total > max_total_bytes:
                raise ValueError("Cohort byte budget exceeded")
            digest = hashlib.sha256()
            read_bytes = 0
            while True:
                bounded()
                chunk = os.read(descriptor, min(1024 * 1024, before.st_size - read_bytes + 1))
                if not chunk:
                    break
                read_bytes += len(chunk)
                if read_bytes > before.st_size:
                    raise ValueError("Cohort file grew during observation")
                digest.update(chunk)
            if read_bytes != before.st_size or _identity(os.fstat(descriptor)) != _identity(before):
                raise ValueError("Cohort file changed during observation")
            held[key] = (descriptor, _identity(before))
            results[key] = digest.hexdigest()
        # Recheck earlier members after the last hash, including pathname swaps.
        for key, (descriptor, identity) in held.items():
            bounded()
            check = open_relative(root, paths[key], directory=False)
            try:
                if _identity(os.fstat(descriptor)) != identity or _identity(os.fstat(check)) != identity:
                    raise ValueError("Cohort member changed before observation completed")
                # ctime can share a filesystem tick with a same-size rewrite.
                # Re-read bytes rather than treating unchanged stat as proof.
                digest = hashlib.sha256()
                read_bytes = 0
                while True:
                    bounded()
                    chunk = os.read(check, min(1024 * 1024, identity[3] - read_bytes + 1))
                    if not chunk:
                        break
                    read_bytes += len(chunk)
                    if read_bytes > identity[3]:
                        raise ValueError("Cohort file grew during final observation")
                    digest.update(chunk)
                if (read_bytes != identity[3] or digest.hexdigest() != results[key]
                        or _identity(os.fstat(check)) != identity):
                    raise ValueError("Cohort bytes changed before observation completed")
            finally:
                os.close(check)
        check_root = open_relative(slash, root_parts, directory=True)
        try:
            if _identity(os.fstat(check_root)) != root_identity:
                raise ValueError("Cohort root changed during observation")
        finally:
            os.close(check_root)
        bounded()
        return results
