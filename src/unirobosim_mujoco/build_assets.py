"""Immutable Core ``BuildInput`` snapshots owned by the adapter world."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from unirobosim import (
    ASSET_DEPENDENCY_INCOMPLETE,
    ASSET_IDENTITY_CHANGED,
    BuildInput,
    BuildResourceEntry,
    ValidationError,
    WorldSpec,
)


@dataclass(slots=True)
class BuildAssetLease:
    """Read-only private bundle retained through the native world's lifetime."""

    root: Path
    build_input: BuildInput
    paths: dict[str, Path]
    _closed: bool = False

    def selected_path(self, *, entity_id: str, asset_uri: str) -> Path:
        candidates = []
        for entry in self.build_input.manifest.entries:
            if not entry.selected_simulation_input:
                continue
            if entry.entity_id == entity_id or asset_uri in {entry.requested_uri, entry.resolved_uri}:
                candidates.append(self.paths[entry.resource_id])
        unique = tuple(dict.fromkeys(candidates))
        if len(unique) != 1:
            raise ValidationError(
                "asset entity does not have exactly one selected simulation input",
                operation="mujoco.session.build",
                details={
                    "detail_code": ASSET_DEPENDENCY_INCOMPLETE,
                    "entity_id": entity_id,
                    "selected_inputs": len(unique),
                },
            ) from None
        return unique[0]

    def entry_for_path(self, path: Path) -> BuildResourceEntry:
        for entry in self.build_input.manifest.entries:
            if self.paths[entry.resource_id] == path:
                return entry
        raise KeyError(path)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for directory, _, _ in os.walk(self.root):
            try:
                Path(directory).chmod(0o700)
            except OSError:
                pass
        shutil.rmtree(self.root, ignore_errors=True)


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _copy_source(source: object, destination: Path) -> tuple[str, str]:
    # ``BuildSourceEntry`` is validated by Core before this private carrier can
    # exist.  Directory-relative no-follow opens keep the source walk beneath
    # its declared root and reject symlink substitution.
    descriptors: list[int] = []
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        current = os.open(os.path.sep, directory_flags)
        descriptors.append(current)
        for part in source.source_root.split(os.path.sep):  # type: ignore[attr-defined]
            if part:
                current = os.open(part, directory_flags, dir_fd=current)
                descriptors.append(current)
        parts = source.relative_source_path.split("/")  # type: ignore[attr-defined]
        for part in parts[:-1]:
            current = os.open(part, directory_flags, dir_fd=current)
            descriptors.append(current)
        source_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=current)
        descriptors.append(source_fd)
        before = os.fstat(source_fd)
        expected = source.expected_identity  # type: ignore[attr-defined]
        expected_identity = (
            expected.device,
            expected.inode,
            expected.mode,
            expected.byte_size,
            expected.mtime_ns,
            expected.ctime_ns,
        )
        if not stat.S_ISREG(before.st_mode) or _identity(before) != expected_identity:
            return source.expected_sha256, ""  # type: ignore[attr-defined]
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        byte_size = 0
        with destination.open("xb") as output:
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                byte_size += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        after = os.fstat(source_fd)
        actual_digest = digest.hexdigest()
        stable = _identity(after) == _identity(before)
        if (
            not stable or byte_size != expected.byte_size or actual_digest != source.expected_sha256  # type: ignore[attr-defined]
        ):
            return source.expected_sha256, actual_digest  # type: ignore[attr-defined]
        destination.chmod(0o444)
        return actual_digest, actual_digest
    except OSError:
        return source.expected_sha256, ""  # type: ignore[attr-defined]
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def snapshot_build_input(spec: WorldSpec, build_input: BuildInput | None) -> BuildAssetLease | None:
    manifest_digest = spec.build_resource_manifest_sha256
    if manifest_digest is None:
        if build_input is not None:
            raise ValidationError(
                "asset-free World cannot receive BuildInput",
                operation="mujoco.session.build",
                details={"detail_code": ASSET_DEPENDENCY_INCOMPLETE},
            ) from None
        return None
    if type(build_input) is not BuildInput or build_input.manifest.sha256 != manifest_digest:
        raise ValidationError(
            "World and BuildInput manifest identities do not match",
            operation="mujoco.session.build",
            details={"detail_code": ASSET_DEPENDENCY_INCOMPLETE},
        ) from None

    root = Path(tempfile.mkdtemp(prefix="unirobosim-mujoco-assets-"))
    try:
        entry_by_id = {entry.resource_id: entry for entry in build_input.manifest.entries}
        paths: dict[str, Path] = {}
        for source in build_input.sources:
            entry = entry_by_id[source.resource_id]
            destination = root / entry.relative_bundle_path
            expected, actual = _copy_source(source, destination)
            if expected != actual:
                raise ValidationError(
                    "build source identity changed",
                    operation="mujoco.session.build",
                    details={
                        "detail_code": ASSET_IDENTITY_CHANGED,
                        "resource_id": source.resource_id,
                        "expected_sha256_prefix": expected[:12],
                        "actual_sha256_prefix": actual[:12],
                    },
                ) from None
            paths[source.resource_id] = destination
        for directory, _, _ in os.walk(root, topdown=False):
            Path(directory).chmod(0o555)
        return BuildAssetLease(root, build_input, paths)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
