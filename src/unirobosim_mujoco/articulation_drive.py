"""Build-time validation and compilation of per-articulation MuJoCo drives."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, NoReturn

from unirobosim import EntityKind, EntityPath, EntitySpec, ValidationError, WorldSpec

from .config import MuJoCoAdapterConfig

ARTICULATION_DRIVE_PROFILE_KEY = "unirobosim_articulation_drive_profile"
ARTICULATION_DRIVE_PROFILE_SCHEMA = "unirobosim-articulation-drive-profile/1"
_BACKEND = "mujoco"


@dataclass(frozen=True, slots=True)
class CompiledMuJoCoArticulationDrive:
    """Joint-order-aligned controller coefficients compiled once at build time."""

    position_stiffness: tuple[float, ...]
    position_damping: tuple[float, ...]


def _reject(
    message: str,
    *,
    spec: WorldSpec,
    entity: EntitySpec,
    provider_id: str,
    **details: object,
) -> NoReturn:
    raise ValidationError(
        message,
        operation="mujoco.session.build.articulation_drive_profile",
        backend_id=provider_id,
        world_id=spec.world_id,
        entity_path=entity.path.value,
        details={"metadata_key": ARTICULATION_DRIVE_PROFILE_KEY, **details},
    ) from None


def _mapping(
    value: object,
    *,
    field: str,
    spec: WorldSpec,
    entity: EntitySpec,
    provider_id: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        _reject(
            f"articulation drive {field} must be a string-keyed mapping",
            spec=spec,
            entity=entity,
            provider_id=provider_id,
            field=field,
        )
    return value


def _exact_fields(
    value: Mapping[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    field: str,
    spec: WorldSpec,
    entity: EntitySpec,
    provider_id: str,
) -> None:
    actual = frozenset(value)
    missing = tuple(sorted(required - actual))
    unknown = tuple(sorted(actual - required - optional))
    if missing or unknown:
        _reject(
            f"articulation drive {field} fields are invalid",
            spec=spec,
            entity=entity,
            provider_id=provider_id,
            field=field,
            missing_fields=missing,
            unknown_fields=unknown,
        )


def _nonnegative_number(
    value: object,
    *,
    field: str,
    joint_name: str,
    spec: WorldSpec,
    entity: EntitySpec,
    provider_id: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        _reject(
            f"MuJoCo articulation drive {field} must be finite and non-negative",
            spec=spec,
            entity=entity,
            provider_id=provider_id,
            field=field,
            joint_name=joint_name,
        )
    return float(value)


def _validated_joint_settings(
    raw_profile: object,
    *,
    spec: WorldSpec,
    entity: EntitySpec,
    provider_id: str,
) -> dict[str, tuple[float, float]]:
    profile = _mapping(
        raw_profile,
        field="profile",
        spec=spec,
        entity=entity,
        provider_id=provider_id,
    )
    _exact_fields(
        profile,
        required=frozenset({"schema", "backend", "joints"}),
        field="profile",
        spec=spec,
        entity=entity,
        provider_id=provider_id,
    )
    if profile["schema"] != ARTICULATION_DRIVE_PROFILE_SCHEMA:
        _reject(
            "MuJoCo articulation drive profile schema is unsupported",
            spec=spec,
            entity=entity,
            provider_id=provider_id,
            expected_schema=ARTICULATION_DRIVE_PROFILE_SCHEMA,
            actual_schema=profile["schema"],
        )
    if profile["backend"] != _BACKEND:
        _reject(
            "articulation drive profile backend does not match the selected provider",
            spec=spec,
            entity=entity,
            provider_id=provider_id,
            expected_backend=_BACKEND,
            actual_backend=profile["backend"],
        )
    joints = _mapping(
        profile["joints"],
        field="joints",
        spec=spec,
        entity=entity,
        provider_id=provider_id,
    )
    if not joints:
        _reject(
            "articulation drive profile must contain at least one declared joint",
            spec=spec,
            entity=entity,
            provider_id=provider_id,
        )
    if len(joints) > len(entity.joint_names):
        _reject(
            "articulation drive profile exceeds the declared joint count",
            spec=spec,
            entity=entity,
            provider_id=provider_id,
            declared_joint_count=len(entity.joint_names),
            profile_joint_count=len(joints),
        )
    declared = frozenset(entity.joint_names)
    unknown_joints = tuple(sorted(name for name in joints if name not in declared))
    if unknown_joints:
        _reject(
            "articulation drive profile names undeclared joints",
            spec=spec,
            entity=entity,
            provider_id=provider_id,
            unknown_joints=unknown_joints,
        )
    result: dict[str, tuple[float, float]] = {}
    for joint_name, raw_settings in joints.items():
        settings = _mapping(
            raw_settings,
            field=f"joints.{joint_name}",
            spec=spec,
            entity=entity,
            provider_id=provider_id,
        )
        _exact_fields(
            settings,
            required=frozenset({"position_stiffness", "position_damping"}),
            field=f"joints.{joint_name}",
            spec=spec,
            entity=entity,
            provider_id=provider_id,
        )
        result[joint_name] = (
            _nonnegative_number(
                settings["position_stiffness"],
                field="position_stiffness",
                joint_name=joint_name,
                spec=spec,
                entity=entity,
                provider_id=provider_id,
            ),
            _nonnegative_number(
                settings["position_damping"],
                field="position_damping",
                joint_name=joint_name,
                spec=spec,
                entity=entity,
                provider_id=provider_id,
            ),
        )
    return result


def compile_articulation_drive_profiles(
    spec: WorldSpec,
    config: MuJoCoAdapterConfig,
    *,
    provider_id: str,
) -> Mapping[EntityPath, CompiledMuJoCoArticulationDrive] | None:
    """Validate boundary metadata and compile joint-order arrays.

    ``None`` is the exact omission path.  When at least one profile is authored,
    every articulation receives an immutable path-scoped array using adapter
    defaults for joints that were not overridden.
    """

    authored: dict[EntityPath, dict[str, tuple[float, float]]] = {}
    for entity in spec.entities:
        if ARTICULATION_DRIVE_PROFILE_KEY not in entity.metadata:
            continue
        if entity.kind is not EntityKind.ARTICULATION:
            _reject(
                "articulation drive profile is only valid on articulation entities",
                spec=spec,
                entity=entity,
                provider_id=provider_id,
                entity_kind=entity.kind.value,
            )
        authored[entity.path] = _validated_joint_settings(
            entity.metadata[ARTICULATION_DRIVE_PROFILE_KEY],
            spec=spec,
            entity=entity,
            provider_id=provider_id,
        )
    if not authored:
        return None

    compiled: dict[EntityPath, CompiledMuJoCoArticulationDrive] = {}
    for entity in spec.entities:
        if entity.kind is not EntityKind.ARTICULATION:
            continue
        settings = authored.get(entity.path, {})
        stiffness: list[float] = []
        damping: list[float] = []
        for joint_name in entity.joint_names:
            override = settings.get(joint_name)
            if override is None:
                stiffness.append(config.position_stiffness_for(joint_name))
                damping.append(config.position_damping_for(joint_name))
            else:
                stiffness.append(override[0])
                damping.append(override[1])
        compiled[entity.path] = CompiledMuJoCoArticulationDrive(tuple(stiffness), tuple(damping))
    return MappingProxyType(compiled)


__all__ = (
    "ARTICULATION_DRIVE_PROFILE_KEY",
    "ARTICULATION_DRIVE_PROFILE_SCHEMA",
    "CompiledMuJoCoArticulationDrive",
    "compile_articulation_drive_profiles",
)
