"""Direct MuJoCo implementation of the UniRoboSim world and scene-control contracts."""

from __future__ import annotations

import hashlib
import importlib
import math
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import unquote, urlparse

import numpy as np
from unirobosim import (
    ARTICULATION_AXIS_UNITS_MISMATCH,
    PHYSICAL_WORLD_SCHEMA_VERSION,
    ArrayValue,
    ArticulationCommand,
    ArticulationState,
    BuildFingerprint,
    BuildReport,
    CameraModality,
    CommandError,
    CommandMode,
    ContactState,
    DebugBatch,
    DebugLifetimeMode,
    DebugPrimitive,
    DebugPublishReport,
    DeformableCommand,
    DeformableState,
    EntityHandle,
    EntityKind,
    EntityNotFoundError,
    EntityPath,
    EntitySpec,
    FrozenMap,
    LifecycleError,
    ParticleFluidCommand,
    ParticleFluidState,
    Pose,
    ResetResult,
    RigidBodyCommand,
    RigidBodyState,
    SceneCommand,
    SceneCommandKind,
    SceneCommandResult,
    SceneCommandStatus,
    SceneDelta,
    SceneDragMode,
    SceneEntityState,
    SceneSnapshot,
    SceneVisual,
    SceneVisualKind,
    SensorChannel,
    SensorSample,
    StaleHandleError,
    Tick,
    UnsupportedCapabilityError,
    ValidationError,
    WorldBuildError,
    WorldSpec,
    WorldState,
)

if TYPE_CHECKING:
    from .articulation_drive import CompiledMuJoCoArticulationDrive
    from .build_assets import BuildAssetLease
    from .provider import MuJoCoSession

mujoco: Any = importlib.import_module("mujoco")


@dataclass(frozen=True)
class _NativeEntity:
    spec: EntitySpec
    body_id: int | None
    free_qpos_address: int | None
    free_dof_address: int | None
    joint_ids: tuple[int, ...]
    joint_qpos_addresses: tuple[int, ...]
    joint_dof_addresses: tuple[int, ...]
    camera_name: str | None


def _wxyz(xyzw: tuple[float, float, float, float]) -> str:
    x, y, z, w = xyzw
    return f"{w} {x} {y} {z}"


def _xyz(values: tuple[float, float, float]) -> str:
    return " ".join(str(value) for value in values)


def _uint8_array(shape: tuple[int, ...], data: bytes) -> ArrayValue:
    """Use Core's compact byte storage while retaining Core 0.9 compatibility."""

    factory = getattr(ArrayValue, "from_uint8_bytes", None)
    if callable(factory):
        return cast(ArrayValue, factory(shape, data))
    return ArrayValue(shape, tuple(data), dtype="uint8")


class MuJoCoWorld:
    def __init__(
        self,
        session: MuJoCoSession,
        spec: WorldSpec,
        generation: int,
        asset_lease: BuildAssetLease | None = None,
        *,
        native_substeps_per_logical_step: int,
        articulation_drive_profiles: Mapping[EntityPath, CompiledMuJoCoArticulationDrive] | None = None,
    ) -> None:
        self._validate_spec(spec, session.descriptor.provider_id)
        if not session.config.headless and spec.environments.count != 1:
            raise ValidationError(
                "MuJoCo GUI mode requires exactly one environment",
                operation="mujoco.build.preflight",
                backend_id=session.descriptor.provider_id,
                world_id=spec.world_id,
                details={"environment_count": spec.environments.count},
            )
        self._session = session
        self._spec = spec
        self._generation = generation
        self._state = WorldState.READY
        self._step_index = 0
        self._reset_count = 0
        self._scene_sequence = 0
        self._entities = {entity.path: entity for entity in spec.entities}
        self._asset_lease = asset_lease
        self._native_substeps_per_logical_step = native_substeps_per_logical_step
        self._native_time_step_seconds = spec.physics.time_step_seconds / self._native_substeps_per_logical_step
        self._articulation_drive_profiles = articulation_drive_profiles
        self._apply_controls_for_step: Callable[[int], None] = (
            self._apply_controls if articulation_drive_profiles is None else self._apply_profiled_controls
        )
        self._models, names = self._build_models(
            spec,
            asset_lease,
            native_time_step_seconds=self._native_time_step_seconds,
        )
        self._data = [mujoco.MjData(model) for model in self._models]
        self._native = self._resolve_native(self._models[0], spec, names)
        self._commands: list[dict[EntityPath, list[tuple[CommandMode, float]]]] = [
            {
                entity.path: [(CommandMode.POSITION, value) for value in entity.initial_joint_positions]
                for entity in spec.entities
                if entity.kind is EntityKind.ARTICULATION
            }
            for _ in range(spec.environments.count)
        ]
        self._rigid_wrenches: list[dict[EntityPath, tuple[tuple[float, ...], tuple[float, ...]]]] = [
            {} for _ in range(spec.environments.count)
        ]
        self._debug: dict[tuple[str, str, str], DebugPrimitive] = {}
        self._debug_expiration: dict[tuple[str, str, str], int | None] = {}
        self._scene_results: dict[str, SceneCommandResult] = {}
        self._drags: dict[str, tuple[EntityPath, int, Pose]] = {}
        self._renderers: dict[tuple[int, int, int], Any] = {}
        self._viewers: list[Any] = []
        for model, data in zip(self._models, self._data, strict=True):
            self._write_initial_articulation_positions(data)
            mujoco.mj_forward(model, data)
        if not session.config.headless:
            viewer_module = importlib.import_module("mujoco.viewer")
            try:
                for model, data in zip(self._models, self._data, strict=True):
                    viewer = viewer_module.launch_passive(
                        model,
                        data,
                        show_left_ui=False,
                        show_right_ui=False,
                    )
                    self._viewers.append(viewer)
                    viewer.sync()
            except Exception:
                for viewer in self._viewers:
                    viewer.close()
                self._viewers.clear()
                raise
        self._build_report = BuildReport(
            BuildFingerprint(
                session.descriptor.provider_id,
                session.descriptor.version,
                session.descriptor.contract_version,
                spec.digest,
                session.descriptor.capabilities.digest,
            ),
            spec.world_id,
            generation,
            spec.environments.count,
            len(spec.entities),
        )

    def _write_initial_articulation_positions(self, data: Any) -> None:
        """Set runtime qpos without changing MuJoCo's asset reference configuration."""

        for entity in self._spec.entities:
            if entity.kind is not EntityKind.ARTICULATION:
                continue
            native = self._native[entity.path]
            for address, value in zip(
                native.joint_qpos_addresses,
                entity.initial_joint_positions,
                strict=True,
            ):
                data.qpos[address] = value

    @staticmethod
    def _validate_spec(spec: WorldSpec, backend_id: str) -> None:
        supported = {EntityKind.RIGID_BODY, EntityKind.ARTICULATION, EntityKind.CAMERA_SENSOR}
        for entity in spec.entities:
            if entity.kind not in supported:
                raise UnsupportedCapabilityError(
                    "MuJoCo adapter does not claim this entity-kind contract",
                    operation="mujoco.build.preflight",
                    backend_id=backend_id,
                    world_id=spec.world_id,
                    entity_path=entity.path.value,
                    details={"entity_kind": entity.kind.value},
                )
        asset_entities = tuple(entity for entity in spec.entities if entity.asset_uri is not None)
        if len(asset_entities) > 1:
            raise WorldBuildError(
                "the current MuJoCo asset composition profile accepts one native asset entity per world",
                operation="mujoco.build.preflight",
                backend_id=backend_id,
                world_id=spec.world_id,
                details={"asset_entities": [entity.path.value for entity in asset_entities]},
            )

    @staticmethod
    def _local_asset_path(
        asset_uri: str,
        entity: EntitySpec,
        asset_lease: BuildAssetLease | None,
    ) -> Path:
        if asset_lease is not None:
            return asset_lease.selected_path(entity_id=entity.path.value, asset_uri=asset_uri)
        parsed = urlparse(asset_uri)
        if parsed.scheme not in {"", "file"}:
            raise WorldBuildError(
                "MuJoCo requires a local native asset",
                operation="mujoco.build.asset",
                entity_path=entity.path.value,
                details={"asset_uri": asset_uri, "scheme": parsed.scheme},
            )
        path = Path(unquote(parsed.path) if parsed.scheme == "file" else asset_uri).expanduser().resolve()
        if not path.is_file():
            raise WorldBuildError(
                "MuJoCo native asset does not exist",
                operation="mujoco.build.asset",
                entity_path=entity.path.value,
                details={"asset_uri": asset_uri, "resolved_path": str(path)},
            )
        return path

    @classmethod
    def _build_models(
        cls,
        spec: WorldSpec,
        asset_lease: BuildAssetLease | None,
        *,
        native_time_step_seconds: float,
    ) -> tuple[list[Any], dict[EntityPath, dict[str, object]]]:
        asset = next((entity for entity in spec.entities if entity.asset_uri is not None), None)
        if asset is None:
            xml, procedural_names = cls._build_xml(
                spec,
                native_time_step_seconds=native_time_step_seconds,
            )
            return (
                [mujoco.MjModel.from_xml_string(xml) for _ in range(spec.environments.count)],
                procedural_names,
            )
        assert asset.kind in {EntityKind.RIGID_BODY, EntityKind.ARTICULATION} and asset.asset_uri is not None
        path = cls._local_asset_path(asset.asset_uri, asset, asset_lease)
        native_spec = mujoco.MjSpec.from_file(str(path))
        # Vendor URDFs frequently contain rounded inertias that violate the strict
        # triangle inequality. MuJoCo's documented compiler repair preserves the
        # source file and produces a positive, balanced inertia tensor.
        native_spec.compiler.balanceinertia = True
        native_spec.option.timestep = native_time_step_seconds
        native_spec.option.gravity = spec.physics.gravity_m_s2
        cameras = tuple(
            entity.camera for entity in spec.entities if entity.kind is EntityKind.CAMERA_SENSOR and entity.camera
        )
        if cameras:
            native_spec.visual.global_.offwidth = max(
                native_spec.visual.global_.offwidth, *(camera.width_px for camera in cameras)
            )
            native_spec.visual.global_.offheight = max(
                native_spec.visual.global_.offheight, *(camera.height_px for camera in cameras)
            )
        for joint in native_spec.joints:
            if int(joint.type) in {
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            }:
                joint.armature = max(float(joint.armature), 0.02)
                joint.damping = (max(float(joint.damping[0]), 0.5), 0.0, 0.0)
        root_body = native_spec.worldbody.first_body()
        if root_body is None or native_spec.worldbody.next_body(root_body) is not None:
            raise WorldBuildError(
                "MuJoCo native assets must contain exactly one top-level body",
                operation="mujoco.build.asset",
                entity_path=asset.path.value,
                details={"asset_uri": asset.asset_uri},
            )
        root_body.pos = asset.pose.position
        x, y, z, w = asset.pose.orientation_xyzw
        root_body.quat = (w, x, y, z)
        if asset.kind is EntityKind.ARTICULATION:
            # PyBullet's default URDF profile and the current Isaac asset both disable
            # internal robot collisions. Mirror that policy while retaining collisions
            # between the articulation and external scene bodies.
            asset_body_names = tuple(body.name for body in native_spec.bodies if body.name)
            existing_excludes = {frozenset((exclude.bodyname1, exclude.bodyname2)) for exclude in native_spec.excludes}
            for first, second in combinations(asset_body_names, 2):
                if frozenset((first, second)) not in existing_excludes:
                    native_spec.add_exclude(bodyname1=first, bodyname2=second)
            asset_record: dict[str, object] = {"body": root_body.name, "joints": asset.joint_names}
        else:
            free_joint = next(
                (joint for joint in native_spec.joints if int(joint.type) == int(mujoco.mjtJoint.mjJNT_FREE)),
                None,
            )
            if free_joint is None or not free_joint.name:
                raise WorldBuildError(
                    "MuJoCo rigid assets must contain one named free joint",
                    operation="mujoco.build.rigid",
                    entity_path=asset.path.value,
                    details={"asset_uri": asset.asset_uri},
                )
            asset_record = {"body": root_body.name, "free": free_joint.name}
        names: dict[EntityPath, dict[str, object]] = {asset.path: asset_record}
        native_spec.worldbody.add_geom(
            name="__unirobosim_ground",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=(10.0, 10.0, 0.1),
            rgba=(0.2, 0.23, 0.28, 1.0),
            contype=1,
            conaffinity=1,
        )
        for index, entity in enumerate(spec.entities):
            if entity is asset:
                continue
            prefix = f"entity_{index}"
            record: dict[str, object] = {}
            if entity.kind is EntityKind.RIGID_BODY:
                dimensions = (0.5, 0.5, 0.5) if entity.box is None else entity.box.dimensions_m
                body_name = f"{prefix}_body"
                body = native_spec.worldbody.add_body(
                    name=body_name,
                    pos=entity.pose.position,
                    quat=(
                        entity.pose.orientation_xyzw[3],
                        entity.pose.orientation_xyzw[0],
                        entity.pose.orientation_xyzw[1],
                        entity.pose.orientation_xyzw[2],
                    ),
                )
                free_name = f"{prefix}_free"
                body.add_freejoint(name=free_name)
                body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=tuple(value / 2.0 for value in dimensions),
                    mass=1.0 if entity.box is None else entity.box.mass_kg,
                    rgba=(0.15, 0.7, 0.95, 1.0) if entity.box is None else entity.box.color_rgba,
                    # MuJoCo exposes one sliding Coulomb coefficient rather than
                    # separate static/dynamic values. Preserve the safer static
                    # bound and report this deterministic approximation.
                    friction=(1.0 if entity.box is None else entity.box.static_friction, 0.005, 0.0001),
                )
                record.update(body=body_name, free=free_name)
            elif entity.kind is EntityKind.ARTICULATION:
                body_name = f"{prefix}_base"
                parent = native_spec.worldbody.add_body(name=body_name, pos=entity.pose.position)
                parent.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=(0.2, 0.2, 0.12),
                    mass=1.0,
                    rgba=(0.8, 0.45, 0.15, 1.0),
                )
                joint_names: list[str] = []
                for joint_index, _ in enumerate(entity.joint_names):
                    child = parent.add_body(name=f"{prefix}_link_{joint_index}", pos=(0.0, 0.0, 0.3))
                    joint_name = f"{prefix}_joint_{joint_index}"
                    child.add_joint(
                        name=joint_name,
                        type=mujoco.mjtJoint.mjJNT_HINGE,
                        axis=(0.0, 1.0, 0.0),
                        damping=1.0,
                        armature=0.05,
                    )
                    child.add_geom(
                        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                        fromto=(0.0, 0.0, 0.0, 0.0, 0.0, 0.3),
                        size=(0.07, 0.0, 0.0),
                        mass=0.3,
                    )
                    joint_names.append(joint_name)
                    parent = child
                record.update(body=body_name, joints=tuple(joint_names))
            else:
                camera_name = f"{prefix}_camera"
                orientation = entity.pose.orientation_xyzw
                assert entity.camera is not None
                aspect = entity.camera.width_px / entity.camera.height_px
                vertical_fov = math.degrees(
                    2.0 * math.atan(math.tan(math.radians(entity.camera.horizontal_fov_degrees) / 2.0) / aspect)
                )
                native_spec.worldbody.add_camera(
                    name=camera_name,
                    pos=entity.pose.position,
                    quat=(orientation[3], orientation[0], orientation[1], orientation[2]),
                    fovy=vertical_fov,
                )
                record["camera"] = camera_name
            names[entity.path] = record
        models = [native_spec.copy().compile() for _ in range(spec.environments.count)]
        return models, names

    @staticmethod
    def _build_xml(
        spec: WorldSpec,
        *,
        native_time_step_seconds: float,
    ) -> tuple[str, dict[EntityPath, dict[str, object]]]:
        root = ET.Element("mujoco", model=spec.world_id)
        ET.SubElement(
            root,
            "option",
            timestep=str(native_time_step_seconds),
            gravity=_xyz(spec.physics.gravity_m_s2),
        )
        worldbody = ET.SubElement(root, "worldbody")
        cameras = tuple(
            entity.camera for entity in spec.entities if entity.kind is EntityKind.CAMERA_SENSOR and entity.camera
        )
        if cameras:
            visual = ET.SubElement(root, "visual")
            ET.SubElement(
                visual,
                "global",
                offwidth=str(max(camera.width_px for camera in cameras)),
                offheight=str(max(camera.height_px for camera in cameras)),
            )
        ET.SubElement(worldbody, "geom", name="__ground", type="plane", size="10 10 0.1", rgba="0.2 0.23 0.28 1")
        names: dict[EntityPath, dict[str, object]] = {}
        for index, entity in enumerate(spec.entities):
            prefix = f"entity_{index}"
            record: dict[str, object] = {}
            if entity.kind is EntityKind.RIGID_BODY:
                dimensions = (0.5, 0.5, 0.5) if entity.box is None else entity.box.dimensions_m
                half_extents = (dimensions[0] / 2.0, dimensions[1] / 2.0, dimensions[2] / 2.0)
                mass_kg = 1.0 if entity.box is None else entity.box.mass_kg
                body_name = f"{prefix}_body"
                body = ET.SubElement(
                    worldbody,
                    "body",
                    name=body_name,
                    pos=_xyz(entity.pose.position),
                    quat=_wxyz(entity.pose.orientation_xyzw),
                )
                free_name = f"{prefix}_free"
                ET.SubElement(body, "freejoint", name=free_name)
                box_geometry = entity.box
                ET.SubElement(
                    body,
                    "geom",
                    type="box",
                    size=_xyz(half_extents),
                    mass=str(mass_kg),
                    rgba=" ".join(
                        str(value)
                        for value in ((0.15, 0.7, 0.95, 1.0) if entity.box is None else entity.box.color_rgba)
                    ),
                    friction=(
                        "1 0.005 0.0001" if box_geometry is None else f"{box_geometry.static_friction} 0.005 0.0001"
                    ),
                )
                record.update(body=body_name, free=free_name)
            elif entity.kind is EntityKind.ARTICULATION:
                body_name = f"{prefix}_base"
                parent = ET.SubElement(
                    worldbody,
                    "body",
                    name=body_name,
                    pos=_xyz(entity.pose.position),
                    quat=_wxyz(entity.pose.orientation_xyzw),
                )
                ET.SubElement(parent, "geom", type="box", size="0.2 0.2 0.12", mass="1", rgba="0.8 0.45 0.15 1")
                joint_names: list[str] = []
                for joint_index, _ in enumerate(entity.joint_names):
                    child = ET.SubElement(parent, "body", name=f"{prefix}_link_{joint_index}", pos="0 0 0.3")
                    native_joint = f"{prefix}_joint_{joint_index}"
                    ET.SubElement(
                        child,
                        "joint",
                        name=native_joint,
                        type="hinge",
                        axis="0 1 0",
                        damping="1.0",
                        armature="0.05",
                    )
                    ET.SubElement(child, "geom", type="capsule", fromto="0 0 0 0 0 0.3", size="0.07", mass="0.3")
                    joint_names.append(native_joint)
                    parent = child
                record.update(body=body_name, joints=tuple(joint_names))
            else:
                camera_name = f"{prefix}_camera"
                assert entity.camera is not None
                aspect = entity.camera.width_px / entity.camera.height_px
                vertical_fov = math.degrees(
                    2.0 * math.atan(math.tan(math.radians(entity.camera.horizontal_fov_degrees) / 2.0) / aspect)
                )
                ET.SubElement(
                    worldbody,
                    "camera",
                    name=camera_name,
                    pos=_xyz(entity.pose.position),
                    quat=_wxyz(entity.pose.orientation_xyzw),
                    fovy=str(vertical_fov),
                )
                record["camera"] = camera_name
            names[entity.path] = record
        return ET.tostring(root, encoding="unicode"), names

    @staticmethod
    def _resolve_native(
        model: Any,
        spec: WorldSpec,
        names: dict[EntityPath, dict[str, object]],
    ) -> dict[EntityPath, _NativeEntity]:
        result: dict[EntityPath, _NativeEntity] = {}
        for entity in spec.entities:
            record = names[entity.path]
            body_name = record.get("body")
            body_id = None if body_name is None else int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))
            free_name = record.get("free")
            if free_name is None:
                free_qpos = free_dof = None
            else:
                free_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, free_name))
                free_qpos = int(model.jnt_qposadr[free_id])
                free_dof = int(model.jnt_dofadr[free_id])
            joint_names = tuple(cast(tuple[str, ...], record.get("joints", ())))
            joint_ids = tuple(int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)) for name in joint_names)
            missing = tuple(name for name, joint_id in zip(joint_names, joint_ids, strict=True) if joint_id < 0)
            invalid = tuple(
                name
                for name, joint_id in zip(joint_names, joint_ids, strict=True)
                if joint_id >= 0
                and int(model.jnt_type[joint_id])
                not in {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
            )
            if missing or invalid:
                raise WorldBuildError(
                    "declared articulation joints do not map to one-DOF MuJoCo joints",
                    operation="mujoco.build.articulation",
                    entity_path=entity.path.value,
                    details={"missing": missing, "invalid": invalid, "asset_uri": entity.asset_uri},
                )
            expected_units = tuple(
                "m" if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_SLIDE) else "rad"
                for joint_id in joint_ids
            )
            if entity.kind is EntityKind.ARTICULATION and entity.joint_position_units != expected_units:
                raise WorldBuildError(
                    "declared articulation units do not match native MuJoCo joint types",
                    operation="mujoco.build.articulation",
                    entity_path=entity.path.value,
                    details={
                        "detail_code": ARTICULATION_AXIS_UNITS_MISMATCH,
                        "expected_units": expected_units,
                        "actual_units": entity.joint_position_units,
                    },
                ) from None
            result[entity.path] = _NativeEntity(
                entity,
                body_id,
                free_qpos,
                free_dof,
                joint_ids,
                tuple(int(model.jnt_qposadr[joint]) for joint in joint_ids),
                tuple(int(model.jnt_dofadr[joint]) for joint in joint_ids),
                cast(str, record["camera"]) if isinstance(record.get("camera"), str) else None,
            )
        return result

    @property
    def world_id(self) -> str:
        return self._spec.world_id

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def state(self) -> WorldState:
        return self._state

    @property
    def tick(self) -> Tick:
        return Tick(self._step_index, self._step_index * self._spec.physics.time_step_seconds)

    @property
    def native_time_step_seconds(self) -> float:
        """Effective MuJoCo integration step beneath one logical World step."""

        return self._native_time_step_seconds

    @property
    def logical_time_step_seconds(self) -> float:
        """World time committed by one public step."""

        return self._spec.physics.time_step_seconds

    @property
    def native_substeps_per_logical_step(self) -> int:
        """Number of MuJoCo integrations committed by one logical World step."""

        return self._native_substeps_per_logical_step

    @property
    def build_report(self) -> BuildReport:
        return self._build_report

    def _ensure(self, operation: str) -> None:
        if self._state is not WorldState.READY:
            raise LifecycleError("world is closed", operation=operation, world_id=self.world_id)

    @staticmethod
    def _indices(values: Iterable[int] | None, size: int, name: str, operation: str) -> tuple[int, ...]:
        result = tuple(range(size)) if values is None else tuple(values)
        if (
            not result
            or len(result) != len(set(result))
            or any(not isinstance(item, int) or isinstance(item, bool) or not 0 <= item < size for item in result)
        ):
            raise ValidationError(f"{name} selection is invalid", operation=operation)
        return result

    def _token(self, path: EntityPath) -> str:
        value = f"{self._session.session_id}|{self.world_id}|{self.generation}|{path.value}"
        return hashlib.sha256(value.encode()).hexdigest()

    def resolve(self, path: EntityPath) -> EntityHandle:
        self._ensure("mujoco.world.resolve")
        if not isinstance(path, EntityPath):
            raise ValidationError("resolve requires EntityPath", operation="mujoco.world.resolve")
        entity = self._entities.get(path)
        if entity is None:
            raise EntityNotFoundError("entity does not exist", operation="mujoco.world.resolve", entity_path=path.value)
        return EntityHandle(
            self._session.descriptor.provider_id,
            self._session.session_id,
            self.world_id,
            self.generation,
            path,
            entity.kind,
            self._token(path),
        )

    def _entity(self, handle: EntityHandle, operation: str) -> EntitySpec:
        if not isinstance(handle, EntityHandle):
            raise StaleHandleError("operation requires EntityHandle", operation=operation)
        entity = self._entities.get(handle.path)
        if (
            entity is None
            or handle.provider_id != self._session.descriptor.provider_id
            or handle.session_id != self._session.session_id
            or handle.world_id != self.world_id
            or handle.generation != self.generation
            or handle.token != self._token(handle.path)
            or handle.entity_kind is not entity.kind
        ):
            raise StaleHandleError("handle is stale or foreign", operation=operation)
        return entity

    def reset(self, environment_indices: Iterable[int] | None = None) -> ResetResult:
        self._ensure("mujoco.world.reset")
        environments = self._indices(
            environment_indices,
            self._spec.environments.count,
            "environment_indices",
            "mujoco.world.reset",
        )
        for environment in environments:
            mujoco.mj_resetData(self._models[environment], self._data[environment])
            self._write_initial_articulation_positions(self._data[environment])
            mujoco.mj_forward(self._models[environment], self._data[environment])
            self._rigid_wrenches[environment].clear()
            for entity in self._spec.entities:
                if entity.kind is EntityKind.ARTICULATION:
                    self._commands[environment][entity.path] = [
                        (CommandMode.POSITION, value) for value in entity.initial_joint_positions
                    ]
        self._drags = {key: value for key, value in self._drags.items() if value[1] not in environments}
        self._reset_count += 1
        self._scene_sequence += 1
        return ResetResult(environments, self._reset_count, self.tick)

    def apply_articulation_command(self, command: ArticulationCommand) -> None:
        operation = "mujoco.world.apply_articulation_command"
        self._ensure(operation)
        if not isinstance(command, ArticulationCommand):
            raise CommandError("operation requires ArticulationCommand", operation=operation)
        entity = self._entity(command.handle, operation)
        if entity.kind is not EntityKind.ARTICULATION:
            raise CommandError("entity is not an articulation", operation=operation)
        environments = self._indices(
            command.environment_indices, self._spec.environments.count, "environment", operation
        )
        degrees = self._indices(command.degree_of_freedom_indices, len(entity.joint_names), "degree", operation)
        if command.targets.shape != (len(environments), len(degrees)):
            raise CommandError("articulation target shape is invalid", operation=operation)
        position_units = tuple(entity.joint_position_units[degree] for degree in degrees)
        expected_units = {
            CommandMode.POSITION: position_units,
            CommandMode.VELOCITY: tuple("rad/s" if unit == "rad" else "m/s" for unit in position_units),
            CommandMode.EFFORT: tuple("N*m" if unit == "rad" else "N" for unit in position_units),
        }[command.mode]
        actual_units = command.target_units
        if not actual_units and self._spec.schema_version != PHYSICAL_WORLD_SCHEMA_VERSION:
            actual_units = expected_units
        if actual_units != expected_units:
            raise CommandError(
                "command target units do not match selected articulation axes",
                operation=operation,
                backend_id=self._session.descriptor.provider_id,
                world_id=self.world_id,
                entity_path=entity.path.value,
                details={
                    "detail_code": ARTICULATION_AXIS_UNITS_MISMATCH,
                    "expected_units": expected_units,
                    "actual_units": actual_units,
                },
            ) from None
        rows = command.targets.rows()
        updates = tuple(
            (environment, degree, command.mode, float(rows[row][column]))
            for row, environment in enumerate(environments)
            for column, degree in enumerate(degrees)
        )
        for environment, degree, mode, target in updates:
            self._commands[environment][entity.path][degree] = (
                mode,
                target,
            )

    def read_articulation(self, handle: EntityHandle) -> ArticulationState:
        operation = "mujoco.world.read_articulation"
        self._ensure(operation)
        entity = self._entity(handle, operation)
        if entity.kind is not EntityKind.ARTICULATION:
            raise CommandError("entity is not an articulation", operation=operation)
        native = self._native[entity.path]
        positions = tuple(
            tuple(float(self._data[environment].qpos[address]) for address in native.joint_qpos_addresses)
            for environment in range(self._spec.environments.count)
        )
        velocities = tuple(
            tuple(float(self._data[environment].qvel[address]) for address in native.joint_dof_addresses)
            for environment in range(self._spec.environments.count)
        )
        return ArticulationState(
            entity_id=entity.path.value,
            generation=self.generation,
            tick=self.tick,
            joint_names=entity.joint_names,
            joint_positions=ArrayValue.from_rows(positions),
            joint_velocities=ArrayValue.from_rows(velocities),
            joint_position_units=entity.joint_position_units,
            joint_velocity_units=tuple("rad/s" if unit == "rad" else "m/s" for unit in entity.joint_position_units),
        )

    def apply_rigid_body_command(self, command: RigidBodyCommand) -> None:
        operation = "mujoco.world.apply_rigid_body_command"
        self._ensure(operation)
        if not isinstance(command, RigidBodyCommand):
            raise CommandError("operation requires RigidBodyCommand", operation=operation)
        entity = self._entity(command.handle, operation)
        if entity.kind is not EntityKind.RIGID_BODY:
            raise CommandError("entity is not rigid", operation=operation)
        environments = self._indices(
            command.environment_indices, self._spec.environments.count, "environment", operation
        )
        if command.forces_n.shape != (len(environments), 3) or command.torques_n_m.shape != (len(environments), 3):
            raise CommandError("wrench shape is invalid", operation=operation)
        for row, environment in enumerate(environments):
            self._rigid_wrenches[environment][entity.path] = (
                tuple(float(value) for value in command.forces_n.rows()[row]),
                tuple(float(value) for value in command.torques_n_m.rows()[row]),
            )

    def _rigid_pose(self, path: EntityPath, environment: int) -> Pose:
        native = self._native[path]
        assert native.free_qpos_address is not None
        qpos = self._data[environment].qpos
        address = native.free_qpos_address
        return Pose(
            (float(qpos[address]), float(qpos[address + 1]), float(qpos[address + 2])),
            (
                float(qpos[address + 4]),
                float(qpos[address + 5]),
                float(qpos[address + 6]),
                float(qpos[address + 3]),
            ),
        )

    def read_rigid_body(self, handle: EntityHandle) -> RigidBodyState:
        operation = "mujoco.world.read_rigid_body"
        self._ensure(operation)
        entity = self._entity(handle, operation)
        if entity.kind is not EntityKind.RIGID_BODY:
            raise CommandError("entity is not rigid", operation=operation)
        native = self._native[entity.path]
        assert native.free_dof_address is not None
        poses = tuple(
            self._rigid_pose(entity.path, environment) for environment in range(self._spec.environments.count)
        )
        linear = []
        angular = []
        for environment in range(self._spec.environments.count):
            qvel = self._data[environment].qvel
            address = native.free_dof_address
            linear.append(tuple(float(qvel[address + axis]) for axis in range(3)))
            angular.append(tuple(float(qvel[address + 3 + axis]) for axis in range(3)))
        return RigidBodyState(
            ArrayValue.from_rows(pose.position for pose in poses),
            ArrayValue.from_rows(pose.orientation_xyzw for pose in poses),
            ArrayValue.from_rows(linear),
            ArrayValue.from_rows(angular),
            self.tick,
        )

    def read_contact(self, handle: EntityHandle, force_threshold_n: float = 1.0e-6) -> ContactState:
        operation = "mujoco.world.read_contact"
        self._ensure(operation)
        entity = self._entity(handle, operation)
        if entity.kind is not EntityKind.RIGID_BODY:
            raise CommandError("entity is not rigid", operation=operation)
        if (
            isinstance(force_threshold_n, bool)
            or not isinstance(force_threshold_n, (int, float))
            or force_threshold_n < 0
        ):
            raise ValidationError("force threshold is invalid", operation=operation)
        native = self._native[entity.path]
        assert native.body_id is not None
        forces: list[tuple[float, float, float]] = []
        flags: list[bool] = []
        for model, data in zip(self._models, self._data, strict=True):
            total = [0.0, 0.0, 0.0]
            for contact_index in range(int(data.ncon)):
                contact = data.contact[contact_index]
                body1 = int(model.geom_bodyid[contact.geom1])
                body2 = int(model.geom_bodyid[contact.geom2])
                if native.body_id not in {body1, body2}:
                    continue
                local_force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(model, data, contact_index, local_force)
                sign = 1.0 if body2 == native.body_id else -1.0
                for axis in range(3):
                    total[axis] += sign * float(local_force[0]) * float(contact.frame[axis])
            forces.append((total[0], total[1], total[2]))
            flags.append(math.sqrt(sum(value * value for value in total)) > float(force_threshold_n))
        return ContactState(
            ArrayValue.from_rows(forces),
            ArrayValue((len(flags),), tuple(flags), dtype="bool"),
            self.tick,
        )

    def apply_deformable_command(self, command: DeformableCommand) -> None:
        raise UnsupportedCapabilityError("deformable control is not declared", operation="mujoco.deformable")

    def read_deformable(self, handle: EntityHandle) -> DeformableState:
        raise UnsupportedCapabilityError("deformable state is not declared", operation="mujoco.deformable")

    def apply_particle_fluid_command(self, command: ParticleFluidCommand) -> None:
        raise UnsupportedCapabilityError("particle fluid is not declared", operation="mujoco.fluid")

    def read_particle_fluid(self, handle: EntityHandle) -> ParticleFluidState:
        raise UnsupportedCapabilityError("particle fluid is not declared", operation="mujoco.fluid")

    def read_sensor(self, handle: EntityHandle) -> SensorSample:
        operation = "mujoco.world.read_sensor"
        self._ensure(operation)
        entity = self._entity(handle, operation)
        if entity.kind is not EntityKind.CAMERA_SENSOR or entity.camera is None:
            raise CommandError("entity is not a camera", operation=operation)
        native = self._native[entity.path]
        assert native.camera_name is not None
        channels: list[SensorChannel] = []
        for modality in entity.camera.modalities:
            rgb_frames: list[bytes] = []
            depth_values: list[float | int] = []
            for environment, (model, data) in enumerate(zip(self._models, self._data, strict=True)):
                key = (environment, entity.camera.width_px, entity.camera.height_px)
                renderer = self._renderers.get(key)
                if renderer is None:
                    renderer = mujoco.Renderer(model, height=entity.camera.height_px, width=entity.camera.width_px)
                    self._renderers[key] = renderer
                if modality is CameraModality.DEPTH:
                    renderer.enable_depth_rendering()
                else:
                    renderer.disable_depth_rendering()
                renderer.update_scene(data, camera=native.camera_name)
                image = renderer.render()
                if modality is CameraModality.RGB:
                    expected_native_shape = (entity.camera.height_px, entity.camera.width_px, 3)
                    if image.shape != expected_native_shape or image.dtype != np.dtype("uint8"):
                        raise CommandError(
                            "native RGB frame has an unexpected shape or dtype",
                            operation=operation,
                        )
                    rgb_frames.append(np.ascontiguousarray(image).tobytes(order="C"))
                else:
                    depth_values.extend(image.reshape(-1).tolist())
            shape = (
                (self._spec.environments.count, entity.camera.height_px, entity.camera.width_px, 3)
                if modality is CameraModality.RGB
                else (self._spec.environments.count, entity.camera.height_px, entity.camera.width_px)
            )
            value = (
                _uint8_array(shape, b"".join(rgb_frames))
                if modality is CameraModality.RGB
                else ArrayValue(shape, tuple(depth_values), dtype="float32")
            )
            channels.append(
                SensorChannel(
                    modality,
                    value,
                )
            )
        return SensorSample(handle, tuple(channels), self.tick)

    def publish_debug(self, batch: DebugBatch) -> DebugPublishReport:
        self._ensure("mujoco.world.publish_debug")
        if not isinstance(batch, DebugBatch):
            raise ValidationError("publish requires DebugBatch", operation="mujoco.world.publish_debug")
        for primitive in batch.primitives:
            if any(environment >= self._spec.environments.count for environment in primitive.environment_indices):
                raise ValidationError("debug environment is out of range", operation="mujoco.world.publish_debug")
            self._debug[primitive.key] = primitive
            if primitive.lifetime.mode is DebugLifetimeMode.FRAME:
                expiration = self._step_index + 1
            elif primitive.lifetime.mode is DebugLifetimeMode.STEPS:
                assert primitive.lifetime.step_count is not None
                expiration = self._step_index + primitive.lifetime.step_count
            else:
                expiration = None
            self._debug_expiration[primitive.key] = expiration
        return DebugPublishReport(len(batch.primitives), 0, len(self._debug))

    def clear_debug(
        self,
        *,
        layer: str | None = None,
        group: str | None = None,
        primitive_id: str | None = None,
    ) -> int:
        self._ensure("mujoco.world.clear_debug")
        keys = tuple(
            key
            for key in self._debug
            if (layer is None or key[0] == layer)
            and (group is None or key[1] == group)
            and (primitive_id is None or key[2] == primitive_id)
        )
        for key in keys:
            del self._debug[key]
            del self._debug_expiration[key]
        return len(keys)

    def _apply_controls(self, environment: int) -> None:
        data = self._data[environment]
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0
        for path, commands in self._commands[environment].items():
            native = self._native[path]
            for index, (mode, target) in enumerate(commands):
                qpos_address = native.joint_qpos_addresses[index]
                dof_address = native.joint_dof_addresses[index]
                if mode is CommandMode.POSITION:
                    joint_name = native.spec.joint_names[index]
                    stiffness = self._session.config.position_stiffness_for(joint_name)
                    damping = self._session.config.position_damping_for(joint_name)
                    joint_force = stiffness * (target - float(data.qpos[qpos_address])) - damping * float(
                        data.qvel[dof_address]
                    )
                    joint_force += float(data.qfrc_bias[dof_address])
                elif mode is CommandMode.VELOCITY:
                    joint_force = self._session.config.velocity_gain * (target - float(data.qvel[dof_address]))
                    joint_force += float(data.qfrc_bias[dof_address])
                else:
                    joint_force = target
                if native.spec.joint_effort_limits:
                    effort_limit = native.spec.joint_effort_limits[index]
                    lower, upper = -effort_limit, effort_limit
                else:
                    effort_limit = self._session.config.max_motor_effort
                    lower, upper = -effort_limit, effort_limit
                data.qfrc_applied[dof_address] += max(lower, min(upper, joint_force))
        for path, (wrench_force, wrench_torque) in self._rigid_wrenches[environment].items():
            body_id = self._native[path].body_id
            assert body_id is not None
            data.xfrc_applied[body_id, :3] = wrench_force
            data.xfrc_applied[body_id, 3:] = wrench_torque

    def _apply_profiled_controls(self, environment: int) -> None:
        profiles = self._articulation_drive_profiles
        assert profiles is not None
        data = self._data[environment]
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0
        for path, commands in self._commands[environment].items():
            native = self._native[path]
            drive = profiles[path]
            for index, (mode, target) in enumerate(commands):
                qpos_address = native.joint_qpos_addresses[index]
                dof_address = native.joint_dof_addresses[index]
                if mode is CommandMode.POSITION:
                    joint_force = drive.position_stiffness[index] * (
                        target - float(data.qpos[qpos_address])
                    ) - drive.position_damping[index] * float(data.qvel[dof_address])
                    joint_force += float(data.qfrc_bias[dof_address])
                elif mode is CommandMode.VELOCITY:
                    joint_force = self._session.config.velocity_gain * (target - float(data.qvel[dof_address]))
                    joint_force += float(data.qfrc_bias[dof_address])
                else:
                    joint_force = target
                if native.spec.joint_effort_limits:
                    effort_limit = native.spec.joint_effort_limits[index]
                    lower, upper = -effort_limit, effort_limit
                else:
                    effort_limit = self._session.config.max_motor_effort
                    lower, upper = -effort_limit, effort_limit
                data.qfrc_applied[dof_address] += max(lower, min(upper, joint_force))
        for path, (wrench_force, wrench_torque) in self._rigid_wrenches[environment].items():
            body_id = self._native[path].body_id
            assert body_id is not None
            data.xfrc_applied[body_id, :3] = wrench_force
            data.xfrc_applied[body_id, 3:] = wrench_torque

    def step(self, count: int = 1) -> Tick:
        self._ensure("mujoco.world.step")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValidationError("step count must be positive", operation="mujoco.world.step")
        for _ in range(count):
            for _native_substep in range(self._native_substeps_per_logical_step):
                for environment, (model, data) in enumerate(zip(self._models, self._data, strict=True)):
                    self._apply_controls_for_step(environment)
                    mujoco.mj_step(model, data)
            for viewer in self._viewers:
                if not viewer.is_running():
                    raise LifecycleError(
                        "MuJoCo GUI viewer closed while the world is running",
                        operation="mujoco.world.step",
                        backend_id=self._session.descriptor.provider_id,
                        world_id=self.world_id,
                    )
                viewer.sync()
            self._step_index += 1
            for key, expiration in tuple(self._debug_expiration.items()):
                if expiration is not None and expiration <= self._step_index:
                    del self._debug[key]
                    del self._debug_expiration[key]
        self._scene_sequence += count
        return self.tick

    def _scene_entities(self) -> tuple[SceneEntityState, ...]:
        result: list[SceneEntityState] = []
        for entity in self._spec.entities:
            for environment in range(self._spec.environments.count):
                if entity.kind is EntityKind.RIGID_BODY:
                    state = self.read_rigid_body(self.resolve(entity.path))
                    pose = Pose(
                        tuple(float(value) for value in state.positions_m.rows()[environment]),  # type: ignore[arg-type]
                        tuple(float(value) for value in state.orientations_xyzw.rows()[environment]),  # type: ignore[arg-type]
                    )
                    linear = tuple(float(value) for value in state.linear_velocities_m_s.rows()[environment])
                    angular = tuple(float(value) for value in state.angular_velocities_rad_s.rows()[environment])
                    joints: tuple[float, ...] = ()
                    draggable = True
                elif entity.kind is EntityKind.ARTICULATION:
                    articulation = self.read_articulation(self.resolve(entity.path))
                    pose = entity.pose
                    linear = angular = (0.0, 0.0, 0.0)
                    joints = tuple(float(value) for value in articulation.joint_positions.rows()[environment])
                    draggable = False
                else:
                    pose = entity.pose
                    linear = angular = (0.0, 0.0, 0.0)
                    joints = ()
                    draggable = False
                result.append(
                    SceneEntityState(
                        entity.path,
                        entity.kind,
                        environment,
                        pose,
                        linear,  # type: ignore[arg-type]
                        angular,  # type: ignore[arg-type]
                        entity.joint_names,
                        joints,
                        (
                            SceneVisual(
                                "body",
                                SceneVisualKind.BOX,
                                dimensions_m=((0.5, 0.5, 0.5) if entity.box is None else entity.box.dimensions_m)
                                if entity.kind is EntityKind.RIGID_BODY
                                else (0.55, 0.45, 0.7),
                                color_rgba=((0.15, 0.7, 0.95, 1.0) if entity.box is None else entity.box.color_rgba)
                                if entity.kind is EntityKind.RIGID_BODY
                                else (0.92, 0.49, 0.16, 1.0),
                            ),
                        ),
                        draggable=draggable,
                        metadata=FrozenMap({"native_backend": "mujoco"}),
                    )
                )
        return tuple(result)

    def scene_snapshot(self) -> SceneSnapshot:
        self._ensure("mujoco.world.scene_snapshot")
        return SceneSnapshot(
            self._session.descriptor.provider_id,
            self.world_id,
            self.generation,
            self._scene_sequence,
            self.tick,
            self._scene_entities(),
        )

    def scene_delta(self, base_sequence: int) -> SceneDelta:
        self._ensure("mujoco.world.scene_delta")
        if (
            not isinstance(base_sequence, int)
            or isinstance(base_sequence, bool)
            or not 0 <= base_sequence <= self._scene_sequence
        ):
            raise ValidationError("base sequence is invalid", operation="mujoco.world.scene_delta")
        return SceneDelta(
            self.world_id,
            self.generation,
            base_sequence,
            self._scene_sequence,
            self.tick,
            () if base_sequence == self._scene_sequence else self._scene_entities(),
        )

    def _result(
        self,
        command: SceneCommand,
        status: SceneCommandStatus,
        code: str | None = None,
        message: str | None = None,
    ) -> SceneCommandResult:
        result = SceneCommandResult(
            command.command_id,
            status,
            self.generation,
            self._scene_sequence,
            self.tick,
            code,
            message,
        )
        self._scene_results[command.command_id] = result
        if len(self._scene_results) > self._session.config.max_cached_commands:
            del self._scene_results[next(iter(self._scene_results))]
        return result

    def _set_pose(self, path: EntityPath, environment: int, pose: Pose) -> None:
        native = self._native[path]
        assert native.free_qpos_address is not None and native.free_dof_address is not None
        data = self._data[environment]
        qpos = native.free_qpos_address
        data.qpos[qpos : qpos + 3] = pose.position
        x, y, z, w = pose.orientation_xyzw
        data.qpos[qpos + 3 : qpos + 7] = (w, x, y, z)
        data.qvel[native.free_dof_address : native.free_dof_address + 6] = 0.0
        mujoco.mj_forward(self._models[environment], data)

    def apply_scene_command(self, command: SceneCommand) -> SceneCommandResult:
        self._ensure("mujoco.world.apply_scene_command")
        if not isinstance(command, SceneCommand):
            raise ValidationError("operation requires SceneCommand", operation="mujoco.world.apply_scene_command")
        previous = self._scene_results.get(command.command_id)
        if previous is not None:
            return SceneCommandResult(
                command.command_id,
                SceneCommandStatus.DUPLICATE,
                previous.generation,
                previous.scene_sequence,
                previous.tick,
                message="command was already processed",
            )
        if command.expected_generation != self.generation:
            return self._result(command, SceneCommandStatus.REJECTED, "stale_generation", "generation mismatch")
        entity = self._entities.get(command.entity_path)
        if entity is None or command.environment_index >= self._spec.environments.count:
            return self._result(command, SceneCommandStatus.REJECTED, "target_not_found", "target does not exist")
        if entity.kind is not EntityKind.RIGID_BODY:
            return self._result(
                command,
                SceneCommandStatus.REJECTED,
                "unsupported_entity_kind",
                "only free rigid bodies are draggable",
            )
        environment = command.environment_index
        if command.kind is SceneCommandKind.SET_POSE:
            assert command.target_pose is not None
            self._set_pose(entity.path, environment, command.target_pose)
        elif command.kind is SceneCommandKind.DRAG_BEGIN:
            assert command.drag_id is not None
            if command.drag_mode is not SceneDragMode.KINEMATIC:
                return self._result(command, SceneCommandStatus.REJECTED, "unsupported_drag_mode", "use kinematic")
            if command.drag_id in self._drags:
                return self._result(command, SceneCommandStatus.REJECTED, "drag_exists", "drag already exists")
            self._drags[command.drag_id] = (entity.path, environment, self._rigid_pose(entity.path, environment))
        else:
            assert command.drag_id is not None
            active = self._drags.get(command.drag_id)
            if active is None or active[:2] != (entity.path, environment):
                return self._result(command, SceneCommandStatus.REJECTED, "drag_not_active", "drag is not active")
            if command.kind is SceneCommandKind.DRAG_UPDATE:
                assert command.target_pose is not None
                self._set_pose(entity.path, environment, command.target_pose)
            elif command.kind is SceneCommandKind.DRAG_CANCEL:
                self._set_pose(entity.path, environment, active[2])
                del self._drags[command.drag_id]
            else:
                del self._drags[command.drag_id]
        self._scene_sequence += 1
        return self._result(command, SceneCommandStatus.APPLIED)

    def _close(self, *, notify_session: bool) -> None:
        if self._state is WorldState.CLOSED:
            return
        self._state = WorldState.CLOSED
        for viewer in self._viewers:
            viewer.close()
        self._viewers.clear()
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
        self._entities.clear()
        if self._asset_lease is not None:
            self._asset_lease.close()
            self._asset_lease = None
        if notify_session:
            self._session._world_closed(self)

    def close(self) -> None:
        self._close(notify_session=True)

    def __enter__(self) -> MuJoCoWorld:
        self._ensure("mujoco.world.enter")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
