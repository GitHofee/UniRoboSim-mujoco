"""Demand-only native MuJoCo implementation of ``planning.scene@2``."""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from unirobosim import (
    PLANNING_SYSTEM_ENTITY_ID,
    PLANNING_SYSTEM_ENTITY_PATH,
    EntityKind,
    EntityPath,
    PlanningArticulationState,
    PlanningEntityDescriptor,
    PlanningEntityKind,
    PlanningEntityState,
    PlanningFrameDescriptor,
    PlanningFrameKind,
    PlanningFrameState,
    PlanningGeometryAxisConvention,
    PlanningGeometryContentProfile,
    PlanningGeometryDescriptor,
    PlanningGeometryDType,
    PlanningGeometryLease,
    PlanningGeometryLocalPose,
    PlanningGeometryMotionClass,
    PlanningGeometryPurpose,
    PlanningGeometryRepresentation,
    PlanningGeometryResourceDescriptor,
    PlanningGeometryResourceLayout,
    PlanningGeometryResourceRevokedError,
    PlanningGeometryStorageKind,
    PlanningGeometryTransform,
    PlanningHalfspaceGeometry,
    PlanningJointDescriptor,
    PlanningJointType,
    PlanningLinkDescriptor,
    PlanningLinkState,
    PlanningPose,
    PlanningPrimitiveGeometry,
    PlanningSceneCatalog,
    PlanningSceneContractError,
    PlanningSceneDelta,
    PlanningSceneDeltaContinuityError,
    PlanningSceneDeltaKind,
    PlanningSceneIncompleteError,
    PlanningSceneNotFoundError,
    PlanningSceneRepresentationError,
    PlanningSceneState,
    PlanningTwist,
    ResetResult,
    SceneCommand,
    SceneCommandResult,
    Tick,
    WorldSpec,
    WorldState,
)

from .articulation_drive import CompiledMuJoCoArticulationDrive
from .build_assets import BuildAssetLease
from .world import MuJoCoWorld, mujoco

_WORLD_FRAME_ID = "frame.world"
_SYSTEM_FRAME_ID = "frame.system.simulator_effective"
_SYSTEM_GEOMETRY_ID = "geometry.system.simulator_effective.ground"
_IDENTITY_POSITION = (0.0, 0.0, 0.0)
_IDENTITY_QUATERNION = (0.0, 0.0, 0.0, 1.0)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _provenance(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256(encoded)


def _xyzw(wxyz: Any) -> tuple[float, float, float, float]:
    return float(wxyz[1]), float(wxyz[2]), float(wxyz[3]), float(wxyz[0])


def _position(xyz: Any) -> tuple[float, float, float]:
    return float(xyz[0]), float(xyz[1]), float(xyz[2])


def _matrix_quaternion(matrix: Any) -> tuple[float, float, float, float]:
    value = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(value, np.asarray(matrix, dtype=np.float64).reshape(9))
    return _xyzw(value)


@dataclass(frozen=True, slots=True)
class _TopologyJoint:
    joint_id: str
    authored_name: str
    parent_body_id: int
    child_body_id: int
    native_joint_id: int | None
    joint_type: PlanningJointType
    axis_xyz: tuple[float, float, float]
    position_unit: str
    lower: float | None
    upper: float | None
    max_velocity: float | None
    max_effort: float | None


class _MemoryGeometryLease:
    def __init__(
        self,
        owner: MuJoCoPlanningWorld,
        descriptor: PlanningGeometryResourceDescriptor,
        payload: bytes,
    ) -> None:
        self._owner = owner
        self._descriptor = descriptor
        self._payload = payload
        self._closed = False

    @property
    def descriptor(self) -> PlanningGeometryResourceDescriptor:
        return self._descriptor

    @property
    def closed(self) -> bool:
        return self._closed or self._owner.state is WorldState.CLOSED

    def read(self, offset: int = 0, length: int | None = None) -> bytes:
        if self.closed:
            raise PlanningGeometryResourceRevokedError(
                "planning geometry lease is revoked",
                operation="mujoco.planning_geometry.read",
                world_id=self._descriptor.world_id,
            ) from None
        if type(offset) is not int or offset < 0 or offset > len(self._payload):
            raise PlanningSceneContractError(
                "planning geometry read offset is invalid",
                operation="mujoco.planning_geometry.read",
            ) from None
        if length is None:
            stop = len(self._payload)
        elif type(length) is not int or length < 0:
            raise PlanningSceneContractError(
                "planning geometry read length is invalid",
                operation="mujoco.planning_geometry.read",
            ) from None
        else:
            stop = min(len(self._payload), offset + length)
        return self._payload[offset:stop]

    def close(self) -> None:
        self._closed = True
        self._payload = b""


class MuJoCoPlanningWorld(MuJoCoWorld):
    """Separate concrete type so ordinary worlds pay no planning tick cost."""

    def __init__(
        self,
        session: Any,
        spec: WorldSpec,
        generation: int,
        asset_lease: BuildAssetLease | None,
        *,
        native_substeps_per_logical_step: int,
        articulation_drive_profiles: Mapping[EntityPath, CompiledMuJoCoArticulationDrive] | None = None,
    ) -> None:
        super().__init__(
            session,
            spec,
            generation,
            asset_lease,
            native_substeps_per_logical_step=native_substeps_per_logical_step,
            articulation_drive_profiles=articulation_drive_profiles,
        )
        self._planning_sequence = 1
        self._planning_world_revision = 1
        self._planning_transform_revision = 1
        self._planning_geometry_payloads: dict[str, bytes] = {}
        self._planning_topology: dict[str, tuple[_TopologyJoint, ...]] = {}
        self._planning_body_ids: dict[str, tuple[int, ...]] = {}
        self._planning_geom_ids: dict[str, int] = {}
        self._planning_leases: list[_MemoryGeometryLease] = []
        self._planning_catalogs = tuple(
            self._build_planning_catalog(environment) for environment in range(self._spec.environments.count)
        )
        self._planning_history: list[dict[int, PlanningSceneState]] = [{} for _ in range(self._spec.environments.count)]
        self._publish_planning_state()

    def _environment(self, value: int, operation: str) -> int:
        if type(value) is not int or not 0 <= value < self._spec.environments.count:
            raise PlanningSceneContractError(
                "planning environment index is invalid",
                operation=operation,
                world_id=self.world_id,
            ) from None
        return value

    def _asset_articulation(self) -> tuple[int, Any]:
        assets = [
            (index, entity)
            for index, entity in enumerate(self._spec.entities)
            if entity.kind.value == "articulation" and entity.asset_uri is not None
        ]
        if len(assets) != 1:
            raise PlanningSceneIncompleteError(
                "MuJoCo planning preview requires exactly one native asset articulation",
                operation="planning_scene.preflight",
                backend_id=self._session.descriptor.provider_id,
                world_id=self.world_id,
            ) from None
        return assets[0]

    def _urdf_joints(self, entity: Any) -> dict[str, dict[str, object]]:
        assert entity.asset_uri is not None
        path = self._local_asset_path(entity.asset_uri, entity, self._asset_lease)
        if path.suffix.lower() != ".urdf":
            raise PlanningSceneIncompleteError(
                "MuJoCo DROID planning preview requires a URDF source",
                operation="planning_scene.preflight",
                entity_path=entity.path.value,
            ) from None
        root = ET.parse(path).getroot()
        result: dict[str, dict[str, object]] = {}
        for joint in root.findall("joint"):
            child = joint.find("child")
            parent = joint.find("parent")
            if child is None or parent is None:
                continue
            child_name = child.attrib.get("link", "")
            axis_element = joint.find("axis")
            axis_text = "0 0 1" if axis_element is None else axis_element.attrib.get("xyz", "0 0 1")
            axis_values = tuple(float(item) for item in axis_text.split())
            if len(axis_values) != 3 or math.isclose(sum(value * value for value in axis_values), 0.0):
                axis_values = (0.0, 0.0, 1.0)
            norm = math.sqrt(sum(value * value for value in axis_values))
            axis = tuple(value / norm for value in axis_values)
            limit = joint.find("limit")

            def limit_value(name: str, limit_element: ET.Element | None = limit) -> float | None:
                if limit_element is None or name not in limit_element.attrib:
                    return None
                return float(limit_element.attrib[name])

            result[child_name] = {
                "name": joint.attrib.get("name", f"fixed_{child_name}"),
                "type": joint.attrib.get("type", "fixed"),
                "parent": parent.attrib.get("link", ""),
                "axis": axis,
                "lower": limit_value("lower"),
                "upper": limit_value("upper"),
                "velocity": limit_value("velocity"),
                "effort": limit_value("effort"),
            }
        return result

    def _native_inventory(
        self,
        entity_index: int,
        entity: Any,
    ) -> tuple[tuple[int, ...], tuple[_TopologyJoint, ...]]:
        key = entity.path.value
        existing_bodies = self._planning_body_ids.get(key)
        existing_joints = self._planning_topology.get(key)
        if existing_bodies is not None and existing_joints is not None:
            return existing_bodies, existing_joints
        model = self._models[0]
        root_body_id = self._native[entity.path].body_id
        if root_body_id is None:
            raise PlanningSceneIncompleteError(
                "native articulation root body is unavailable",
                operation="planning_scene.preflight",
                entity_path=entity.path.value,
            ) from None

        def belongs(body_id: int) -> bool:
            current = body_id
            while current != 0 and current != root_body_id:
                current = int(model.body_parentid[current])
            return current == root_body_id

        body_ids = tuple(body_id for body_id in range(1, int(model.nbody)) if belongs(body_id))
        body_names = {
            body_id: str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}")
            for body_id in body_ids
        }
        authored = self._urdf_joints(entity)
        topology: list[_TopologyJoint] = []
        for child_body_id in body_ids:
            if child_body_id == root_body_id:
                continue
            parent_body_id = int(model.body_parentid[child_body_id])
            child_name = body_names[child_body_id]
            source = authored.get(child_name)
            if source is None or source["parent"] != body_names[parent_body_id]:
                raise PlanningSceneIncompleteError(
                    "URDF and compiled MuJoCo body topology do not match",
                    operation="planning_scene.preflight",
                    entity_path=entity.path.value,
                ) from None
            joint_count = int(model.body_jntnum[child_body_id])
            if joint_count > 1:
                raise PlanningSceneIncompleteError(
                    "multi-DOF MuJoCo body joints are outside the DROID planning profile",
                    operation="planning_scene.preflight",
                    entity_path=entity.path.value,
                ) from None
            native_joint_id = int(model.body_jntadr[child_body_id]) if joint_count == 1 else None
            if native_joint_id is None:
                joint_type = PlanningJointType.FIXED
                position_unit = "rad"
                lower = upper = None
            else:
                native_type = int(model.jnt_type[native_joint_id])
                if native_type == int(mujoco.mjtJoint.mjJNT_SLIDE):
                    joint_type = PlanningJointType.PRISMATIC
                    position_unit = "m"
                elif native_type == int(mujoco.mjtJoint.mjJNT_HINGE):
                    joint_type = (
                        PlanningJointType.REVOLUTE
                        if bool(model.jnt_limited[native_joint_id])
                        else PlanningJointType.CONTINUOUS
                    )
                    position_unit = "rad"
                else:
                    raise PlanningSceneIncompleteError(
                        "native MuJoCo articulation contains an unsupported physical joint",
                        operation="planning_scene.preflight",
                        entity_path=entity.path.value,
                    ) from None
                if bool(model.jnt_limited[native_joint_id]):
                    lower = float(model.jnt_range[native_joint_id][0])
                    upper = float(model.jnt_range[native_joint_id][1])
                else:
                    lower = upper = None
            axis = source["axis"] if native_joint_id is None else _position(model.jnt_axis[native_joint_id])
            topology.append(
                _TopologyJoint(
                    joint_id=f"joint.entity.{entity_index:03d}.{child_body_id:04d}",
                    authored_name=str(source["name"]),
                    parent_body_id=parent_body_id,
                    child_body_id=child_body_id,
                    native_joint_id=native_joint_id,
                    joint_type=joint_type,
                    axis_xyz=axis,  # type: ignore[arg-type]
                    position_unit=position_unit,
                    lower=lower,
                    upper=upper,
                    max_velocity=source["velocity"],  # type: ignore[arg-type]
                    max_effort=source["effort"],  # type: ignore[arg-type]
                )
            )
        result = tuple(topology)
        self._planning_body_ids[key] = body_ids
        self._planning_topology[key] = result
        return body_ids, result

    @staticmethod
    def _entity_id(index: int) -> str:
        return f"entity.{index:03d}"

    @staticmethod
    def _entity_frame_id(index: int) -> str:
        return f"frame.entity.{index:03d}"

    @staticmethod
    def _link_id(index: int, body_id: int) -> str:
        return f"link.entity.{index:03d}.{body_id:04d}"

    @staticmethod
    def _link_frame_id(index: int, body_id: int) -> str:
        return f"frame.entity.{index:03d}.link.{body_id:04d}"

    @staticmethod
    def _joint_frame_id(index: int, body_id: int) -> str:
        return f"frame.entity.{index:03d}.joint.{body_id:04d}"

    def _mesh_geometry(
        self,
        entity_index: int,
        entity_id: str,
        body_id: int,
        geom_id: int,
    ) -> PlanningGeometryDescriptor:
        model = self._models[0]
        mesh_id = int(model.geom_dataid[geom_id])
        vertex_address = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        face_address = int(model.mesh_faceadr[mesh_id])
        face_count = int(model.mesh_facenum[mesh_id])
        vertices = np.asarray(
            model.mesh_vert[vertex_address : vertex_address + vertex_count],
            dtype="<f4",
        )
        faces = np.asarray(
            model.mesh_face[face_address : face_address + face_count],
            dtype="<i4",
        )
        payload = vertices.tobytes(order="C") + faces.tobytes(order="C")
        digest = _sha256(payload)
        geometry_id = f"geometry.entity.{entity_index:03d}.{geom_id:04d}"
        resource_id = f"resource.entity.{entity_index:03d}.mesh.{mesh_id:04d}.{geom_id:04d}"
        layout = PlanningGeometryResourceLayout(
            PlanningGeometryRepresentation.TRIANGLE_MESH,
            PlanningGeometryContentProfile.MESH_TRIANGLES_RAW_LE_V1,
            vertex_dtype=PlanningGeometryDType.FLOAT32,
            vertex_shape=(vertex_count, 3),
            index_dtype=PlanningGeometryDType.INT32,
            index_shape=(face_count, 3),
        )
        self._planning_geometry_payloads[geometry_id] = payload
        self._planning_geom_ids[geometry_id] = geom_id
        return PlanningGeometryDescriptor(
            geometry_id,
            entity_id,
            self._link_id(entity_index, body_id),
            self._link_frame_id(entity_index, body_id),
            PlanningGeometryPurpose.COLLISION,
            PlanningGeometryRepresentation.TRIANGLE_MESH,
            PlanningGeometryLocalPose(
                _position(model.geom_pos[geom_id]),
                _xyzw(model.geom_quat[geom_id]),
            ),
            (1.0, 1.0, 1.0),
            PlanningGeometryMotionClass.DYNAMIC,
            int(model.geom_contype[geom_id]),
            int(model.geom_conaffinity[geom_id]),
            _provenance(
                {
                    "adapter": "unirobosim-mujoco@0.9.4",
                    "native_profile": "mujoco-3.11-compiled-mesh",
                    "geom_id": geom_id,
                    "mesh_id": mesh_id,
                    "sha256": digest,
                }
            ),
            resource_id=resource_id,
            sha256=digest,
            content_profile=PlanningGeometryContentProfile.MESH_TRIANGLES_RAW_LE_V1,
            resource_layout=layout,
        )

    def _build_planning_catalog(self, environment_index: int) -> PlanningSceneCatalog:
        model = self._models[0]
        entities: list[PlanningEntityDescriptor] = [
            PlanningEntityDescriptor(
                PLANNING_SYSTEM_ENTITY_ID,
                PLANNING_SYSTEM_ENTITY_PATH,
                PlanningEntityKind.OTHER,
                True,
                _SYSTEM_FRAME_ID,
                (),
                (_SYSTEM_FRAME_ID,),
                (_SYSTEM_GEOMETRY_ID,),
            )
        ]
        links: list[PlanningLinkDescriptor] = []
        joints: list[PlanningJointDescriptor] = []
        frames: list[PlanningFrameDescriptor] = [
            PlanningFrameDescriptor(_WORLD_FRAME_ID, PlanningFrameKind.WORLD, None, None, None),
            PlanningFrameDescriptor(
                _SYSTEM_FRAME_ID,
                PlanningFrameKind.ENTITY,
                _WORLD_FRAME_ID,
                PLANNING_SYSTEM_ENTITY_ID,
                None,
            ),
        ]
        geometries: list[PlanningGeometryDescriptor] = [
            PlanningGeometryDescriptor(
                _SYSTEM_GEOMETRY_ID,
                PLANNING_SYSTEM_ENTITY_ID,
                None,
                _SYSTEM_FRAME_ID,
                PlanningGeometryPurpose.COLLISION,
                PlanningGeometryRepresentation.HALFSPACE,
                PlanningGeometryLocalPose(),
                (1.0, 1.0, 1.0),
                PlanningGeometryMotionClass.STATIC,
                1,
                2**32 - 1,
                _provenance(
                    {
                        "adapter": "unirobosim-mujoco@0.9.4",
                        "native_profile": "mujoco-plane-z0",
                    }
                ),
                PlanningHalfspaceGeometry(),
            )
        ]
        asset_index, asset_entity = self._asset_articulation()
        for entity_index, entity in enumerate(self._spec.entities):
            entity_id = self._entity_id(entity_index)
            entity_frame_id = self._entity_frame_id(entity_index)
            if entity is not asset_entity:
                frames.append(
                    PlanningFrameDescriptor(
                        entity_frame_id,
                        PlanningFrameKind.ENTITY,
                        _WORLD_FRAME_ID,
                        entity_id,
                        None,
                    )
                )
                if entity.kind is EntityKind.RIGID_BODY and entity.box is not None:
                    native_body_id = self._native[entity.path].body_id
                    assert native_body_id is not None
                    native_geom_ids = tuple(
                        geom_id
                        for geom_id in range(1, int(model.ngeom))
                        if int(model.geom_bodyid[geom_id]) == native_body_id
                    )
                    if len(native_geom_ids) != 1 or int(model.geom_type[native_geom_ids[0]]) != int(
                        mujoco.mjtGeom.mjGEOM_BOX
                    ):
                        raise PlanningSceneIncompleteError(
                            "native MuJoCo rigid-box collision inventory is incomplete",
                            operation="planning_scene.preflight",
                            entity_path=entity.path.value,
                        ) from None
                    native_geom_id = native_geom_ids[0]
                    link_id = self._link_id(entity_index, native_body_id)
                    link_frame_id = self._link_frame_id(entity_index, native_body_id)
                    geometry_id = f"geometry.entity.{entity_index:03d}.box"
                    frames.append(
                        PlanningFrameDescriptor(
                            link_frame_id,
                            PlanningFrameKind.LINK,
                            entity_frame_id,
                            entity_id,
                            link_id,
                        )
                    )
                    links.append(
                        PlanningLinkDescriptor(
                            link_id,
                            entity_id,
                            entity.path.name,
                            link_frame_id,
                            None,
                            (geometry_id,),
                        )
                    )
                    dimensions = tuple(float(value) * 2.0 for value in model.geom_size[native_geom_id])
                    local = PlanningGeometryLocalPose(
                        _position(model.geom_pos[native_geom_id]),
                        _xyzw(model.geom_quat[native_geom_id]),
                    )
                    geometries.append(
                        PlanningGeometryDescriptor(
                            geometry_id,
                            entity_id,
                            link_id,
                            link_frame_id,
                            PlanningGeometryPurpose.COLLISION,
                            PlanningGeometryRepresentation.BOX,
                            local,
                            (1.0, 1.0, 1.0),
                            PlanningGeometryMotionClass.DYNAMIC,
                            int(model.geom_contype[native_geom_id]),
                            int(model.geom_conaffinity[native_geom_id]),
                            _provenance(
                                {
                                    "adapter": "unirobosim-mujoco@0.9.4",
                                    "native_profile": "mujoco-box",
                                    "dimensions_m": dimensions,
                                }
                            ),
                            PlanningPrimitiveGeometry(PlanningGeometryRepresentation.BOX, dimensions),
                        )
                    )
                    self._planning_body_ids[entity.path.value] = (native_body_id,)
                    self._planning_geom_ids[geometry_id] = native_geom_id
                    entities.append(
                        PlanningEntityDescriptor(
                            entity_id,
                            entity.path.value,
                            PlanningEntityKind.RIGID_OBJECT,
                            True,
                            entity_frame_id,
                            (link_id,),
                            tuple(sorted((entity_frame_id, link_frame_id))),
                            (geometry_id,),
                        )
                    )
                else:
                    entities.append(
                        PlanningEntityDescriptor(
                            entity_id,
                            entity.path.value,
                            PlanningEntityKind.OTHER,
                            True,
                            entity_frame_id,
                            (),
                            (entity_frame_id,),
                            (),
                        )
                    )
                continue
            body_ids, topology = self._native_inventory(asset_index, asset_entity)
            root_body_id = self._native[entity.path].body_id
            assert root_body_id is not None
            frames.append(
                PlanningFrameDescriptor(
                    entity_frame_id,
                    PlanningFrameKind.ENTITY,
                    _WORLD_FRAME_ID,
                    entity_id,
                    None,
                )
            )
            topology_by_child = {joint.child_body_id: joint for joint in topology}
            geometry_ids_by_body: dict[int, list[str]] = {body_id: [] for body_id in body_ids}
            for geom_id in range(1, int(model.ngeom)):
                body_id = int(model.geom_bodyid[geom_id])
                if body_id not in geometry_ids_by_body:
                    continue
                if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                    raise PlanningSceneIncompleteError(
                        "DROID MuJoCo planning profile encountered a non-mesh native collider",
                        operation="planning_scene.preflight",
                        entity_path=entity.path.value,
                    ) from None
                geometry = self._mesh_geometry(entity_index, entity_id, body_id, geom_id)
                geometries.append(geometry)
                geometry_ids_by_body[body_id].append(geometry.geometry_id)
            body_names = {
                body_id: str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}")
                for body_id in body_ids
            }
            for body_id in body_ids:
                link_id = self._link_id(entity_index, body_id)
                link_frame_id = self._link_frame_id(entity_index, body_id)
                if body_id == root_body_id:
                    parent_link_id = None
                    parent_frame_id = entity_frame_id
                else:
                    topology_joint = topology_by_child[body_id]
                    parent_link_id = self._link_id(entity_index, topology_joint.parent_body_id)
                    joint_frame_id = self._joint_frame_id(entity_index, body_id)
                    frames.append(
                        PlanningFrameDescriptor(
                            joint_frame_id,
                            PlanningFrameKind.JOINT,
                            self._link_frame_id(entity_index, topology_joint.parent_body_id),
                            entity_id,
                            link_id,
                        )
                    )
                    parent_frame_id = joint_frame_id
                    joints.append(
                        PlanningJointDescriptor(
                            topology_joint.joint_id,
                            entity_id,
                            topology_joint.authored_name,
                            parent_link_id,
                            link_id,
                            topology_joint.joint_type,
                            joint_frame_id,
                            topology_joint.axis_xyz,
                            topology_joint.position_unit,
                            topology_joint.lower,
                            topology_joint.upper,
                            topology_joint.max_velocity,
                            topology_joint.max_effort,
                        )
                    )
                frames.append(
                    PlanningFrameDescriptor(
                        link_frame_id,
                        PlanningFrameKind.LINK,
                        parent_frame_id,
                        entity_id,
                        link_id,
                    )
                )
                links.append(
                    PlanningLinkDescriptor(
                        link_id,
                        entity_id,
                        body_names[body_id],
                        link_frame_id,
                        parent_link_id,
                        tuple(sorted(geometry_ids_by_body[body_id])),
                    )
                )
            robot_frame_ids = tuple(
                sorted(
                    (entity_frame_id,)
                    + tuple(self._link_frame_id(entity_index, body_id) for body_id in body_ids)
                    + tuple(self._joint_frame_id(entity_index, item.child_body_id) for item in topology)
                )
            )
            robot_geometry_ids = tuple(
                sorted(geometry_id for values in geometry_ids_by_body.values() for geometry_id in values)
            )
            entities.append(
                PlanningEntityDescriptor(
                    entity_id,
                    entity.path.value,
                    PlanningEntityKind.ROBOT,
                    True,
                    entity_frame_id,
                    tuple(sorted(self._link_id(entity_index, body_id) for body_id in body_ids)),
                    robot_frame_ids,
                    robot_geometry_ids,
                    tuple(item.joint_id for item in topology),
                )
            )
        return PlanningSceneCatalog.build(
            self._session.descriptor.provider_id,
            self.world_id,
            self.generation,
            environment_index,
            1,
            1,
            tuple(sorted(entities, key=lambda item: item.entity_id)),
            tuple(sorted(links, key=lambda item: item.link_id)),
            tuple(sorted(joints, key=lambda item: item.joint_id)),
            tuple(sorted(frames, key=lambda item: item.frame_id)),
            tuple(sorted(geometries, key=lambda item: item.geometry_id)),
        )

    def _body_pose(self, environment_index: int, body_id: int) -> PlanningPose:
        data = self._data[environment_index]
        return PlanningPose(
            _WORLD_FRAME_ID,
            _position(data.xpos[body_id]),
            _xyzw(data.xquat[body_id]),
        )

    def _body_twist(self, environment_index: int, body_id: int) -> PlanningTwist:
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self._models[environment_index],
            self._data[environment_index],
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            velocity,
            0,
        )
        return PlanningTwist(
            _WORLD_FRAME_ID,
            _position(velocity[3:]),
            _position(velocity[:3]),
        )

    def _capture_planning_state(self, environment_index: int) -> PlanningSceneState:
        catalog = self._planning_catalogs[environment_index]
        data = self._data[environment_index]
        entity_states: list[PlanningEntityState] = [
            PlanningEntityState(
                PLANNING_SYSTEM_ENTITY_ID,
                PlanningPose(_WORLD_FRAME_ID, _IDENTITY_POSITION, _IDENTITY_QUATERNION),
                PlanningTwist(_WORLD_FRAME_ID),
            )
        ]
        link_states: list[PlanningLinkState] = []
        frame_states: list[PlanningFrameState] = [
            PlanningFrameState(
                _WORLD_FRAME_ID,
                PlanningPose(_WORLD_FRAME_ID, _IDENTITY_POSITION, _IDENTITY_QUATERNION),
            ),
            PlanningFrameState(
                _SYSTEM_FRAME_ID,
                PlanningPose(_WORLD_FRAME_ID, _IDENTITY_POSITION, _IDENTITY_QUATERNION),
            ),
        ]
        articulations: list[PlanningArticulationState] = []
        for entity_index, entity in enumerate(self._spec.entities):
            entity_id = self._entity_id(entity_index)
            entity_frame_id = self._entity_frame_id(entity_index)
            body_ids = self._planning_body_ids.get(entity.path.value)
            topology = self._planning_topology.get(entity.path.value)
            if body_ids is not None and entity.kind is EntityKind.RIGID_BODY:
                native_body_id = body_ids[0]
                pose = self._body_pose(environment_index, native_body_id)
                twist = self._body_twist(environment_index, native_body_id)
                entity_states.append(PlanningEntityState(entity_id, pose, twist))
                frame_states.extend(
                    (
                        PlanningFrameState(entity_frame_id, pose),
                        PlanningFrameState(self._link_frame_id(entity_index, native_body_id), pose),
                    )
                )
                link_states.append(PlanningLinkState(self._link_id(entity_index, native_body_id), pose, twist))
                continue
            if body_ids is None or topology is None:
                pose = PlanningPose(
                    _WORLD_FRAME_ID,
                    entity.pose.position,
                    entity.pose.orientation_xyzw,
                )
                entity_states.append(PlanningEntityState(entity_id, pose, PlanningTwist(_WORLD_FRAME_ID)))
                frame_states.append(PlanningFrameState(entity_frame_id, pose))
                continue
            root_body_id = self._native[entity.path].body_id
            assert root_body_id is not None
            root_pose = self._body_pose(environment_index, root_body_id)
            entity_states.append(
                PlanningEntityState(entity_id, root_pose, self._body_twist(environment_index, root_body_id))
            )
            frame_states.append(PlanningFrameState(entity_frame_id, root_pose))
            positions: list[float] = []
            velocities: list[float] = []
            units: list[str] = []
            for body_id in body_ids:
                pose = self._body_pose(environment_index, body_id)
                link_states.append(
                    PlanningLinkState(
                        self._link_id(entity_index, body_id),
                        pose,
                        self._body_twist(environment_index, body_id),
                    )
                )
                frame_states.append(PlanningFrameState(self._link_frame_id(entity_index, body_id), pose))
            for joint in topology:
                if joint.native_joint_id is None:
                    pose = self._body_pose(environment_index, joint.child_body_id)
                    position = velocity = 0.0
                else:
                    native_joint = joint.native_joint_id
                    pose = PlanningPose(
                        _WORLD_FRAME_ID,
                        _position(data.xanchor[native_joint]),
                        _xyzw(data.xquat[joint.child_body_id]),
                    )
                    qpos_address = int(self._models[environment_index].jnt_qposadr[native_joint])
                    dof_address = int(self._models[environment_index].jnt_dofadr[native_joint])
                    position = float(data.qpos[qpos_address])
                    velocity = float(data.qvel[dof_address])
                frame_states.append(
                    PlanningFrameState(
                        self._joint_frame_id(entity_index, joint.child_body_id),
                        pose,
                    )
                )
                positions.append(position)
                velocities.append(velocity)
                units.append(joint.position_unit)
            articulations.append(
                PlanningArticulationState(
                    entity_id,
                    tuple(item.joint_id for item in topology),
                    tuple(positions),
                    tuple(velocities),
                    tuple(units),
                )
            )
        geometry_transforms: list[PlanningGeometryTransform] = [
            PlanningGeometryTransform(
                _SYSTEM_GEOMETRY_ID,
                PlanningPose(_WORLD_FRAME_ID, _IDENTITY_POSITION, _IDENTITY_QUATERNION),
            )
        ]
        for geometry_id, native_geom_id in self._planning_geom_ids.items():
            geometry_transforms.append(
                PlanningGeometryTransform(
                    geometry_id,
                    PlanningPose(
                        _WORLD_FRAME_ID,
                        _position(data.geom_xpos[native_geom_id]),
                        _matrix_quaternion(data.geom_xmat[native_geom_id]),
                    ),
                )
            )
        state = PlanningSceneState(
            self._session.descriptor.provider_id,
            self.world_id,
            self.generation,
            environment_index,
            self.tick,
            self._planning_sequence,
            self._planning_world_revision,
            1,
            1,
            catalog.content_sha256,
            self._planning_transform_revision,
            1,
            _WORLD_FRAME_ID,
            tuple(sorted(entity_states, key=lambda item: item.entity_id)),
            tuple(sorted(link_states, key=lambda item: item.link_id)),
            tuple(sorted(frame_states, key=lambda item: item.frame_id)),
            tuple(sorted(articulations, key=lambda item: item.entity_id)),
            tuple(sorted(geometry_transforms, key=lambda item: item.geometry_id)),
        )
        state.validate_against(catalog)
        return state

    def _publish_planning_state(self) -> None:
        for environment in range(self._spec.environments.count):
            state = self._capture_planning_state(environment)
            history = self._planning_history[environment]
            history[state.sequence] = state
            while len(history) > 64:
                del history[min(history)]

    def planning_scene_catalog(self, environment_index: int = 0) -> PlanningSceneCatalog:
        self._ensure("mujoco.world.planning_scene_catalog")
        return self._planning_catalogs[self._environment(environment_index, "mujoco.world.planning_scene_catalog")]

    def planning_scene_state(self, environment_index: int = 0) -> PlanningSceneState:
        self._ensure("mujoco.world.planning_scene_state")
        environment = self._environment(environment_index, "mujoco.world.planning_scene_state")
        return self._planning_history[environment][self._planning_sequence]

    def planning_scene_delta(self, base_sequence: int, environment_index: int = 0) -> PlanningSceneDelta:
        self._ensure("mujoco.world.planning_scene_delta")
        operation = "mujoco.world.planning_scene_delta"
        environment = self._environment(environment_index, operation)
        if type(base_sequence) is not int or base_sequence < 1:
            raise PlanningSceneContractError(
                "planning delta base sequence is invalid",
                operation=operation,
            ) from None
        current = self._planning_history[environment][self._planning_sequence]
        previous = self._planning_history[environment].get(base_sequence)
        if previous is None:
            return PlanningSceneDelta(
                current.provider_id,
                current.world_id,
                current.generation,
                environment,
                current.tick,
                base_sequence,
                current.sequence,
                current.world_revision,
                current.world_revision,
                1,
                1,
                None,
                None,
                1,
                1,
                current.transform_revision,
                current.transform_revision,
                1,
                1,
                PlanningSceneDeltaKind.RESYNC,
                resync_required=True,
            )
        if previous.sequence == current.sequence:
            raise PlanningSceneDeltaContinuityError(
                "no committed planning delta exists after base_sequence",
                operation=operation,
                world_id=self.world_id,
            ) from None
        return PlanningSceneDelta(
            current.provider_id,
            current.world_id,
            current.generation,
            environment,
            current.tick,
            base_sequence,
            current.sequence,
            previous.world_revision,
            current.world_revision,
            1,
            1,
            current.catalog_content_sha256,
            current.catalog_content_sha256,
            1,
            1,
            previous.transform_revision,
            current.transform_revision,
            1,
            1,
            PlanningSceneDeltaKind.STATE,
            state=current,
        )

    def resolve_planning_geometry(
        self,
        geometry_id: str,
        representation: PlanningGeometryRepresentation | None = None,
        environment_index: int = 0,
    ) -> PlanningGeometryLease:
        self._ensure("mujoco.world.resolve_planning_geometry")
        operation = "mujoco.world.resolve_planning_geometry"
        environment = self._environment(environment_index, operation)
        if type(geometry_id) is not str:
            raise PlanningSceneContractError("geometry_id is invalid", operation=operation) from None
        catalog = self._planning_catalogs[environment]
        geometry = next((item for item in catalog.geometries if item.geometry_id == geometry_id), None)
        if geometry is None:
            raise PlanningSceneNotFoundError(
                "planning geometry does not exist",
                operation=operation,
                world_id=self.world_id,
            ) from None
        if representation is not None and representation is not geometry.representation:
            raise PlanningSceneRepresentationError(
                "requested planning representation does not match the catalog",
                operation=operation,
                world_id=self.world_id,
            ) from None
        payload = self._planning_geometry_payloads.get(geometry_id)
        if (
            payload is None
            or geometry.resource_id is None
            or geometry.resource_layout is None
            or geometry.sha256 is None
            or geometry.content_profile is None
        ):
            raise PlanningSceneRepresentationError(
                "inline planning geometry does not have a resource lease",
                operation=operation,
                world_id=self.world_id,
            ) from None
        token = _sha256(
            f"{self._session.session_id}|{self.world_id}|{self.generation}|{environment}|{geometry_id}".encode()
        )
        descriptor = PlanningGeometryResourceDescriptor(
            self._session.descriptor.provider_id,
            self.world_id,
            self.generation,
            environment,
            1,
            1,
            catalog.content_sha256,
            token,
            geometry.resource_id,
            geometry_id,
            geometry.representation,
            PlanningGeometryStorageKind.IMMUTABLE_MEMORY,
            f"memory.{token[:16]}",
            geometry.content_profile,
            "m",
            PlanningGeometryAxisConvention.RIGHT_HANDED_Z_UP,
            geometry.resource_layout,
            len(payload),
            geometry.sha256,
        )
        descriptor.validate_against(catalog)
        lease = _MemoryGeometryLease(self, descriptor, payload)
        self._planning_leases.append(lease)
        return lease

    def step(self, count: int = 1) -> Tick:
        result = super().step(count)
        self._planning_sequence += 1
        self._planning_world_revision += 1
        self._planning_transform_revision += 1
        self._publish_planning_state()
        return result

    def reset(self, environment_indices: Any = None) -> ResetResult:
        result = super().reset(environment_indices)
        self._planning_sequence += 1
        self._planning_world_revision += 1
        self._planning_transform_revision += 1
        self._publish_planning_state()
        return result

    def restore_checkpoint(self, checkpoint: Any) -> Any:
        result = super().restore_checkpoint(checkpoint)
        self._planning_sequence += 1
        self._planning_world_revision += 1
        self._planning_transform_revision += 1
        self._publish_planning_state()
        return result

    def apply_scene_command(self, command: SceneCommand) -> SceneCommandResult:
        before = self._scene_sequence
        result = super().apply_scene_command(command)
        if self._scene_sequence != before:
            self._planning_sequence += 1
            self._planning_world_revision += 1
            self._planning_transform_revision += 1
            self._publish_planning_state()
        return result

    def _close(self, *, notify_session: bool) -> None:
        for lease in getattr(self, "_planning_leases", ()):
            lease.close()
        super()._close(notify_session=notify_session)
