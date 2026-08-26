from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, cast

import pytest
from unirobosim import (
    PHYSICAL_WORLD_SCHEMA_VERSION,
    ArrayValue,
    ArticulationCommand,
    BuildInput,
    BuildResourceEntry,
    BuildResourceManifest,
    BuildSourceEntry,
    CameraModality,
    CameraSpec,
    CapabilityId,
    CapabilityRequirement,
    CommandError,
    CommandMode,
    EntityKind,
    EntityPath,
    EntitySpec,
    LocalSourceIdentity,
    PlanningJointType,
    Pose,
    WorldSpec,
)

from unirobosim_mujoco import MuJoCoProvider

_DROID_ROOT = Path("/home/ubuntu/projects/gen_data/data/robots/droid")
_DROID_URDF = _DROID_ROOT / "droid_mujoco.urdf"
_ARM = tuple(f"panda_joint{index}" for index in range(1, 8))
_GRIPPER = (
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
)


def _build_input() -> BuildInput:
    relative_paths = sorted(
        {_DROID_URDF.name}
        | {element.attrib["filename"] for element in ET.parse(_DROID_URDF).getroot().findall(".//mesh")}
    )
    resource_id = {
        relative_path: (
            "resource.droid.urdf"
            if relative_path == _DROID_URDF.name
            else f"resource.droid.asset.{hashlib.sha256(relative_path.encode()).hexdigest()[:20]}"
        )
        for relative_path in relative_paths
    }
    dependencies = tuple(sorted(resource_id[path] for path in relative_paths if path != _DROID_URDF.name))
    records = []
    for relative_path in relative_paths:
        source_path = (_DROID_ROOT / relative_path).resolve()
        payload = source_path.read_bytes()
        sha256 = hashlib.sha256(payload).hexdigest()
        stat = source_path.stat()
        selected = relative_path == _DROID_URDF.name
        entry = BuildResourceEntry(
            "entity.droid",
            "robot.droid",
            resource_id[relative_path],
            "simulation" if selected else "collision",
            "model/vnd.urdf+xml" if selected else "application/octet-stream",
            str(_DROID_URDF) if selected else relative_path,
            str(source_path),
            f"sha256:{sha256}",
            len(payload),
            sha256,
            selected,
            ("collision", "planning", "simulation", "visual"),
            relative_path,
            dependencies if selected else (),
        )
        source = BuildSourceEntry(
            entry.resource_id,
            "local-file",
            str(source_path.parent),
            source_path.name,
            LocalSourceIdentity(
                stat.st_dev,
                stat.st_ino,
                stat.st_mode,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            ),
            sha256,
        )
        records.append((entry, source))
    records.sort(key=lambda item: item[0].resource_id)
    return BuildInput(
        manifest=BuildResourceManifest(tuple(item[0] for item in records)),
        sources=tuple(item[1] for item in records),
    )


@pytest.mark.skipif(not _DROID_URDF.is_file(), reason="named DROID acceptance asset is unavailable")
def test_core_090_v5_droid_planning_units_resources_and_same_tick_commands() -> None:
    build_input = _build_input()
    joint_names = _ARM + _GRIPPER
    initial = (0.0, -0.2, 0.0, -1.8, 0.0, 1.6, 0.7) + (0.0,) * 6
    spec = WorldSpec(
        "mujoco-droid-core-090",
        (
            EntitySpec(
                EntityPath("/droid"),
                EntityKind.ARTICULATION,
                joint_names=joint_names,
                initial_joint_positions=initial,
                asset_uri=str(_DROID_URDF),
                joint_position_units=("rad",) * 13,
            ),
            EntitySpec(
                EntityPath("/camera"),
                EntityKind.CAMERA_SENSOR,
                pose=Pose((2.0, 0.0, 1.5), (0.0, 0.7071067811865475, 0.0, 0.7071067811865475)),
                camera=CameraSpec(1280, 720, modalities=(CameraModality.RGB,)),
            ),
        ),
        requirements=(CapabilityRequirement(CapabilityId("planning.scene@2")),),
        schema_version=PHYSICAL_WORLD_SCHEMA_VERSION,
        build_resource_manifest_sha256=build_input.manifest.sha256,
    )
    session = MuJoCoProvider().open()
    world = session.build(spec, build_input=build_input)
    assert world._asset_lease is not None
    snapshot_root = world._asset_lease.root
    planning_world = cast(Any, world)
    try:
        handle = world.resolve(EntityPath("/droid"))
        state = world.read_articulation(handle)
        assert state.joint_names == joint_names
        assert state.joint_position_units == ("rad",) * 13
        assert state.joint_velocity_units == ("rad/s",) * 13

        catalog = planning_world.planning_scene_catalog()
        robot = next(entity for entity in catalog.entities if entity.path == "/droid")
        authored_links = {link.authored_name for link in catalog.links if link.entity_id == robot.entity_id}
        moving = {
            joint.authored_name
            for joint in catalog.joints
            if joint.entity_id == robot.entity_id and joint.joint_type is not PlanningJointType.FIXED
        }
        assert "gripper_center" in authored_links
        assert moving == set(joint_names)
        assert len(robot.joint_ids) == 21
        assert len(robot.geometry_ids) == 19
        geometry = next(item for item in catalog.geometries if item.resource_id is not None)
        lease = planning_world.resolve_planning_geometry(geometry.geometry_id)
        assert hashlib.sha256(lease.read()).hexdigest() == geometry.sha256
        lease.close()

        before = world.read_articulation(handle).joint_positions.rows()[0]
        world.apply_articulation_command(
            ArticulationCommand(
                handle,
                CommandMode.POSITION,
                ArrayValue.from_rows(((0.15, -0.3, 0.1, -2.0, 0.1, 2.0, 0.5),)),
                degree_of_freedom_indices=tuple(range(7)),
                target_units=("rad",) * 7,
            )
        )
        world.apply_articulation_command(
            ArticulationCommand(
                handle,
                CommandMode.POSITION,
                ArrayValue.from_rows(((0.2, -0.2, 0.2, -0.2, -0.2, 0.2),)),
                degree_of_freedom_indices=tuple(range(7, 13)),
                target_units=("rad",) * 6,
            )
        )
        assert world.read_articulation(handle).joint_positions.rows()[0] == before
        with pytest.raises(CommandError):
            world.apply_articulation_command(
                ArticulationCommand(
                    handle,
                    CommandMode.POSITION,
                    ArrayValue.from_rows(((0.0,) * 6,)),
                    degree_of_freedom_indices=tuple(range(7, 13)),
                    target_units=("m",) * 6,
                )
            )
        assert world.read_articulation(handle).joint_positions.rows()[0] == before
        sequence = planning_world.planning_scene_state().sequence
        world.step(10)
        assert planning_world.planning_scene_delta(sequence).kind.value == "state"
        assert world.read_articulation(handle).joint_positions.rows()[0] != before
        rgb = world.read_sensor(world.resolve(EntityPath("/camera"))).channel(CameraModality.RGB)
        assert rgb.shape == (1, 720, 1280, 3)
        assert rgb.dtype == "uint8" and rgb.device == "cpu"
        if callable(getattr(ArrayValue, "from_uint8_bytes", None)):
            assert rgb.is_packed is True
            assert len(rgb.to_bytes()) == 720 * 1280 * 3
    finally:
        world.close()
        session.close()
    assert not snapshot_root.exists()
