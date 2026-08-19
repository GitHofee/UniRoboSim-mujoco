from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from unirobosim import (
    ArrayValue,
    ArticulationCommand,
    BoxGeometrySpec,
    CameraModality,
    CameraSpec,
    CapabilityNegotiationError,
    CommandError,
    CommandMode,
    DebugBatch,
    DebugLifetime,
    DebugPrimitive,
    DebugPrimitiveKind,
    EntityKind,
    EntityNotFoundError,
    EntityPath,
    EntitySpec,
    EnvironmentSpec,
    LifecycleError,
    PhysicsSpec,
    Pose,
    RigidBodyCommand,
    SceneCommand,
    SceneCommandKind,
    SceneCommandStatus,
    SceneDragMode,
    SessionState,
    Sim,
    StaleHandleError,
    UnsupportedCapabilityError,
    ValidationError,
    WorldSpec,
)

from unirobosim_mujoco import MuJoCoAdapterConfig, MuJoCoProvider
from unirobosim_mujoco.world import MuJoCoWorld


def _write_two_joint_urdf(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0"?>
<robot name="mapped">
  <link name="base">
    <inertial><mass value="1"/><origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  <link name="first">
    <inertial><mass value="1"/><origin xyz="0 0 0.1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.03"/>
    </inertial>
    <collision><geometry><box size="0.1 0.1 0.2"/></geometry></collision>
  </link>
  <link name="second">
    <inertial><mass value="1"/><origin xyz="0 0 0.1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.03"/>
    </inertial>
    <collision><geometry><box size="0.1 0.1 0.2"/></geometry></collision>
  </link>
  <joint name="joint_a" type="revolute">
    <parent link="base"/><child link="first"/><origin xyz="0 0 0.1"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="20" velocity="2"/>
  </joint>
  <joint name="joint_b" type="revolute">
    <parent link="first"/><child link="second"/><origin xyz="0 0 0.2"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="20" velocity="2"/>
  </joint>
</robot>\n""",
        encoding="utf-8",
    )


def _write_free_rigid_mjcf(path: Path) -> None:
    path.write_text(
        """<mujoco model="rigid_asset">
  <worldbody>
    <body name="object">
      <freejoint name="object_free"/>
      <geom type="box" size="0.1 0.2 0.3" mass="1"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )


def world_spec(*, camera: bool = False, gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> WorldSpec:
    entities = [
        EntitySpec(EntityPath("/box"), EntityKind.RIGID_BODY, pose=Pose((0.0, 0.0, 1.0))),
        EntitySpec(
            EntityPath("/cabinet"),
            EntityKind.ARTICULATION,
            pose=Pose((1.0, 0.0, 0.2)),
            joint_names=("door_hinge", "drawer_slide"),
            initial_joint_positions=(0.1, -0.2),
        ),
    ]
    if camera:
        entities.append(
            EntitySpec(
                EntityPath("/camera"),
                EntityKind.CAMERA_SENSOR,
                pose=Pose((2.0, 0.0, 1.5), (0.0, 0.7071067811865475, 0.0, 0.7071067811865475)),
                camera=CameraSpec(32, 24, modalities=(CameraModality.RGB, CameraModality.DEPTH)),
            )
        )
    return WorldSpec(
        "mujoco-test",
        tuple(entities),
        environments=EnvironmentSpec(2),
        physics=PhysicsSpec(time_step_seconds=0.002, gravity_m_s2=gravity),
    )


def scene_command(
    world: MuJoCoWorld,
    kind: SceneCommandKind,
    command_id: str,
    *,
    pose: Pose | None = None,
    drag_id: str | None = None,
    mode: SceneDragMode | None = None,
) -> SceneCommand:
    return SceneCommand(
        command_id,
        "chrome",
        "lease",
        world.generation,
        kind,
        EntityPath("/box"),
        1,
        pose,
        drag_id,
        mode,
        (0.0, 0.0, 1.0) if kind is SceneCommandKind.DRAG_BEGIN else None,
    )


def test_probe_config_and_lifecycle() -> None:
    provider = MuJoCoProvider()
    probe = provider.probe()
    assert probe.available and provider.descriptor.provider_id == "google-deepmind.mujoco"
    with pytest.raises(ValidationError):
        MuJoCoAdapterConfig(render_width=0)
    for field, value in (
        ("position_stiffness", -1.0),
        ("position_damping", float("nan")),
        ("velocity_gain", -1.0),
    ):
        with pytest.raises(ValidationError):
            MuJoCoAdapterConfig(**{field: value})
    session = provider.open()
    world = session.build(world_spec())
    assert session.state is SessionState.READY and world.build_report.environment_count == 2
    world.close()
    assert session.state.value == "open"
    session.close()


def test_native_rigid_articulation_wrench_reset_and_contact() -> None:
    session = MuJoCoProvider().open()
    world = session.build(world_spec())
    try:
        box = world.resolve(EntityPath("/box"))
        cabinet = world.resolve(EntityPath("/cabinet"))
        rigid = world.read_rigid_body(box)
        assert rigid.positions_m.rows() == ((0.0, 0.0, 1.0),) * 2
        articulation = world.read_articulation(cabinet)
        assert all(row == pytest.approx((0.1, -0.2)) for row in articulation.joint_positions.rows())
        world.apply_rigid_body_command(
            RigidBodyCommand(
                box,
                ArrayValue.from_rows(((10.0, 0.0, 0.0),)),
                ArrayValue.from_rows(((0.0, 0.0, 0.0),)),
                environment_indices=(1,),
            )
        )
        world.apply_articulation_command(
            ArticulationCommand(
                cabinet,
                CommandMode.POSITION,
                ArrayValue.from_rows(((0.4,),)),
                environment_indices=(0,),
                degree_of_freedom_indices=(0,),
            )
        )
        world.step(50)
        state = world.read_rigid_body(box)
        assert state.positions_m.rows()[1][0] > state.positions_m.rows()[0][0]
        assert world.read_articulation(cabinet).joint_positions.rows()[0][0] > 0.1
        contact = world.read_contact(box)
        assert contact.in_contact.shape == (2,)
        world.reset((1,))
        assert world.read_rigid_body(box).positions_m.rows()[1] == pytest.approx((0.0, 0.0, 1.0))
    finally:
        session.close()


def test_box_visual_and_contact_material_are_lowered_to_native() -> None:
    path = EntityPath("/red_box")
    spec = WorldSpec(
        "mujoco-material",
        (
            EntitySpec(
                path,
                EntityKind.RIGID_BODY,
                pose=Pose((0.0, 0.0, 1.0)),
                box=BoxGeometrySpec(
                    color_rgba=(1.0, 0.0, 0.0, 1.0),
                    static_friction=2.0,
                    dynamic_friction=1.75,
                ),
            ),
        ),
    )
    session = MuJoCoProvider().open()
    world = session.build(spec)
    try:
        native = world._native[path]
        assert native.body_id is not None
        geom_id = int(world._models[0].body_geomadr[native.body_id])
        assert tuple(world._models[0].geom_friction[geom_id]) == pytest.approx((2.0, 0.005, 0.0001))
        assert tuple(world._models[0].geom_rgba[geom_id]) == pytest.approx((1.0, 0.0, 0.0, 1.0))
    finally:
        session.close()


def test_portable_joint_effort_limits_override_native_defaults() -> None:
    path = EntityPath("/arm")
    spec = WorldSpec(
        "mujoco-portable-effort",
        (
            EntitySpec(
                path,
                EntityKind.ARTICULATION,
                joint_names=("shoulder", "wrist"),
                joint_effort_limits=(3.0, 4.0),
            ),
        ),
    )
    session = MuJoCoProvider().open()
    world = session.build(spec)
    try:
        world._commands[0][path] = [(CommandMode.POSITION, 100.0), (CommandMode.POSITION, -100.0)]
        world._apply_controls(0)
        native = world._native[path]
        efforts = tuple(float(world._data[0].qfrc_applied[address]) for address in native.joint_dof_addresses)
        assert efforts == pytest.approx((3.0, -4.0))
    finally:
        session.close()


def test_native_urdf_joint_mapping_inertia_repair_and_control(tmp_path: Path) -> None:
    asset = tmp_path / "mapped.urdf"
    _write_two_joint_urdf(asset)
    spec = WorldSpec(
        "mujoco-urdf",
        (
            EntitySpec(
                EntityPath("/asset"),
                EntityKind.ARTICULATION,
                joint_names=("joint_b", "joint_a"),
                initial_joint_positions=(-0.2, 0.1),
                asset_uri=str(asset),
            ),
        ),
        physics=PhysicsSpec(time_step_seconds=0.002, gravity_m_s2=(0.0, 0.0, 0.0)),
    )
    session = MuJoCoProvider().open()
    world = session.build(spec)
    try:
        handle = world.resolve(EntityPath("/asset"))
        assert world.read_articulation(handle).joint_positions.rows()[0] == pytest.approx((-0.2, 0.1))
        native = world._native[EntityPath("/asset")]
        assert tuple(world._models[0].qpos0[address] for address in native.joint_qpos_addresses) == (0.0, 0.0)
        world.apply_articulation_command(
            ArticulationCommand(handle, CommandMode.POSITION, ArrayValue.from_rows(((0.35,),)), None, (0,))
        )
        world.step(200)
        assert world.read_articulation(handle).joint_positions.rows()[0][0] == pytest.approx(0.35, abs=0.01)
        world.reset()
        assert world.read_articulation(handle).joint_positions.rows()[0] == pytest.approx((-0.2, 0.1))
    finally:
        session.close()


def test_native_mjcf_rigid_asset_easyapi_state_wrench_and_reset(tmp_path: Path) -> None:
    asset = tmp_path / "rigid.xml"
    _write_free_rigid_mjcf(asset)
    expected = (1.0, 2.0, 3.0)
    sim = Sim(
        provider=MuJoCoProvider(),
        world_id="mujoco-native-rigid",
        time_step_seconds=0.002,
        gravity_m_s2=(0.0, 0.0, 0.0),
    )
    body = sim.add_rigid_body("object", asset_uri=str(asset), position_m=expected)
    try:
        sim.start()
        assert body.state.positions_m.rows()[0] == pytest.approx(expected)
        body.apply_wrench((2.0, 0.0, 0.0))
        sim.step(20)
        assert body.state.positions_m.rows()[0][0] > expected[0]
        sim.reset()
        assert body.state.positions_m.rows()[0] == pytest.approx(expected)
    finally:
        sim.close()


def test_native_debug_scene_snapshot_and_drag_transaction() -> None:
    session = MuJoCoProvider().open()
    world = session.build(world_spec())
    try:
        primitive = DebugPrimitive(
            "point",
            "test",
            DebugPrimitiveKind.POINT_SET,
            ArrayValue.from_nested([[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]]),
            (0, 1),
            lifetime=DebugLifetime.steps(1),
        )
        assert world.publish_debug(DebugBatch((primitive,))).accepted_count == 1
        assert world.scene_snapshot().entities[1].draggable
        begin = scene_command(
            world,
            SceneCommandKind.DRAG_BEGIN,
            "begin",
            drag_id="drag",
            mode=SceneDragMode.KINEMATIC,
        )
        assert world.apply_scene_command(begin).status is SceneCommandStatus.APPLIED
        target = Pose((2.0, 3.0, 4.0))
        update = scene_command(
            world,
            SceneCommandKind.DRAG_UPDATE,
            "update",
            pose=target,
            drag_id="drag",
        )
        world.apply_scene_command(update)
        world.apply_scene_command(scene_command(world, SceneCommandKind.DRAG_END, "end", drag_id="drag"))
        assert world.read_rigid_body(world.resolve(EntityPath("/box"))).positions_m.rows()[1] == target.position
        assert world.apply_scene_command(update).status is SceneCommandStatus.DUPLICATE
        constraint = replace(begin, command_id="constraint", drag_id="constraint", drag_mode=SceneDragMode.CONSTRAINT)
        assert world.apply_scene_command(constraint).error_code == "unsupported_drag_mode"
        world.step()
        assert world.clear_debug() == 0
    finally:
        session.close()


def test_unsupported_soft_matter_fails_explicitly() -> None:
    session = MuJoCoProvider().open()
    world = session.build(world_spec())
    try:
        with pytest.raises(UnsupportedCapabilityError):
            world.read_deformable(world.resolve(EntityPath("/box")))
    finally:
        session.close()


def test_native_offscreen_camera_rgb_and_depth() -> None:
    session = MuJoCoProvider().open()
    world = session.build(world_spec(camera=True))
    try:
        sample = world.read_sensor(world.resolve(EntityPath("/camera")))
        assert sample.channel(CameraModality.RGB).shape == (2, 24, 32, 3)
        assert sample.channel(CameraModality.DEPTH).shape == (2, 24, 32)
    finally:
        session.close()


def test_lifecycle_and_command_validation_boundaries() -> None:
    with pytest.raises(ValidationError):
        MuJoCoProvider(object())  # type: ignore[arg-type]
    provider = MuJoCoProvider()
    session = provider.open()
    with pytest.raises(LifecycleError):
        provider.open()
    with pytest.raises(ValidationError):
        session.build(object())  # type: ignore[arg-type]
    world = session.build(world_spec())
    box = world.resolve(EntityPath("/box"))
    cabinet = world.resolve(EntityPath("/cabinet"))
    with pytest.raises(LifecycleError):
        session.build(world_spec())
    with pytest.raises(EntityNotFoundError):
        world.resolve(EntityPath("/missing"))
    with pytest.raises(ValidationError):
        world.resolve("/box")  # type: ignore[arg-type]
    with pytest.raises(StaleHandleError):
        world.read_rigid_body(replace(box, token="foreign"))
    with pytest.raises(CommandError):
        world.read_rigid_body(cabinet)
    with pytest.raises(CommandError):
        world.read_articulation(box)
    with pytest.raises(CommandError):
        world.apply_rigid_body_command(object())  # type: ignore[arg-type]
    with pytest.raises(CommandError):
        world.apply_articulation_command(object())  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        world.reset((0, 0))
    with pytest.raises(ValidationError):
        world.step(0)
    with pytest.raises(ValidationError):
        world.read_contact(box, -1.0)
    world.close()
    assert session.state is SessionState.OPEN
    with pytest.raises(LifecycleError):
        world.step()
    session.close()
    session.close()
    with pytest.raises(LifecycleError):
        session.negotiate(())


def test_velocity_effort_debug_and_scene_rejection_paths() -> None:
    session = MuJoCoProvider(MuJoCoAdapterConfig(max_cached_commands=2)).open()
    world = session.build(world_spec())
    try:
        box = world.resolve(EntityPath("/box"))
        cabinet = world.resolve(EntityPath("/cabinet"))
        for mode, target in ((CommandMode.VELOCITY, 0.5), (CommandMode.EFFORT, 0.25)):
            world.apply_articulation_command(
                ArticulationCommand(
                    cabinet,
                    mode,
                    ArrayValue.from_rows(((target,),)),
                    environment_indices=(0,),
                    degree_of_freedom_indices=(1,),
                )
            )
            world.step()
        with pytest.raises(CommandError):
            world.apply_articulation_command(
                ArticulationCommand(
                    cabinet,
                    CommandMode.POSITION,
                    ArrayValue.from_rows(((0.0, 0.0),)),
                    environment_indices=(0,),
                    degree_of_freedom_indices=(0,),
                )
            )
        with pytest.raises(CommandError):
            world.apply_rigid_body_command(
                RigidBodyCommand(
                    box,
                    ArrayValue.from_rows(((1.0, 0.0, 0.0),)),
                    ArrayValue.from_rows(((0.0, 0.0, 0.0),)),
                    environment_indices=(0, 1),
                )
            )
        with pytest.raises(ValidationError):
            world.publish_debug(object())  # type: ignore[arg-type]

        before = world.read_rigid_body(box).positions_m.rows()[0]
        begin = scene_command(
            world,
            SceneCommandKind.DRAG_BEGIN,
            "cancel-begin",
            drag_id="cancel-drag",
            mode=SceneDragMode.KINEMATIC,
        )
        assert world.apply_scene_command(begin).status is SceneCommandStatus.APPLIED
        assert world.apply_scene_command(replace(begin, command_id="duplicate-begin")).error_code == "drag_exists"
        world.apply_scene_command(
            scene_command(
                world,
                SceneCommandKind.DRAG_UPDATE,
                "cancel-update",
                drag_id="cancel-drag",
                pose=Pose((8.0, 9.0, 10.0)),
            )
        )
        cancelled = world.apply_scene_command(
            scene_command(world, SceneCommandKind.DRAG_CANCEL, "cancel", drag_id="cancel-drag")
        )
        assert cancelled.status is SceneCommandStatus.APPLIED
        assert world.read_rigid_body(box).positions_m.rows()[0] == pytest.approx(before)
        assert (
            world.apply_scene_command(
                scene_command(world, SceneCommandKind.DRAG_END, "missing-drag", drag_id="missing")
            ).error_code
            == "drag_not_active"
        )
        assert world.apply_scene_command(replace(begin, command_id="stale", expected_generation=999)).error_code == (
            "stale_generation"
        )
        assert (
            world.apply_scene_command(
                replace(begin, command_id="wrong-kind", entity_path=EntityPath("/cabinet"), drag_id="wrong-kind")
            ).error_code
            == "unsupported_entity_kind"
        )
        assert (
            world.apply_scene_command(
                replace(begin, command_id="missing-target", environment_index=99, drag_id="missing-target")
            ).error_code
            == "target_not_found"
        )
        with pytest.raises(ValidationError):
            world.scene_delta(world.scene_snapshot().sequence + 1)
        with pytest.raises(ValidationError):
            world.apply_scene_command(object())  # type: ignore[arg-type]
    finally:
        session.close()


def test_all_soft_matter_operations_are_explicitly_unsupported() -> None:
    session = MuJoCoProvider().open()
    world = session.build(world_spec())
    try:
        box = world.resolve(EntityPath("/box"))
        for operation, argument in (
            (world.apply_deformable_command, object()),
            (world.read_deformable, box),
            (world.apply_particle_fluid_command, object()),
            (world.read_particle_fluid, box),
        ):
            with pytest.raises(UnsupportedCapabilityError):
                operation(argument)  # type: ignore[arg-type]
    finally:
        session.close()


def test_easy_api_native_common_development_flow() -> None:
    with Sim(
        backend="mujoco",
        world_id="mujoco-easy",
        num_envs=2,
        time_step_seconds=0.002,
        gravity_m_s2=(0.0, 0.0, 0.0),
    ) as sim:
        box = sim.add_box(
            "box",
            size_m=(0.2, 0.4, 0.6),
            mass_kg=2.0,
            color_rgba=(1.0, 0.0, 0.0, 1.0),
            position_m=(0.0, 0.0, 1.0),
        )
        cabinet = sim.add_articulation(
            "cabinet",
            joint_names=("door_hinge", "drawer_slide"),
            initial_positions=(0.1, -0.2),
            position_m=(2.0, 0.0, 0.3),
        )
        camera = sim.add_camera("camera", resolution=(24, 16), outputs=("rgb", "depth"))
        sim.start()
        box.apply_wrench((4.0, 0.0, 0.0), environments=(1,))
        cabinet.command((0.5,), joints=("door_hinge",), environments=(0,))
        sim.step(30)
        assert box.state.positions_m.rows()[1][0] > box.state.positions_m.rows()[0][0]
        assert cabinet.state.joint_positions.rows()[0][0] > 0.1
        assert camera.read("rgb").shape == (2, 16, 24, 3)
        assert camera.read("depth").shape == (2, 16, 24)
        scene_box = next(item for item in sim.scene_snapshot().entities if item.path == EntityPath("/box"))
        assert scene_box.visuals[0].dimensions_m == (0.2, 0.4, 0.6)
        assert scene_box.visuals[0].color_rgba == (1.0, 0.0, 0.0, 1.0)


def test_native_contact_force_uses_mujoco_writable_buffer() -> None:
    with Sim(backend="mujoco", world_id="mujoco-contact", time_step_seconds=0.002) as sim:
        box = sim.add_box("box", size_m=0.1, position_m=(0.0, 0.0, 0.05))
        sim.start()
        sim.step(20)
        contact = box.contact(force_threshold_n=1.0e-4)
        assert contact.in_contact.values == (True,)
        assert any(abs(float(value)) > 1.0e-4 for value in contact.net_normal_forces_n.values)


def test_easy_api_rejects_unsupported_soft_matter_before_world_creation() -> None:
    sim = Sim(provider=MuJoCoProvider(), world_id="mujoco-soft-rejection")
    sim.add_deformable(
        "cloth",
        rest_positions_m=((0, 0, 1), (1, 0, 1), (0, 1, 1)),
        surface_triangles=((0, 1, 2),),
    )
    with pytest.raises(CapabilityNegotiationError) as error:
        sim.start()
    negotiation = error.value.details["negotiation"]
    assert negotiation["accepted"] is False
    assert any(issue["id"] == "control.deformable.points@1" for issue in negotiation["required_issues"])
    sim.close()
