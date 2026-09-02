from __future__ import annotations

import math
import subprocess
import sys
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from unirobosim import (
    ArrayValue,
    ArticulationCommand,
    BoxGeometrySpec,
    CameraModality,
    CameraSpec,
    CapabilityId,
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
    ProviderSelectionError,
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
    WorldBuildError,
    WorldSpec,
)

import unirobosim_mujoco as package_module
import unirobosim_mujoco.provider as provider_module
import unirobosim_mujoco.world as world_module
from unirobosim_mujoco import DESCRIPTOR, MuJoCoAdapterConfig, MuJoCoProvider, __version__, create_provider
from unirobosim_mujoco.droid_acceptance import _EntryPoint
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
    assert __version__ == "0.9.4"
    assert DESCRIPTOR.version == __version__
    assert DESCRIPTOR.contract_version == "v0alpha5"
    provider = MuJoCoProvider()
    probe = provider.probe()
    assert probe.available and provider.descriptor.provider_id == "google-deepmind.mujoco"
    with pytest.raises(ValidationError):
        MuJoCoAdapterConfig(render_width=0)
    with pytest.raises(ValidationError):
        MuJoCoAdapterConfig(headless=1)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        MuJoCoAdapterConfig(enable_cameras=1)  # type: ignore[arg-type]
    for field, value in (
        ("position_stiffness", -1.0),
        ("position_damping", float("nan")),
        ("velocity_gain", -1.0),
    ):
        with pytest.raises(ValidationError):
            cast(Any, MuJoCoAdapterConfig)(**{field: value})
    configured = MuJoCoAdapterConfig(
        joint_position_stiffness=(("door_hinge", 250.0),),
        joint_position_damping=(("door_hinge", 15.0),),
    )
    assert configured.position_stiffness_for("door_hinge") == 250.0
    assert configured.position_stiffness_for("drawer_slide") == 100.0
    assert configured.position_damping_for("door_hinge") == 15.0
    assert configured.position_damping_for("drawer_slide") == 8.0
    with pytest.raises(ValidationError):
        MuJoCoAdapterConfig(joint_position_stiffness=(("door_hinge", 1.0), ("door_hinge", 2.0)))
    session = provider.open()
    world = session.build(world_spec())
    assert session.state is SessionState.READY and world.build_report.environment_count == 2
    world.close()
    assert session.state.value == "open"
    session.close()


def test_scene_command_descriptor_matches_implemented_semantics() -> None:
    pose_command = DESCRIPTOR.capabilities.get(CapabilityId("scene.command.pose@1"))
    drag_command = DESCRIPTOR.capabilities.get(CapabilityId("scene.command.drag@1"))

    assert pose_command is not None
    assert drag_command is not None
    assert drag_command.properties["entity_kinds"] == ("rigid_body",)
    assert drag_command.properties["modes"] == ("kinematic",)
    assert any("constraint drag" in limitation for limitation in drag_command.limitations)

    physics_only = create_provider(launch_profile="headless-physics").descriptor.capabilities
    assert physics_only.get(CapabilityId("scene.command.pose@1")) is pose_command
    assert physics_only.get(CapabilityId("scene.command.drag@1")) is drag_command


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_native_time_step_seconds", 0.0),
        ("max_native_time_step_seconds", -0.001),
        ("max_native_time_step_seconds", float("nan")),
        ("max_native_time_step_seconds", float("inf")),
        ("max_native_time_step_seconds", True),
        ("max_native_substeps_per_logical_step", 0),
        ("max_native_substeps_per_logical_step", -1),
        ("max_native_substeps_per_logical_step", 1.5),
        ("max_native_substeps_per_logical_step", True),
    ],
)
def test_native_step_config_rejects_invalid_limits(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        cast(Any, MuJoCoAdapterConfig)(**{field: value})


def test_native_step_schedule_handles_exact_nondivisible_authored_and_extreme_steps() -> None:
    defaults = MuJoCoAdapterConfig()
    assert defaults.max_native_time_step_seconds == pytest.approx(1.0 / 240.0)
    assert defaults.native_substeps_for(1.0 / 60.0, 1) == 4
    assert defaults.native_substeps_for(1.0 / 60.0, 8) == 8
    assert defaults.native_substeps_for(1.0e-9, 1) == 1

    nondivisible = MuJoCoAdapterConfig(max_native_time_step_seconds=0.004)
    resolved = nondivisible.native_substeps_for(0.01, 2)
    assert resolved == 3
    assert 0.01 / resolved <= nondivisible.max_native_time_step_seconds
    assert 0.01 / (resolved - 1) > nondivisible.max_native_time_step_seconds

    bounded = MuJoCoAdapterConfig(
        max_native_time_step_seconds=0.001,
        max_native_substeps_per_logical_step=4,
    )
    with pytest.raises(ValidationError, match="more native substeps") as caught:
        bounded.native_substeps_for(0.01, 1)
    assert caught.value.details["required_native_substeps"] == 10
    with pytest.raises(ValidationError, match="more native substeps"):
        bounded.native_substeps_for(1.0e308, 1)
    with pytest.raises(ValidationError, match="more native substeps") as authored_caught:
        bounded.native_substeps_for(0.001, 5)
    assert authored_caught.value.details["required_native_substeps"] == 5


def test_native_substeps_preserve_logical_tick_time_and_authored_minimum() -> None:
    config = MuJoCoAdapterConfig(max_native_time_step_seconds=0.004)
    session = MuJoCoProvider(config).open()
    spec = replace(
        world_spec(),
        environments=EnvironmentSpec(1),
        physics=PhysicsSpec(time_step_seconds=0.01, substeps=2, gravity_m_s2=(0.0, 0.0, 0.0)),
    )
    world = session.build(spec)
    try:
        assert world.logical_time_step_seconds == pytest.approx(0.01)
        assert world.native_substeps_per_logical_step == 3
        assert world.native_time_step_seconds == pytest.approx(0.01 / 3.0)
        assert float(world._models[0].opt.timestep) == pytest.approx(0.01 / 3.0)
        tick = world.step(2)
        assert tick.step_index == 2
        assert tick.sim_time_seconds == pytest.approx(0.02)
        assert float(world._data[0].time) == pytest.approx(0.02)
    finally:
        world.close()
        session.close()

    authored_session = MuJoCoProvider().open()
    authored_spec = replace(
        world_spec(),
        environments=EnvironmentSpec(1),
        physics=PhysicsSpec(time_step_seconds=0.002, substeps=3, gravity_m_s2=(0.0, 0.0, 0.0)),
    )
    authored_world = authored_session.build(authored_spec)
    try:
        assert authored_world.native_substeps_per_logical_step == 3
        assert authored_world.native_time_step_seconds == pytest.approx(0.002 / 3.0)
        tick = authored_world.step()
        assert tick.step_index == 1
        assert tick.sim_time_seconds == pytest.approx(0.002)
        assert float(authored_world._data[0].time) == pytest.approx(0.002)
    finally:
        authored_world.close()
        authored_session.close()


def test_native_substep_control_hold_reset_and_determinism() -> None:
    def run_once() -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
        session = MuJoCoProvider().open()
        spec = replace(
            world_spec(),
            environments=EnvironmentSpec(1),
            physics=PhysicsSpec(time_step_seconds=1.0 / 60.0, gravity_m_s2=(0.0, 0.0, 0.0)),
        )
        world = session.build(spec)
        try:
            assert world.native_substeps_per_logical_step == 4
            articulation = world.resolve(EntityPath("/cabinet"))
            initial = world.read_articulation(articulation).joint_positions.rows()[0]
            target = (0.45, -0.45)
            world.apply_articulation_command(
                ArticulationCommand(
                    articulation,
                    CommandMode.POSITION,
                    ArrayValue.from_rows((target,)),
                    target_units=("rad", "rad"),
                )
            )
            world.step()
            first = world.read_articulation(articulation).joint_positions.rows()[0]
            world.step(59)
            held = world.read_articulation(articulation).joint_positions.rows()[0]
            assert max(abs(value - expected) for value, expected in zip(held, target, strict=True)) < max(
                abs(value - expected) for value, expected in zip(first, target, strict=True)
            )
            world.reset()
            reset = world.read_articulation(articulation).joint_positions.rows()[0]
            world.step(10)
            reset_held = world.read_articulation(articulation).joint_positions.rows()[0]
            assert reset == pytest.approx(initial)
            assert reset_held == pytest.approx(initial, abs=1.0e-12)
            return first, held, reset_held
        finally:
            world.close()
            session.close()

    assert run_once() == run_once()


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("visible", (False, True)),
        ("headless", (True, True)),
        ("headless-physics", (True, False)),
    ],
)
def test_factory_maps_canonical_launch_profiles(profile: str, expected: tuple[bool, bool]) -> None:
    provider = create_provider(launch_profile=profile)

    assert (provider._config.headless, provider._config.enable_cameras) == expected
    camera_capabilities = (
        CapabilityId("sensor.camera@1"),
        CapabilityId("sensor.camera.rgb@1"),
        CapabilityId("sensor.camera.depth@1"),
    )
    assert all(
        (provider.descriptor.capabilities.get(capability) is not None) is expected[1]
        for capability in camera_capabilities
    )


def test_factory_zero_argument_compatibility_and_profile_override() -> None:
    default = create_provider()
    configured = create_provider(
        MuJoCoAdapterConfig(render_width=1280, headless=False, enable_cameras=False),
        launch_profile="headless",
    )

    assert (default._config.headless, default._config.enable_cameras) == (True, True)
    assert default.descriptor is DESCRIPTOR
    assert configured._config.render_width == 1280
    assert (configured._config.headless, configured._config.enable_cameras) == (True, True)


def test_acceptance_entry_point_accepts_fastsim_launch_profile() -> None:
    entry_point = _EntryPoint(visible_window=True)
    provider = entry_point.load()(launch_profile="headless-physics")

    assert entry_point.group == "unirobosim.backends"
    assert (provider._config.headless, provider._config.enable_cameras) == (True, False)
    assert provider._config.position_stiffness == 800.0


@pytest.mark.parametrize("config", [False, 0, "mujoco", object()])
def test_provider_and_factory_reject_wrong_config_types(config: object) -> None:
    with pytest.raises(ValidationError, match="config must be MuJoCoAdapterConfig"):
        MuJoCoProvider(config)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="config must be MuJoCoAdapterConfig"):
        create_provider(config, launch_profile="headless")  # type: ignore[arg-type]


def test_lazy_world_export_and_unknown_attribute() -> None:
    assert package_module.MuJoCoWorld is MuJoCoWorld
    with pytest.raises(AttributeError, match="has no attribute 'unknown'"):
        _ = package_module.unknown  # type: ignore[attr-defined]


def test_probe_and_open_fail_closed_for_missing_or_wrong_native_version(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = MuJoCoProvider()

    def missing_native(name: str) -> object:
        del name
        raise ImportError("missing native")

    monkeypatch.setattr(provider_module.importlib, "import_module", missing_native)
    probe = provider.probe()
    assert not probe.available and "missing native" in str(probe.reason)
    with pytest.raises(ProviderSelectionError, match="compatibility profile is unavailable"):
        provider.open()

    class WrongVersion:
        __version__ = "0.0"

    monkeypatch.setattr(provider_module.importlib, "import_module", lambda name: WrongVersion())
    probe = provider.probe()
    assert not probe.available and probe.reason == "expected MuJoCo 3.11.0, found 0.0"


def test_session_context_manager_closes_and_build_rejects_wrong_type() -> None:
    provider = MuJoCoProvider()
    with provider.open() as session:
        with pytest.raises(ValidationError, match="build requires a WorldSpec"):
            session.build(None)  # type: ignore[arg-type]
    assert session.state is SessionState.CLOSED


@pytest.mark.parametrize("profile", ["", "VISIBLE", " visible", "visible ", "auto", True, 1])
def test_factory_rejects_noncanonical_launch_profiles(profile: object) -> None:
    with pytest.raises(ValidationError) as caught:
        create_provider(launch_profile=profile)  # type: ignore[arg-type]

    assert caught.value.operation == "mujoco.launch_profile.resolve"
    assert len(str(caught.value)) < 512


def test_physics_only_profile_rejects_camera_world_before_native_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = create_provider(launch_profile="headless-physics")
    session = provider.open()
    monkeypatch.setattr(
        "unirobosim_mujoco.provider.snapshot_build_input",
        lambda *args, **kwargs: pytest.fail("asset or native allocation was reached"),
    )

    with pytest.raises(UnsupportedCapabilityError) as caught:
        session.build(world_spec(camera=True))

    assert caught.value.operation == "mujoco.session.build.preflight"
    assert caught.value.entity_path == "/camera"
    assert session.state is SessionState.OPEN
    session.close()


def test_factory_discovery_and_profile_mapping_are_native_import_safe() -> None:
    script = """
import importlib.abc
import sys

class RejectNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.', 1)[0] == 'mujoco':
            raise RuntimeError('native MuJoCo import attempted')
        return None

sys.meta_path.insert(0, RejectNative())
from unirobosim_mujoco import create_provider
provider = create_provider(launch_profile='headless-physics')
assert provider._config.enable_cameras is False
assert 'mujoco' not in sys.modules
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_joint_gain_lookups_are_precomputed_immutable_and_keep_the_scalar_fast_path() -> None:
    configured = MuJoCoAdapterConfig(
        joint_position_stiffness=(("door_hinge", 250.0),),
        joint_position_damping=(("door_hinge", 15.0),),
    )
    stiffness_lookup = configured._joint_position_stiffness_lookup
    damping_lookup = configured._joint_position_damping_lookup
    assert configured.position_stiffness_for("door_hinge") == 250.0
    assert configured.position_damping_for("door_hinge") == 15.0
    assert configured._joint_position_stiffness_lookup is stiffness_lookup
    assert configured._joint_position_damping_lookup is damping_lookup
    with pytest.raises(TypeError):
        cast(Any, stiffness_lookup)["door_hinge"] = 1.0
    with pytest.raises(TypeError):
        cast(Any, damping_lookup)["door_hinge"] = 1.0

    class FailOnLookup(Mapping[str, float]):
        def __getitem__(self, key: str) -> float:
            raise AssertionError(f"default scalar path consulted override lookup for {key}")

        def __iter__(self) -> Iterator[str]:
            return iter(())

        def __len__(self) -> int:
            return 0

    defaults = MuJoCoAdapterConfig()
    object.__setattr__(defaults, "_joint_position_stiffness_lookup", FailOnLookup())
    object.__setattr__(defaults, "_joint_position_damping_lookup", FailOnLookup())
    assert defaults.position_stiffness_for("any_joint") == 100.0
    assert defaults.position_damping_for("any_joint") == 8.0


def test_gui_mode_launches_syncs_and_closes_passive_viewer(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeViewer:
        def __init__(self) -> None:
            self.sync_count = 0
            self.closed = False

        def sync(self) -> None:
            self.sync_count += 1

        def is_running(self) -> bool:
            return not self.closed

        def close(self) -> None:
            self.closed = True

    viewers: list[FakeViewer] = []

    class FakeViewerModule:
        @staticmethod
        def launch_passive(*args: object, **kwargs: object) -> FakeViewer:
            del args, kwargs
            viewer = FakeViewer()
            viewers.append(viewer)
            return viewer

    world_importlib = cast(Any, world_module).importlib
    native_import_module = world_importlib.import_module

    def import_with_fake_viewer(name: str) -> Any:
        if name == "mujoco.viewer":
            return FakeViewerModule
        return native_import_module(name)

    monkeypatch.setattr(world_importlib, "import_module", import_with_fake_viewer)
    provider = MuJoCoProvider(MuJoCoAdapterConfig(headless=False))
    session = provider.open()
    single_environment = replace(
        world_spec(),
        environments=EnvironmentSpec(1),
        physics=PhysicsSpec(time_step_seconds=1.0 / 60.0, gravity_m_s2=(0.0, 0.0, 0.0)),
    )
    world = session.build(single_environment)
    assert world.native_substeps_per_logical_step == 4
    assert len(viewers) == 1 and viewers[0].sync_count == 1
    world.step()
    # Native integration may subdivide, but GUI publication remains once per
    # logical World step so camera/operator scheduling does not speed up.
    assert viewers[0].sync_count == 2
    world.close()
    assert viewers[0].closed
    session.close()

    session = provider.open()
    with pytest.raises(WorldBuildError):
        session.build(world_spec())
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
    spec = replace(
        world_spec(camera=True),
        physics=PhysicsSpec(time_step_seconds=1.0 / 60.0, gravity_m_s2=(0.0, 0.0, 0.0)),
    )
    world = session.build(spec)
    try:
        assert world.native_substeps_per_logical_step == 4
        sample = world.read_sensor(world.resolve(EntityPath("/camera")))
        rgb = sample.channel(CameraModality.RGB)
        assert rgb.shape == (2, 24, 32, 3)
        assert rgb.dtype == "uint8" and rgb.device == "cpu"
        if callable(getattr(ArrayValue, "from_uint8_bytes", None)):
            assert rgb.is_packed is True
            assert type(rgb.to_bytes()) is bytes
            assert len(rgb.to_bytes()) == 2 * 24 * 32 * 3
        assert sample.channel(CameraModality.DEPTH).shape == (2, 24, 32)
        camera = world._entities[EntityPath("/camera")].camera
        assert camera is not None
        expected_vertical_fov = math.degrees(
            2.0
            * math.atan(
                math.tan(math.radians(camera.horizontal_fov_degrees) / 2.0) / (camera.width_px / camera.height_px)
            )
        )
        assert float(world._models[0].cam_fovy[0]) == pytest.approx(expected_vertical_fov)
        tick = world.step()
        assert tick.step_index == 1
        assert tick.sim_time_seconds == pytest.approx(1.0 / 60.0)
        assert world.read_sensor(world.resolve(EntityPath("/camera"))).tick == tick
        world.reset()
        reset_sample = world.read_sensor(world.resolve(EntityPath("/camera")))
        assert reset_sample.tick == tick
        assert world.native_substeps_per_logical_step == 4
        assert reset_sample.channel(CameraModality.RGB).shape == (2, 24, 32, 3)
    finally:
        session.close()


def test_1280x720_rgb_payload_uses_compact_immutable_storage() -> None:
    shape = (1, 720, 1280, 3)
    payload = bytes(720 * 1280 * 3)

    value = world_module._uint8_array(shape, payload)

    assert value.shape == shape and value.dtype == "uint8" and value.device == "cpu"
    if callable(getattr(ArrayValue, "from_uint8_bytes", None)):
        assert value.is_packed is True
        assert value.to_bytes() is payload


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
