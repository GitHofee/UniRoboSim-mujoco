from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest
from unirobosim import (
    ArrayValue,
    ArticulationCommand,
    CommandMode,
    EntityKind,
    EntityPath,
    EntitySpec,
    EnvironmentSpec,
    FrozenMap,
    PhysicsSpec,
    Pose,
    ValidationError,
    WorldSpec,
)

from unirobosim_mujoco import MuJoCoProvider
from unirobosim_mujoco.articulation_drive import (
    ARTICULATION_DRIVE_PROFILE_KEY,
    ARTICULATION_DRIVE_PROFILE_SCHEMA,
    _validated_joint_settings,
)


def _profile(joints: object, **updates: object) -> FrozenMap:
    value: dict[str, object] = {
        "schema": ARTICULATION_DRIVE_PROFILE_SCHEMA,
        "backend": "mujoco",
        "joints": joints,
    }
    value.update(updates)
    return FrozenMap({ARTICULATION_DRIVE_PROFILE_KEY: value})


def _world_spec(*, metadata: FrozenMap | None = None) -> WorldSpec:
    return WorldSpec(
        "mujoco-drive-profile",
        (
            EntitySpec(EntityPath("/box"), EntityKind.RIGID_BODY, pose=Pose((0.0, 0.0, 1.0))),
            EntitySpec(
                EntityPath("/cabinet"),
                EntityKind.ARTICULATION,
                joint_names=("door_hinge", "drawer_slide"),
                initial_joint_positions=(0.0, 0.0),
                metadata=FrozenMap() if metadata is None else metadata,
            ),
        ),
        environments=EnvironmentSpec(1),
        physics=PhysicsSpec(time_step_seconds=1.0 / 60.0, gravity_m_s2=(0.0, 0.0, 0.0)),
    )


@pytest.mark.parametrize(
    "metadata",
    [
        FrozenMap({ARTICULATION_DRIVE_PROFILE_KEY: []}),
        FrozenMap(
            {
                ARTICULATION_DRIVE_PROFILE_KEY: {
                    "schema": "unirobosim-articulation-drive-profile/2",
                    "backend": "mujoco",
                    "joints": {},
                }
            }
        ),
        _profile({}),
        _profile({}, backend="pybullet"),
        _profile({}, extra=True),
        _profile([]),
        _profile({"unknown_joint": {"position_stiffness": 1.0, "position_damping": 1.0}}),
        _profile({"door_hinge": {"position_stiffness": 1.0}}),
        _profile(
            {
                "door_hinge": {
                    "position_stiffness": 1.0,
                    "position_damping": 1.0,
                    "position_gain": 0.1,
                }
            }
        ),
        _profile({"door_hinge": {"position_stiffness": True, "position_damping": 1.0}}),
        _profile({"door_hinge": {"position_stiffness": 1.0, "position_damping": -1.0}}),
    ],
)
def test_mujoco_drive_profile_invalid_matrix_fails_before_world_allocation(metadata: FrozenMap) -> None:
    session = MuJoCoProvider().open()
    try:
        with pytest.raises(ValidationError) as caught:
            session.build(_world_spec(metadata=metadata))
        assert caught.value.operation == "mujoco.session.build.articulation_drive_profile"
        assert session._generation == 0
        assert session._world is None
    finally:
        session.close()


def test_mujoco_drive_profile_revalidates_nonfinite_boundary_value() -> None:
    spec = _world_spec()
    entity = spec.entities[1]
    raw = {
        "schema": ARTICULATION_DRIVE_PROFILE_SCHEMA,
        "backend": "mujoco",
        "joints": {
            "door_hinge": {
                "position_stiffness": float("nan"),
                "position_damping": 1.0,
            }
        },
    }
    with pytest.raises(ValidationError, match="finite and non-negative"):
        _validated_joint_settings(
            raw,
            spec=spec,
            entity=entity,
            provider_id="google-deepmind.mujoco",
        )


def test_mujoco_drive_profile_is_rejected_on_non_articulation() -> None:
    spec = _world_spec()
    rigid = replace(spec.entities[0], metadata=_profile({}))
    session = MuJoCoProvider().open()
    try:
        with pytest.raises(ValidationError, match="only valid on articulation"):
            session.build(replace(spec, entities=(rigid, spec.entities[1])))
        assert session._generation == 0
    finally:
        session.close()


def test_mujoco_omitted_drive_profile_preserves_exact_default_control_path() -> None:
    session = MuJoCoProvider().open()
    world = session.build(_world_spec())
    try:
        assert world._articulation_drive_profiles is None
        assert world._apply_controls_for_step.__func__ is world._apply_controls.__func__
    finally:
        world.close()
        session.close()


def test_mujoco_drive_profiles_are_compiled_per_entity_path_with_same_joint_names() -> None:
    joint_names = ("shared_joint",)
    first = EntitySpec(
        EntityPath("/first"),
        EntityKind.ARTICULATION,
        pose=Pose((-0.5, 0.0, 0.0)),
        joint_names=joint_names,
        initial_joint_positions=(0.0,),
        metadata=_profile({"shared_joint": {"position_stiffness": 10.0, "position_damping": 0.0}}),
    )
    second = EntitySpec(
        EntityPath("/second"),
        EntityKind.ARTICULATION,
        pose=Pose((0.5, 0.0, 0.0)),
        joint_names=joint_names,
        initial_joint_positions=(0.0,),
        metadata=_profile({"shared_joint": {"position_stiffness": 40.0, "position_damping": 0.0}}),
    )
    spec = WorldSpec(
        "mujoco-two-drive-profiles",
        (first, second),
        environments=EnvironmentSpec(1),
        physics=PhysicsSpec(time_step_seconds=1.0 / 60.0, gravity_m_s2=(0.0, 0.0, 0.0)),
    )
    session = MuJoCoProvider().open()
    world = session.build(spec)
    try:
        profiles = world._articulation_drive_profiles
        assert profiles is not None
        assert profiles[first.path].position_stiffness == (10.0,)
        assert profiles[second.path].position_stiffness == (40.0,)
        with pytest.raises(TypeError):
            cast(Any, profiles)[first.path] = profiles[second.path]

        for entity in (first, second):
            world.apply_articulation_command(
                ArticulationCommand(
                    world.resolve(entity.path),
                    CommandMode.POSITION,
                    ArrayValue.from_rows(((1.0,),)),
                    target_units=("rad",),
                )
            )
        world._apply_controls_for_step(0)
        data = world._data[0]
        first_dof = world._native[first.path].joint_dof_addresses[0]
        second_dof = world._native[second.path].joint_dof_addresses[0]
        assert float(data.qfrc_applied[first_dof]) == pytest.approx(10.0)
        assert float(data.qfrc_applied[second_dof]) == pytest.approx(40.0)
    finally:
        world.close()
        session.close()
