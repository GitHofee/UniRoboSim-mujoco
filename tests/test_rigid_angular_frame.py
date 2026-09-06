"""Native public wrench/readback world-frame angular velocity controls."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
from unirobosim import (
    ArrayValue,
    BoxGeometrySpec,
    EntityKind,
    EntityPath,
    EntitySpec,
    EnvironmentSpec,
    PhysicsSpec,
    Pose,
    RigidBodyCommand,
    WorldSpec,
)

from unirobosim_mujoco import MuJoCoProvider


@pytest.mark.parametrize(
    "angle, torque", ((0.0, (0, 0, 1)), (math.pi / 4, (1, 0, 0)), (math.pi / 4, (0, 0, 1)), (-math.pi / 4, (0, 0, -1)))
)
@pytest.mark.parametrize("phase", ("motion", "reset", "checkpoint"))
@pytest.mark.parametrize("consumer", ("rigid", "snapshot"))
@pytest.mark.parametrize("translation", (0.0, 1e3, 1e5))
def test_public_rigid_angular_velocity_uses_world_axes(angle, torque, phase, consumer, translation):
    path = EntityPath("/box")
    spec = WorldSpec(
        "angular-frame",
        (
            EntitySpec(
                path,
                EntityKind.RIGID_BODY,
                box=BoxGeometrySpec((0.2, 0.2, 0.2)),
                pose=Pose((translation, 0, 10), (math.sin(angle / 2), 0, 0, math.cos(angle / 2))),
            ),
        ),
        environments=EnvironmentSpec(2),
        physics=PhysicsSpec(time_step_seconds=0.001, gravity_m_s2=(0, 0, 0)),
    )
    with MuJoCoProvider().open() as session:
        world = session.build(spec)
        handle = world.resolve(path)
        world.apply_rigid_body_command(
            RigidBodyCommand(
                handle, ArrayValue.from_rows(((0, 0, 0),)), ArrayValue.from_rows((torque,)), environment_indices=(0,)
            )
        )
        world.step(10)
        assert np.linalg.norm(world._data[0].qvel[3:6]) > 0.1
        if phase == "reset":
            world.reset((0,))
        elif phase == "checkpoint":
            checkpoint = world.create_checkpoint()
            saved = world._data[0].qpos.copy()
            world.step(5)
            assert not np.allclose(world._data[0].qpos, saved, atol=1e-8, rtol=0)
            world.restore_checkpoint(checkpoint)
            assert world._data[0].qpos == pytest.approx(saved, abs=1e-12)
        if consumer == "rigid":
            angular = world.read_rigid_body(handle).angular_velocities_rad_s.rows()
        else:
            angular = tuple(entity.angular_velocity_rad_s for entity in world.scene_snapshot().entities)
        # Independent native data copy: forward computes an oracle at exactly
        # current qpos/qvel without changing the adapter's state or hiding caches.
        model = world._models[0]
        data = mujoco.MjData(model)
        data.qpos[:] = world._data[0].qpos
        data.qvel[:] = world._data[0].qvel
        mujoco.mj_forward(model, data)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, world._native[path].body_id, velocity, 0)
        if phase != "reset":
            assert np.linalg.norm(velocity[:3]) > 0.1
        assert angular[1] == pytest.approx((0, 0, 0), abs=1e-12)
        assert angular[0] == pytest.approx(tuple(velocity[:3]), abs=1e-10)
