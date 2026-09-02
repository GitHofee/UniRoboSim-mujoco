from __future__ import annotations

from test_mujoco_adapter import world_spec
from unirobosim import ArrayValue, ArticulationCommand, CheckpointWorld, CommandMode, EntityPath

from unirobosim_mujoco import MuJoCoProvider


def test_mujoco_checkpoint_restores_state_and_control_without_rewinding_clock() -> None:
    session = MuJoCoProvider().open()
    try:
        world = session.build(world_spec())
        assert isinstance(world, CheckpointWorld)
        robot = world.resolve(EntityPath("/cabinet"))
        world.apply_articulation_command(
            ArticulationCommand(
                robot,
                CommandMode.POSITION,
                ArrayValue.from_rows(((0.4, 0.3), (0.2, 0.1))),
            )
        )
        world.step(30)
        expected = world.read_articulation(robot)
        checkpoint = world.create_checkpoint()

        world.apply_articulation_command(
            ArticulationCommand(
                robot,
                CommandMode.POSITION,
                ArrayValue.from_rows(((-0.8, -0.7), (-0.6, -0.5))),
            )
        )
        world.step(20)
        live_tick = world.tick
        assert world.read_articulation(robot).joint_positions != expected.joint_positions

        result = world.restore_checkpoint(checkpoint)

        restored = world.read_articulation(robot)
        assert result.tick == live_tick
        assert world.tick == live_tick
        assert restored.joint_positions == expected.joint_positions
        assert restored.joint_velocities == expected.joint_velocities

        world.step()
        advanced = world.read_articulation(robot)
        assert advanced.joint_positions.rows()[0][0] > restored.joint_positions.rows()[0][0]
    finally:
        session.close()
