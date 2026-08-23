# UniRoboSim MuJoCo adapter

[English](README.md) | [简体中文](README.zh-CN.md)

`unirobosim-mujoco` is the MuJoCo 3.11 backend for UniRoboSim `0.9.x`. It is selected through the standard `unirobosim.backends` entry point, so the same EasyAPI application can switch to MuJoCo with `backend="mujoco"`.

## Compatibility and installation

- Python `>=3.12,<3.13`
- UniRoboSim `>=0.9,<0.10`
- MuJoCo `3.11.0`
- NumPy `>=2.2,<3`
- Runtime contract `v0alpha5`

```bash
conda create -n unirobosim-mujoco python=3.12 pip -y
conda activate unirobosim-mujoco
git clone https://github.com/GitHofee/UniRoboSim.git
git clone https://github.com/GitHofee/UniRoboSim-mujoco.git
python -m pip install ./UniRoboSim ./UniRoboSim-mujoco
```

The GitHub repository, Python distribution, import package, backend identifier, and simulator name all use the standard `mujoco` spelling.

## Usage

```python
from unirobosim import Sim

with Sim(backend="mujoco", num_envs=2, time_step_seconds=0.002) as sim:
    box = sim.add_box(
        "red_box",
        size_m=(0.2, 0.2, 0.2),
        mass_kg=1.0,
        color_rgba=(1.0, 0.0, 0.0, 1.0),
        position_m=(0.0, 0.0, 1.0),
    )
    cabinet = sim.add_articulation(
        "cabinet",
        joint_names=("door_hinge", "drawer_slide"),
        initial_positions=(0.0, 0.0),
    )
    camera = sim.add_camera("camera", resolution=(640, 360), outputs=("rgb", "depth"))
    sim.start()
    cabinet.command((0.5,), joints=("door_hinge",), mode="position")
    box.apply_wrench((2.0, 0.0, 0.0), environments=(1,))
    sim.step(30)
    print(box.state.positions_m.rows())
    print(camera.read("rgb").shape)
```

Custom adapter gains can be supplied by injecting a provider:

```python
from unirobosim import Sim
from unirobosim_mujoco import MuJoCoAdapterConfig, MuJoCoProvider

provider = MuJoCoProvider(MuJoCoAdapterConfig(position_stiffness=150.0, max_motor_effort=80.0))
sim = Sim(provider=provider)
```

## Implemented capabilities

- native MJCF and URDF articulation loading;
- native MJCF rigid assets and portable boxes;
- rigid state, persistent wrench, binary/net-normal contact;
- articulation position, velocity, and effort commands;
- RGB/depth cameras;
- multi-environment worlds;
- backend-neutral debug storage/native-scene lowering;
- scene snapshots/deltas, pose writes, and kinematic rigid drag;
- browser Unified Scene and Native Stream support through Studio.

## Assets and limitations

MuJoCo does not read USD directly. Install `unirobosim-usd-converter` to compile rigid USD into a provenance-tracked MJCF package, or use an `AssetBundle` with a verified MJCF/URDF variant. The converter is rigid-only and does not promise automatic articulated USD conversion.

This adapter does not advertise UniRoboSim surface/volume deformable or particle-fluid capabilities. Requests for them fail during capability negotiation before world creation.

## Verification

```bash
python -m pip install -e '.[dev]'
ruff format --check src tests
ruff check src tests
mypy src
coverage run -m pytest
coverage report
```

The 0.9.0 native suite passes 17 tests, including the DROID planning and same-tick command slice.

Portable API documentation is maintained in [UniRoboSim Core](https://github.com/GitHofee/UniRoboSim.git).
