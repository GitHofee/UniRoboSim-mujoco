# UniRoboSim MuJoCo Adapter

[English](README.md) | [简体中文](README.zh-CN.md)

`unirobosim-mujoco` 是 UniRoboSim `0.7.x` 的 MuJoCo 3.11 后端。它通过标准 `unirobosim.backends` entry point 发现，同一套 EasyAPI 业务只需设置 `backend="mujoco"` 即可切换。

## 兼容与安装

- Python `>=3.12,<3.13`
- UniRoboSim `>=0.7.0,<0.8`
- MuJoCo `3.11.0`
- NumPy `>=2.2,<3`
- Runtime contract `v0alpha4`

```bash
conda create -n unirobosim-mujoco python=3.12 pip -y
conda activate unirobosim-mujoco
git clone https://github.com/GitHofee/UniRoboSim.git
git clone https://github.com/GitHofee/UniRoboSim-mujoco.git
python -m pip install ./UniRoboSim ./UniRoboSim-mujoco
```

GitHub 仓库、Python distribution、import 包、backend ID 和仿真器名称均使用标准 `mujoco` 拼写。

## 使用

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

定制 Adapter 增益时注入 Provider：

```python
from unirobosim import Sim
from unirobosim_mujoco import MuJoCoAdapterConfig, MuJoCoProvider

provider = MuJoCoProvider(MuJoCoAdapterConfig(position_stiffness=150.0, max_motor_effort=80.0))
sim = Sim(provider=provider)
```

## 已实现能力

- 原生 MJCF 与 URDF 铰接体；
- 原生 MJCF 刚体和可移植 Box；
- 刚体状态、持续 wrench、二值/净法向接触；
- 关节位置、速度和力矩控制；
- RGB/深度相机；
- 多环境世界；
- 后端中立 Debug 存储和 native scene lowering；
- 场景快照/增量、位姿写入和刚体运动学拖拽；
- 通过 Studio 使用 Unified Scene 与 Native Stream。

## 资产与限制

MuJoCo 不直接读取 USD。安装 `unirobosim-usd-converter` 可将刚体 USD 编译为带来源追踪的 MJCF 包，也可以通过 `AssetBundle` 提供验证过的 MJCF/URDF 变体。转换器只承诺刚体，不承诺自动转换铰接 USD。

本 Adapter 不声明 UniRoboSim 表面/体积柔性体或粒子流体能力；相关请求会在创建世界前的能力协商阶段明确失败。

## 验证

```bash
python -m pip install -e '.[dev]'
ruff format --check src tests
ruff check src tests
mypy src
coverage run -m pytest
coverage report
```

0.7.0 原生套件通过 15 项测试，分支覆盖率高于发布阈值；构建出的 wheel 也已在全新环境中通过 EasyAPI 相机、状态与场景 smoke test。

可移植 API 文档位于 [UniRoboSim Core](https://github.com/GitHofee/UniRoboSim.git)。
