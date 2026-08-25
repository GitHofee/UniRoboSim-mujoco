"""DROID equivalence hook consumed by the frozen FastSim acceptance runner."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import mujoco  # type: ignore[import-untyped]
import numpy as np
from unirobosim import (
    WORLD_SCHEMA_VERSION,
    BoxGeometrySpec,
    CameraModality,
    CameraSpec,
    CapabilityId,
    CapabilityRequirement,
    EntityKind,
    EntityPath,
    EntitySpec,
    FrozenMap,
    Pose,
)

from . import MuJoCoAdapterConfig, create_provider

_ASSET = Path("/home/ubuntu/projects/gen_data/data/robots/droid/droid_mujoco.urdf")
_ARM = tuple(f"panda_joint{index}" for index in range(1, 8))
_GRIPPER = (
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
)
_ROBOT_PATH = EntityPath("/robots/droid")
_CAMERA_PATH = EntityPath("/acceptance-camera")

_POSITION_STIFFNESS = 800.0
_POSITION_DAMPING = 24.0
_WINDOW_LINE = re.compile(r'^\s*(0x[0-9a-fA-F]+)\s+(?:"([^"]*)"|\(has no name\))')


def _x11_windows(display: str) -> dict[str, str]:
    completed = subprocess.run(
        ("xwininfo", "-display", display, "-root", "-tree"),
        check=False,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"xwininfo failed for DISPLAY={display}: {completed.stderr.strip()}")
    windows: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        match = _WINDOW_LINE.match(line)
        if match is not None:
            windows[match.group(1).lower()] = match.group(2) or ""
    return windows


def _visible_window_detail(display: str, window_id: str, title: str) -> tuple[int, str] | None:
    completed = subprocess.run(
        ("xwininfo", "-display", display, "-id", window_id),
        check=False,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    if completed.returncode != 0 or "Map State: IsViewable" not in completed.stdout:
        return None
    width = re.search(r"^\s*Width:\s*(\d+)\s*$", completed.stdout, re.MULTILINE)
    height = re.search(r"^\s*Height:\s*(\d+)\s*$", completed.stdout, re.MULTILINE)
    if width is None or height is None:
        return None
    area = int(width.group(1)) * int(height.group(1))
    if area < 4096:
        return None
    return area, title


def _create_acceptance_provider(
    *,
    visible_window: bool = False,
    launch_profile: str | None = None,
) -> Any:
    return create_provider(
        MuJoCoAdapterConfig(
            headless=not visible_window,
            position_stiffness=_POSITION_STIFFNESS,
            position_damping=_POSITION_DAMPING,
            joint_position_stiffness=tuple((joint, 100.0) for joint in _GRIPPER),
            joint_position_damping=(
                tuple((joint, 36.0) for joint in _ARM[:4]) + tuple((joint, 8.0) for joint in _GRIPPER)
            ),
        ),
        launch_profile=launch_profile,
    )


@dataclass(frozen=True, slots=True)
class _Distribution:
    name: str = "unirobosim-mujoco"
    version: str = "0.9.1"


@dataclass(frozen=True, slots=True)
class _EntryPoint:
    name: str = "mujoco"
    value: str = "unirobosim_mujoco:create_provider"
    group: str = "unirobosim.backends"
    dist: _Distribution = _Distribution()
    visible_window: bool = False

    def load(self) -> Any:
        def factory(*, launch_profile: str | None = None) -> Any:
            return _create_acceptance_provider(
                visible_window=self.visible_window,
                launch_profile=launch_profile,
            )

        return factory


def _normalize(values: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(value * value for value in values))
    return tuple(value / length for value in values)  # type: ignore[return-value]


def _cross(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _tuple3(values: Any) -> tuple[float, float, float]:
    converted = tuple(float(value) for value in values)
    if len(converted) != 3:
        raise ValueError("expected exactly three numeric values")
    return converted


def _tuple4(values: Any) -> tuple[float, float, float, float]:
    converted = tuple(float(value) for value in values)
    if len(converted) != 4:
        raise ValueError("expected exactly four numeric values")
    return converted


def _matrix_quaternion(matrix: tuple[tuple[float, float, float], ...]) -> tuple[float, float, float, float]:
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return ((m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale, 0.25 * scale)
    if m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        return (0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale, (m21 - m12) / scale)
    if m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        return ((m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale, (m02 - m20) / scale)
    scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
    return ((m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale, (m10 - m01) / scale)


def _camera_pose(camera: Any) -> Pose:
    eye = _tuple3(camera["eye_m"])
    look_at = _tuple3(camera["look_at_m"])
    authored_up = _tuple3(camera["up"])
    forward = _normalize(_tuple3(look_at[index] - eye[index] for index in range(3)))
    right = _normalize(_cross(forward, authored_up))
    up = _normalize(_cross(right, forward))
    backward = tuple(-value for value in forward)
    rotation = tuple((right[row], up[row], backward[row]) for row in range(3))
    return Pose(eye, _matrix_quaternion(rotation))


def _scenario(entity: dict[str, object]) -> Any:
    from fastsim.plan import ScenarioPlan, content_digest, freeze  # type: ignore[import-not-found]

    scene = {"robots": {"droid": entity}}
    return ScenarioPlan(
        schema="fastsim-scenario-plan/1",
        source=cast(Any, freeze({"authoring_path": "droid-equivalence", "kind": "inline"})),
        scene=cast(Any, freeze(scene)),
        behavior=None,
        evaluation=None,
        digest=content_digest(
            {
                "behavior": None,
                "evaluation": None,
                "scene": scene,
                "schema": "fastsim-scenario-plan/1",
            }
        ),
    )


def _execution_plan(spec: Any, *, visible_window: bool = False) -> Any:
    from fastsim.plan import (
        ExecutionPlan,
        ResolvedComponent,
        ResourceRecord,
        content_digest,
        freeze,
    )

    asset = _ASSET.resolve(strict=True)
    asset_sha256 = hashlib.sha256(asset.read_bytes()).hexdigest()
    joints = _ARM + _GRIPPER
    initial = tuple(float(value) for value in spec["robot"]["initial_joint_position"]) + tuple(
        float(value) for value in spec["robot"]["gripper"]["open_position"]
    )
    semantics = {
        "groups": {"arm": list(_ARM), "gripper": list(_GRIPPER)},
        "joint_units": {joint: "rad" for joint in joints},
        "joints": list(joints),
    }
    resource = ResourceRecord(
        name="model",
        role="simulation",
        format="model/vnd.urdf+xml",
        requested_uri=str(asset),
        resolved_uri=asset.as_uri(),
        local_path=str(asset),
        sha256=asset_sha256,
        cache_hit=True,
    )
    component_digest = hashlib.sha256(f"droid:mujoco:{asset_sha256}".encode()).hexdigest()
    component = ResolvedComponent(
        requested="robot://droid",
        identity="robot://droid",
        version="0.9.0",
        kind="robot",
        manifest_sha256=component_digest,
        manifest_source=str(asset),
        variant="mujoco",
        capabilities=(),
        requires=(),
        semantics=cast(Any, freeze(semantics)),
        defaults=cast(Any, freeze({})),
        config_schema=cast(Any, freeze({})),
        resources=(resource,),
    )
    entity = {
        "capabilities": [],
        "component": {
            "identity": component.identity,
            "manifest_sha256": component.manifest_sha256,
            "variant": component.variant,
            "version": component.version,
        },
        "enabled": True,
        "id": "robots.droid",
        "initial_state": {"joints": dict(zip(joints, initial, strict=True))},
        "kind": "robot",
        "mount": None,
        "params": {},
        "pose": spec["robot"]["base_pose_world"]
        | {
            "xyz_m": spec["robot"]["base_pose_world"]["position_m"],
            "quat_xyzw": spec["robot"]["base_pose_world"]["quaternion_xyzw"],
        },
        "resources": [resource.to_scenario_dict()],
        "scale": [1.0, 1.0, 1.0],
        "semantics": semantics,
    }
    entity["pose"] = {
        "xyz_m": list(spec["robot"]["base_pose_world"]["position_m"]),
        "quat_xyzw": list(spec["robot"]["base_pose_world"]["quaternion_xyzw"]),
    }
    runtime: dict[str, object] = {
        "control_hz": float(spec["simulation"]["physics_hz"]),
        "launch_profile": "visible" if visible_window else "headless",
        "physics_hz": float(spec["simulation"]["physics_hz"]),
        "rate_policy": "exact",
        "seed": int(spec["simulation"]["seed"]),
        "sensor_hz": {},
    }
    plan = ExecutionPlan(
        schema="fastsim-execution-plan/2",
        name="droid-equivalence-mujoco",
        backend="mujoco",
        runtime=cast(Any, freeze(runtime)),
        scenario=_scenario(entity),
        entities=cast(Any, freeze({"robots.droid": entity})),
        control=cast(Any, freeze({"default_robot": "droid", "precedence": []})),
        plugins=cast(Any, freeze({})),
        components=(component,),
        provenance=cast(Any, freeze({})),
        authoring_sha256=hashlib.sha256(b"droid-equivalence-authoring/1").hexdigest(),
        effective_sha256=hashlib.sha256(b"droid-equivalence-effective/mujoco/1").hexdigest(),
        digest="",
    )
    return replace(plan, digest=content_digest(plan.to_dict(include_digest=False, portable=True)))


def _acceptance_projection(plan: Any, spec: Any, *, visible_window: bool = False) -> Any:
    from fastsim.integrations.unirobosim.projection import project_execution_plan  # type: ignore[import-not-found]

    projection = project_execution_plan(plan)
    if visible_window:
        projection = replace(
            projection,
            backend=replace(projection.backend, runtime_connection_mode="gui"),
        )
    articulation = replace(projection.articulations[0], entity_id="droid")
    projection = replace(
        projection,
        articulations=(articulation,),
        default_entity_id="droid",
    )
    camera_spec = spec["camera"]
    cube_spec = spec["scene_probe"]
    requirements = projection.world_spec.requirements + (CapabilityRequirement(CapabilityId("planning.scene@2")),)
    cube = EntitySpec(
        EntityPath("/red-cube"),
        EntityKind.RIGID_BODY,
        pose=Pose(
            _tuple3(cube_spec["pose_world"]["position_m"]),
            _tuple4(cube_spec["pose_world"]["quaternion_xyzw"]),
        ),
        box=BoxGeometrySpec(
            dimensions_m=_tuple3(cube_spec["size_m"]),
            color_rgba=_tuple4(cube_spec["rgba"]),
        ),
        metadata=FrozenMap({"planning_entity_kind": "rigid_object"}),
    )
    camera = EntitySpec(
        _CAMERA_PATH,
        EntityKind.CAMERA_SENSOR,
        pose=_camera_pose(camera_spec),
        camera=CameraSpec(
            int(camera_spec["width"]),
            int(camera_spec["height"]),
            modalities=(CameraModality.RGB,),
            horizontal_fov_degrees=float(camera_spec["horizontal_fov_deg"]),
            near_plane_m=float(camera_spec["near_m"]),
            far_plane_m=float(camera_spec["far_m"]),
        ),
    )
    gravity = _tuple3(spec["simulation"]["gravity_m_s2"])
    physics = replace(projection.world_spec.physics, gravity_m_s2=gravity)
    world_spec = replace(
        projection.world_spec,
        entities=projection.world_spec.entities + (cube, camera),
        requirements=requirements,
        physics=physics,
        schema_version=WORLD_SCHEMA_VERSION,
    )
    return replace(projection, world_spec=world_spec)


class _CaptureParticipant:
    def __init__(self, adapter: Any, *, sample_stride: int, camera: Any) -> None:
        self._adapter = adapter
        self._sample_stride = sample_stride
        self._camera = camera
        self._lock = threading.Lock()
        self._samples: dict[int, MappingProxyType[str, object]] = {}
        self._frame_id: str | None = None
        self._camera_calibration: MappingProxyType[str, object] | None = None
        self._physics_diagnostics: MappingProxyType[str, object] | None = None
        self._render_tuned = False
        self._closed = False

    def participant_spec(self) -> Any:
        from fastsim.runtime import AuthorityParticipantSpec  # type: ignore[import-not-found]

        return AuthorityParticipantSpec("fastsim.droid_acceptance.capture", 300, self)

    def prepare(self, context: Any) -> None:
        del context

    def generation_started(self, context: Any) -> None:
        self._capture(context.runtime, context.world_state)

    def before_physics(self, context: Any) -> None:
        del context

    def after_commit(self, context: Any) -> None:
        if context.runtime.tick % self._sample_stride == 0:
            self._capture(context.runtime, context.world_state)
        return None

    def generation_ending(self, context: Any) -> None:
        del context

    def close(self, context: Any) -> None:
        del context
        with self._lock:
            self._closed = True

    def sample(self, tick: int) -> MappingProxyType[str, object]:
        with self._lock:
            try:
                result = self._samples.pop(tick)
            except KeyError:
                raise RuntimeError(f"authority sample for tick {tick} is unavailable") from None
            for stale in tuple(value for value in self._samples if value < tick):
                del self._samples[stale]
            return result

    def camera_calibration(self) -> MappingProxyType[str, object]:
        with self._lock:
            if self._camera_calibration is None:
                raise RuntimeError("MuJoCo effective camera calibration is unavailable")
            return self._camera_calibration

    def physics_diagnostics(self) -> MappingProxyType[str, object]:
        with self._lock:
            if self._physics_diagnostics is None:
                raise RuntimeError("MuJoCo effective physics diagnostics are unavailable")
            return self._physics_diagnostics

    def discard(self) -> None:
        with self._lock:
            self._samples.clear()
            self._closed = True

    def _capture(self, runtime: Any, world_state: Any) -> None:
        world = self._adapter._world
        if world is None:
            raise RuntimeError("MuJoCo acceptance world is unavailable")
        value = np.asarray(world_state.values["droid.joint_position"].value, dtype=np.float64)
        catalog = world.planning_scene_catalog(0)
        if self._frame_id is None:
            end_effector_link = next(link for link in catalog.links if link.authored_name == "gripper_center")
            self._frame_id = next(
                frame.frame_id for frame in catalog.frames if frame.owner_link_id == end_effector_link.link_id
            )
        planning = world.planning_scene_state(0)
        frame = next(item for item in planning.frames if item.frame_id == self._frame_id)
        native = world._native[_CAMERA_PATH]
        key = (0, int(self._camera["width"]), int(self._camera["height"]))
        model = world._models[0]
        data = world._data[0]
        camera_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, native.camera_name))
        if camera_id < 0:
            raise RuntimeError("MuJoCo acceptance camera is absent from the native model")
        if not self._render_tuned:
            # Acceptance-only illumination: these visual fields do not enter
            # dynamics, collision, state, camera geometry, or control timing.
            model.vis.headlight.ambient[:] = (0.55, 0.55, 0.55)
            model.vis.headlight.diffuse[:] = (0.85, 0.85, 0.85)
            model.vis.headlight.specular[:] = (0.2, 0.2, 0.2)
            camera_spec = world._entities[_CAMERA_PATH].camera
            assert camera_spec is not None
            model.vis.map.znear = camera_spec.near_plane_m / float(model.stat.extent)
            model.vis.map.zfar = camera_spec.far_plane_m / float(model.stat.extent)
            self._render_tuned = True
        renderer = world._renderers.get(key)
        if renderer is None:
            renderer = mujoco.Renderer(model, height=key[2], width=key[1])
            world._renderers[key] = renderer
        renderer.disable_depth_rendering()
        renderer.update_scene(data, camera=native.camera_name)
        image = np.ascontiguousarray(renderer.render(), dtype=np.uint8)
        rgb = image.tobytes()
        actual_height, actual_width = int(image.shape[0]), int(image.shape[1])
        vertical_fov = float(model.cam_fovy[camera_id])
        focal_px = actual_height / (2.0 * math.tan(math.radians(vertical_fov) / 2.0))
        horizontal_fov = math.degrees(2.0 * math.atan(actual_width / (2.0 * focal_px)))
        camera_rotation = np.asarray(data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
        eye = _tuple3(data.cam_xpos[camera_id])
        forward = _normalize(_tuple3(-camera_rotation[index, 2] for index in range(3)))
        authored_eye = _tuple3(self._camera["eye_m"])
        authored_target = _tuple3(self._camera["look_at_m"])
        focus_distance = math.dist(authored_eye, authored_target)
        look_at = tuple(eye[index] + focus_distance * forward[index] for index in range(3))
        authored_up = _normalize(_tuple3(self._camera["up"]))
        optical_up = _normalize(_tuple3(camera_rotation[index, 1] for index in range(3)))
        up_forward_component = sum(authored_up[index] * forward[index] for index in range(3))
        projected_up = _normalize(
            _tuple3(authored_up[index] - up_forward_component * forward[index] for index in range(3))
        )
        if max(abs(left - right) for left, right in zip(optical_up, projected_up, strict=True)) > 1.0e-9:
            raise RuntimeError("MuJoCo native camera roll differs from the authored world-up reference")
        calibration = MappingProxyType(
            {
                "schema_version": "unirobosim-effective-camera-calibration/1",
                "resolution_px": [actual_width, actual_height],
                "model": "pinhole",
                "K_row_major": [
                    focal_px,
                    0.0,
                    actual_width / 2.0,
                    0.0,
                    focal_px,
                    actual_height / 2.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                "projection": {
                    "horizontal_fov_deg": horizontal_fov,
                    "vertical_fov_deg": vertical_fov,
                    "near_m": float(model.vis.map.znear) * float(model.stat.extent),
                    "far_m": float(model.vis.map.zfar) * float(model.stat.extent),
                },
                "extrinsics": {
                    "eye_m": eye,
                    "look_at_m": look_at,
                    "up": authored_up,
                },
                "evidence": {
                    "intrinsics_source": "MjModel.cam_fovy plus native renderer RGB array shape",
                    "projection_source": "MjModel.cam_fovy and vis.map clip fractions times MjModel.stat.extent",
                    "extrinsics_source": (
                        "MjData.cam_xpos/cam_xmat; native optical-up verified against world-up reference"
                    ),
                },
            }
        )
        physics_diagnostics = MappingProxyType(
            {
                "gravity_m_s2": tuple(float(item) for item in model.opt.gravity),
                "source": "MjModel.opt.gravity read after native world build",
            }
        )
        sample = MappingProxyType(
            {
                "simulation_tick": int(runtime.tick),
                "simulation_time_s": float(runtime.sim_time_seconds),
                "arm": {"joint_ids": _ARM, "position_rad": tuple(float(item) for item in value[:7])},
                "gripper": {
                    "joint_ids": _GRIPPER,
                    "position_rad": tuple(float(item) for item in value[7:13]),
                },
                "end_effector": {
                    "frame_id": "gripper_center",
                    "position_m": frame.world_pose.position_m,
                    "quaternion_xyzw": frame.world_pose.orientation_xyzw,
                },
                "rgb": {
                    "data": rgb,
                    "width": key[1],
                    "height": key[2],
                    "format": "rgb8",
                },
            }
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("MuJoCo acceptance capture is closed")
            self._camera_calibration = calibration
            self._physics_diagnostics = physics_diagnostics
            self._samples[int(runtime.tick)] = sample


class _BackendRun:
    def __init__(
        self,
        bundle: Any,
        capture: _CaptureParticipant,
        *,
        visible_window: bool,
        output_dir: Path,
        run_kind: str,
    ) -> None:
        self.bundle = bundle
        self._capture = capture
        self._visible_window = visible_window
        self._window_lifecycle_path = output_dir / f"droid-{run_kind}-mujoco.visible-window-lifecycle.json"
        self._display = os.environ.get("DISPLAY", "")
        self._window_ids_before = (
            frozenset(_x11_windows(self._display)) if visible_window and self._display else frozenset()
        )
        self._window_evidence: MappingProxyType[str, object] | None = None

    def sample(self, tick: int) -> MappingProxyType[str, object]:
        return self._capture.sample(tick)

    @property
    def camera_calibration(self) -> MappingProxyType[str, object]:
        return self._capture.camera_calibration()

    @property
    def physics_diagnostics(self) -> MappingProxyType[str, object]:
        return self._capture.physics_diagnostics()

    @property
    def window_evidence(self) -> MappingProxyType[str, object]:
        if not self._visible_window:
            raise RuntimeError("MuJoCo visible-window evidence was not requested")
        if not self._display:
            raise RuntimeError("MuJoCo GUI was requested without DISPLAY")
        if self._window_evidence is not None:
            return self._window_evidence
        deadline = time.monotonic() + 5.0
        candidates: list[tuple[bool, int, str, str]] = []
        while time.monotonic() < deadline:
            current = _x11_windows(self._display)
            candidates.clear()
            for window_id in sorted(set(current) - self._window_ids_before):
                title = current[window_id]
                detail = _visible_window_detail(self._display, window_id, title)
                if detail is None:
                    continue
                area, title = detail
                candidates.append(("mujoco" in title.lower(), area, window_id, title))
            if candidates:
                break
            time.sleep(0.1)
        if not candidates:
            raise RuntimeError("MuJoCo GUI window was not observed as IsViewable")
        _, _, window_id, title = max(candidates)
        if not title:
            title = "MuJoCo native viewer"
        self._window_evidence = MappingProxyType(
            {
                "schema_version": "fastsim-visible-window-evidence/1",
                "requested": True,
                "headless": False,
                "observed": True,
                "display": self._display,
                "native_window_id": window_id,
                "window_title": title,
                "source": (
                    f"xwininfo -display {self._display} -root -tree; "
                    f"xwininfo -display {self._display} -id {window_id}: Map State IsViewable; "
                    f"closure log={self._window_lifecycle_path.name}"
                ),
            }
        )
        return self._window_evidence

    def close(self) -> None:
        self._capture.discard()
        if not self._visible_window or self._window_evidence is None:
            return
        window_id = str(self._window_evidence["native_window_id"])
        deadline = time.monotonic() + 5.0
        while True:
            completed = subprocess.run(
                ("xwininfo", "-display", self._display, "-id", window_id),
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            destroyed = completed.returncode != 0 and (
                "No such window" in completed.stderr or "Bad Drawable" in completed.stderr
            )
            if destroyed or time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        lifecycle = {
            "schema_version": "fastsim-visible-window-lifecycle/1",
            "backend": "mujoco",
            "display": self._display,
            "native_window_id": window_id,
            "window_title": self._window_evidence["window_title"],
            "destroyed_after_bundle_close": destroyed,
            "xwininfo_returncode": completed.returncode,
            "xwininfo_stderr": completed.stderr.strip(),
        }
        temporary = self._window_lifecycle_path.with_name(f".{self._window_lifecycle_path.name}.tmp")
        temporary.write_text(json.dumps(lifecycle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self._window_lifecycle_path)
        if not destroyed:
            raise RuntimeError("MuJoCo GUI window survived bundle close")


def _bundle(
    plan: Any,
    projection: Any,
    capture_factory: Any,
    *,
    run_id: str,
    visible_window: bool,
) -> tuple[Any, _CaptureParticipant]:
    from fastsim.control import (  # type: ignore[import-not-found]
        ControlAuthorityDriver,
        ControlChunkExecutor,
        ControlChunkExecutorOptions,
        ControlService,
        ScheduleRatePolicy,
    )
    from fastsim.integrations.unirobosim._simulation_query import (  # type: ignore[import-not-found]
        _SimulationQueryRoot,
    )
    from fastsim.integrations.unirobosim.adapter import _UniRoboSimAdapter  # type: ignore[import-not-found]
    from fastsim.integrations.unirobosim.composition import (  # type: ignore[import-not-found]
        UniRoboSimRuntimeBundle,
        _controller_registry,
    )
    from fastsim.integrations.unirobosim.projection import observation_providers
    from fastsim.integrations.unirobosim.services import (  # type: ignore[import-not-found]
        PlanControlCapabilities,
        PlanWorldDependencies,
    )
    from fastsim.runtime import RuntimeKernel, RuntimeOptions

    adapter = _UniRoboSimAdapter(
        projection,
        entry_points=lambda: (_EntryPoint(visible_window=visible_window),),
    )
    simulation_root = _SimulationQueryRoot(run_id=run_id, projection=projection)
    executor = ControlChunkExecutor(
        run_id,
        controller_registry=_controller_registry(),
        dependency_provider=PlanWorldDependencies(projection, generation=lambda: adapter.generation),
        capability_provider=PlanControlCapabilities(projection),
        options=ControlChunkExecutorOptions(rate_policy=ScheduleRatePolicy(projection.rate_policy)),
    )
    driver = ControlAuthorityDriver(executor, adapter)
    capture = capture_factory(adapter)
    runtime = RuntimeKernel(
        plan,
        adapter,
        run_id=run_id,
        seed=int(plan.runtime["seed"]),
        options=RuntimeOptions(step_pacing_seconds=0.0),
        authority_participants=(driver.participant_spec(), capture.participant_spec()),
    )
    simulation_root._bind_authority_reads(runtime.submit_authority_read, lambda: runtime.snapshot)
    executor.bind_authority_submitter(runtime.submit_authority)
    for provider in observation_providers(projection):
        runtime.observations.register(provider)
    return (
        UniRoboSimRuntimeBundle(
            runtime=runtime,
            control=ControlService(executor),
            projection=projection,
            adapter=adapter,
            executor=executor,
            planning_raw=None,
            simulation_root=simulation_root,
            recording_capture=None,
        ),
        capture,
    )


def create_backend_run(
    spec: Any,
    run_kind: str,
    output_dir: Path,
    *,
    visible_window: bool = False,
) -> _BackendRun:
    """Return one unprepared real FastSim bundle and its committed sample cache."""

    if not isinstance(visible_window, bool):
        raise TypeError("visible_window must be a boolean")
    if run_kind not in {"rulebased_blocking", "model_servo_preempt"}:
        raise ValueError("unsupported DROID equivalence run kind")
    plan = _execution_plan(spec, visible_window=visible_window)
    projection = _acceptance_projection(plan, spec, visible_window=visible_window)
    sample_stride = int(spec["simulation"]["physics_hz"]) // int(spec["simulation"]["sample_hz"])
    bundle, capture = _bundle(
        plan,
        projection,
        lambda adapter: _CaptureParticipant(adapter, sample_stride=sample_stride, camera=spec["camera"]),
        run_id=f"droid-{run_kind}-mujoco",
        visible_window=visible_window,
    )
    return _BackendRun(
        bundle,
        capture,
        visible_window=visible_window,
        output_dir=output_dir,
        run_kind=run_kind,
    )


__all__ = ("create_backend_run",)
