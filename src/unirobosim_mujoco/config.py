"""Validated launch configuration for the MuJoCo adapter."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from unirobosim import ValidationError


@dataclass(frozen=True)
class MuJoCoAdapterConfig:
    render_width: int = 640
    render_height: int = 480
    max_cached_commands: int = 4096
    position_stiffness: float = 100.0
    position_damping: float = 8.0
    joint_position_stiffness: tuple[tuple[str, float], ...] = ()
    joint_position_damping: tuple[tuple[str, float], ...] = ()
    velocity_gain: float = 20.0
    max_motor_effort: float = 50.0
    max_native_time_step_seconds: float = 1.0 / 240.0
    max_native_substeps_per_logical_step: int = 4096
    headless: bool = True
    enable_cameras: bool = True
    _joint_position_stiffness_lookup: Mapping[str, float] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _joint_position_damping_lookup: Mapping[str, float] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.headless, bool) or not isinstance(self.enable_cameras, bool):
            raise ValidationError("launch flags must be boolean", operation="mujoco.config")
        values = (
            self.render_width,
            self.render_height,
            self.max_cached_commands,
            self.max_native_substeps_per_logical_step,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values):
            raise ValidationError("MuJoCo adapter limits must be positive integers", operation="mujoco.config")
        for name in ("position_stiffness", "position_damping", "velocity_gain"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValidationError(f"{name} must be non-negative and finite", operation="mujoco.config")
        if (
            isinstance(self.max_native_time_step_seconds, bool)
            or not isinstance(self.max_native_time_step_seconds, (int, float))
            or not math.isfinite(float(self.max_native_time_step_seconds))
            or float(self.max_native_time_step_seconds) <= 0.0
        ):
            raise ValidationError(
                "max_native_time_step_seconds must be positive and finite",
                operation="mujoco.config",
            )
        for name in ("joint_position_stiffness", "joint_position_damping"):
            entries = getattr(self, name)
            if not isinstance(entries, tuple):
                raise ValidationError(f"{name} must be a tuple", operation="mujoco.config")
            seen: set[str] = set()
            for entry in entries:
                if not isinstance(entry, tuple) or len(entry) != 2:
                    raise ValidationError(
                        f"{name} entries must be (joint_name, gain) tuples",
                        operation="mujoco.config",
                    )
                joint_name, value = entry
                if not isinstance(joint_name, str) or not joint_name or joint_name in seen:
                    raise ValidationError(
                        f"{name} joint names must be non-empty and unique",
                        operation="mujoco.config",
                    )
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                ):
                    raise ValidationError(
                        f"{name} gains must be non-negative and finite",
                        operation="mujoco.config",
                    )
                seen.add(joint_name)
        if (
            isinstance(self.max_motor_effort, bool)
            or not isinstance(self.max_motor_effort, (int, float))
            or not math.isfinite(float(self.max_motor_effort))
            or float(self.max_motor_effort) <= 0.0
        ):
            raise ValidationError("max_motor_effort must be positive and finite", operation="mujoco.config")

        object.__setattr__(
            self,
            "_joint_position_stiffness_lookup",
            MappingProxyType({joint_name: float(value) for joint_name, value in self.joint_position_stiffness}),
        )
        object.__setattr__(
            self,
            "_joint_position_damping_lookup",
            MappingProxyType({joint_name: float(value) for joint_name, value in self.joint_position_damping}),
        )

    def position_stiffness_for(self, joint_name: str) -> float:
        if not self.joint_position_stiffness:
            return float(self.position_stiffness)
        return self._joint_position_stiffness_lookup.get(joint_name, float(self.position_stiffness))

    def position_damping_for(self, joint_name: str) -> float:
        if not self.joint_position_damping:
            return float(self.position_damping)
        return self._joint_position_damping_lookup.get(joint_name, float(self.position_damping))

    def native_substeps_for(self, logical_time_step_seconds: float, authored_substeps: int) -> int:
        """Resolve the minimum bounded native subdivision for one logical step."""

        if (
            isinstance(logical_time_step_seconds, bool)
            or not isinstance(logical_time_step_seconds, (int, float))
            or not math.isfinite(float(logical_time_step_seconds))
            or float(logical_time_step_seconds) <= 0.0
        ):
            raise ValidationError(
                "logical_time_step_seconds must be positive and finite",
                operation="mujoco.config.native_substeps",
            )
        if not isinstance(authored_substeps, int) or isinstance(authored_substeps, bool) or authored_substeps <= 0:
            raise ValidationError(
                "authored_substeps must be a positive integer",
                operation="mujoco.config.native_substeps",
            )
        logical_step = float(logical_time_step_seconds)
        maximum_native_step = float(self.max_native_time_step_seconds)
        ratio = logical_step / maximum_native_step
        if not math.isfinite(ratio) or ratio > self.max_native_substeps_per_logical_step:
            required_native_substeps = (
                self.max_native_substeps_per_logical_step + 1 if not math.isfinite(ratio) else math.ceil(ratio)
            )
            raise ValidationError(
                "logical step requires more native substeps than the configured safety limit",
                operation="mujoco.config.native_substeps",
                details={
                    "logical_time_step_seconds": logical_step,
                    "max_native_time_step_seconds": maximum_native_step,
                    "authored_substeps": authored_substeps,
                    "required_native_substeps": required_native_substeps,
                    "max_native_substeps_per_logical_step": self.max_native_substeps_per_logical_step,
                },
            )
        automatic_substeps = max(1, math.ceil(ratio))
        # Correct only floating-point boundary noise while retaining the exact
        # invariant that every native step is no longer than the configured cap.
        while automatic_substeps > 1 and logical_step / (automatic_substeps - 1) <= maximum_native_step:
            automatic_substeps -= 1
        while logical_step / automatic_substeps > maximum_native_step:
            automatic_substeps += 1
        resolved = max(authored_substeps, automatic_substeps)
        if resolved > self.max_native_substeps_per_logical_step:
            raise ValidationError(
                "logical step requires more native substeps than the configured safety limit",
                operation="mujoco.config.native_substeps",
                details={
                    "logical_time_step_seconds": logical_step,
                    "max_native_time_step_seconds": maximum_native_step,
                    "authored_substeps": authored_substeps,
                    "required_native_substeps": resolved,
                    "max_native_substeps_per_logical_step": self.max_native_substeps_per_logical_step,
                },
            )
        return resolved
