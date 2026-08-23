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
    headless: bool = True
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
        if not isinstance(self.headless, bool):
            raise ValidationError("headless must be a boolean", operation="mujoco.config")
        values = (self.render_width, self.render_height, self.max_cached_commands)
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
