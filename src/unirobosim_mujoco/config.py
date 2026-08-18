"""Validated launch configuration for the MuJoCo adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass

from unirobosim import ValidationError


@dataclass(frozen=True)
class MuJoCoAdapterConfig:
    render_width: int = 640
    render_height: int = 480
    max_cached_commands: int = 4096
    position_stiffness: float = 100.0
    position_damping: float = 8.0
    velocity_gain: float = 20.0
    max_motor_effort: float = 50.0

    def __post_init__(self) -> None:
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
        if (
            isinstance(self.max_motor_effort, bool)
            or not isinstance(self.max_motor_effort, (int, float))
            or not math.isfinite(float(self.max_motor_effort))
            or float(self.max_motor_effort) <= 0.0
        ):
            raise ValidationError("max_motor_effort must be positive and finite", operation="mujoco.config")
