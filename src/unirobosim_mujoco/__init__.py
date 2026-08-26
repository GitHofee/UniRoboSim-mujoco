"""UniRoboSim MuJoCo provider entry point."""

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from unirobosim import ValidationError

from .config import MuJoCoAdapterConfig
from .descriptor import CAMERA_CAPABILITIES, DESCRIPTOR, PHYSICS_ONLY_CAPABILITIES, descriptor_for_config
from .provider import MuJoCoProvider, MuJoCoSession

if TYPE_CHECKING:
    from .world import MuJoCoWorld

_LAUNCH_PROFILES = {
    "visible": (False, True),
    "headless": (True, True),
    "headless-physics": (True, False),
}


def create_provider(
    config: MuJoCoAdapterConfig | None = None,
    *,
    launch_profile: str | None = None,
) -> MuJoCoProvider:
    """Create the provider, optionally applying FastSim's canonical launch profile."""

    resolved_config = config if config is not None else MuJoCoAdapterConfig()
    if not isinstance(resolved_config, MuJoCoAdapterConfig):
        raise ValidationError("config must be MuJoCoAdapterConfig", operation="mujoco.provider.init")
    if launch_profile is not None:
        mapping = _LAUNCH_PROFILES.get(launch_profile) if isinstance(launch_profile, str) else None
        if mapping is None:
            raise ValidationError(
                "MuJoCo launch profile must be 'visible', 'headless', or 'headless-physics'",
                operation="mujoco.launch_profile.resolve",
            ) from None
        resolved_config = replace(resolved_config, headless=mapping[0], enable_cameras=mapping[1])
    return MuJoCoProvider(resolved_config)


def __getattr__(name: str) -> Any:
    """Keep factory discovery import-safe while preserving the public World export."""

    if name == "MuJoCoWorld":
        from .world import MuJoCoWorld

        return MuJoCoWorld
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DESCRIPTOR",
    "CAMERA_CAPABILITIES",
    "PHYSICS_ONLY_CAPABILITIES",
    "MuJoCoAdapterConfig",
    "MuJoCoProvider",
    "MuJoCoSession",
    "MuJoCoWorld",
    "create_provider",
    "descriptor_for_config",
]

__version__ = "0.9.2"
