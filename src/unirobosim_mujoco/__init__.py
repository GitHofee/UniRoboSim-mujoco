"""UniRoboSim MuJoCo provider entry point."""

from .config import MuJoCoAdapterConfig
from .descriptor import DESCRIPTOR
from .provider import MuJoCoProvider, MuJoCoSession
from .world import MuJoCoWorld


def create_provider(config: MuJoCoAdapterConfig | None = None) -> MuJoCoProvider:
    return MuJoCoProvider(config)


__all__ = [
    "DESCRIPTOR",
    "MuJoCoAdapterConfig",
    "MuJoCoProvider",
    "MuJoCoSession",
    "MuJoCoWorld",
    "create_provider",
]

__version__ = "0.9.0"
