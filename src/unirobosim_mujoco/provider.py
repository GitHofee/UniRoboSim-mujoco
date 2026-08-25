"""MuJoCo provider/session lifecycle with lazy native imports."""

from __future__ import annotations

import importlib
import itertools
from collections.abc import Iterable
from types import TracebackType
from typing import TYPE_CHECKING

from unirobosim import (
    WORLD_SCHEMA_UNSUPPORTED,
    BuildInput,
    CapabilityId,
    CapabilityNegotiationError,
    CapabilityRequirement,
    EntityKind,
    LifecycleError,
    NegotiationReport,
    PlanningSceneError,
    ProbeReport,
    ProviderDescriptor,
    ProviderSelectionError,
    SessionState,
    UnsupportedCapabilityError,
    ValidationError,
    WorldBuildError,
    WorldSpec,
)

from .build_assets import snapshot_build_input
from .config import MuJoCoAdapterConfig
from .descriptor import descriptor_for_config

if TYPE_CHECKING:
    from .world import MuJoCoWorld

_SESSION_IDS = itertools.count(1)


class MuJoCoProvider:
    def __init__(self, config: MuJoCoAdapterConfig | None = None) -> None:
        self._config = config if config is not None else MuJoCoAdapterConfig()
        if not isinstance(self._config, MuJoCoAdapterConfig):
            raise ValidationError("config must be MuJoCoAdapterConfig", operation="mujoco.provider.init")
        self._descriptor = descriptor_for_config(self._config)
        self._active: MuJoCoSession | None = None

    @property
    def descriptor(self) -> ProviderDescriptor:
        return self._descriptor

    def probe(self) -> ProbeReport:
        try:
            module = importlib.import_module("mujoco")
            version = str(module.__version__)
        except Exception as exc:
            return ProbeReport(self.descriptor, False, f"MuJoCo import failed: {exc}")
        available = version == "3.11.0"
        return ProbeReport(
            self.descriptor,
            available,
            None if available else f"expected MuJoCo 3.11.0, found {version}",
        )

    def open(self) -> MuJoCoSession:
        if self._active is not None and self._active.state is not SessionState.CLOSED:
            raise LifecycleError("provider already owns a live session", operation="mujoco.provider.open")
        probe = self.probe()
        if not probe.available:
            raise ProviderSelectionError(
                "MuJoCo compatibility profile is unavailable",
                operation="mujoco.provider.open",
                backend_id=self.descriptor.provider_id,
                details={"reason": probe.reason},
            )
        session = MuJoCoSession(self, self._config)
        self._active = session
        return session

    def _closed(self, session: MuJoCoSession) -> None:
        if self._active is session:
            self._active = None


class MuJoCoSession:
    def __init__(self, provider: MuJoCoProvider, config: MuJoCoAdapterConfig) -> None:
        self._provider = provider
        self.config = config
        self._session_id = f"mujoco-session-{next(_SESSION_IDS)}"
        self._state = SessionState.OPEN
        self._generation = 0
        self._world: MuJoCoWorld | None = None

    @property
    def descriptor(self) -> ProviderDescriptor:
        return self._provider.descriptor

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def state(self) -> SessionState:
        return self._state

    def _ensure(self, operation: str, *, ready: bool = False) -> None:
        accepted = {SessionState.OPEN, SessionState.READY} if ready else {SessionState.OPEN}
        if self._state not in accepted:
            raise LifecycleError("session lifecycle state is invalid", operation=operation)

    def negotiate(self, requirements: Iterable[CapabilityRequirement]) -> NegotiationReport:
        self._ensure("mujoco.session.negotiate", ready=True)
        return self.descriptor.capabilities.negotiate(tuple(requirements))

    def build(self, spec: WorldSpec, *, build_input: BuildInput | None = None) -> MuJoCoWorld:
        self._ensure("mujoco.session.build")
        if not isinstance(spec, WorldSpec):
            raise ValidationError("build requires a WorldSpec", operation="mujoco.session.build")
        if spec.schema_version not in self.descriptor.supported_world_schema_versions:
            raise UnsupportedCapabilityError(
                "provider does not support the requested World schema",
                operation="mujoco.session.build",
                backend_id=self.descriptor.provider_id,
                world_id=spec.world_id,
                details={
                    "detail_code": WORLD_SCHEMA_UNSUPPORTED,
                    "requested_schema": spec.schema_version,
                    "provider_id": self.descriptor.provider_id,
                    "supported_world_schema_versions": self.descriptor.supported_world_schema_versions,
                },
            ) from None
        if not self.config.enable_cameras:
            camera = next((entity for entity in spec.entities if entity.kind is EntityKind.CAMERA_SENSOR), None)
            if camera is not None:
                raise UnsupportedCapabilityError(
                    "camera entities are disabled by this MuJoCo launch profile",
                    operation="mujoco.session.build.preflight",
                    backend_id=self.descriptor.provider_id,
                    world_id=spec.world_id,
                    entity_path=camera.path.value,
                    details={"enable_cameras": False},
                ) from None
        negotiation = self.negotiate(spec.requirements)
        if not negotiation.accepted:
            raise CapabilityNegotiationError(
                "world requirements are not satisfied",
                operation="mujoco.session.build",
                backend_id=self.descriptor.provider_id,
                world_id=spec.world_id,
                details={"negotiation": negotiation.to_dict()},
            )
        asset_lease = snapshot_build_input(spec, build_input)
        self._generation += 1
        from .world import MuJoCoWorld

        world: MuJoCoWorld
        try:
            if CapabilityId("planning.scene@2") in negotiation.matched:
                from .planning import MuJoCoPlanningWorld

                world = MuJoCoPlanningWorld(self, spec, self._generation, asset_lease)
            else:
                world = MuJoCoWorld(self, spec, self._generation, asset_lease)
        except PlanningSceneError:
            if asset_lease is not None:
                asset_lease.close()
            raise
        except Exception as exc:
            if asset_lease is not None:
                asset_lease.close()
            raise WorldBuildError(
                "MuJoCo world build failed",
                operation="mujoco.session.build",
                backend_id=self.descriptor.provider_id,
                world_id=spec.world_id,
                cause=exc,
            ) from exc
        self._world = world
        self._state = SessionState.READY
        return world

    def _world_closed(self, world: MuJoCoWorld) -> None:
        if self._world is world:
            self._world = None
            if self._state is not SessionState.CLOSED:
                self._state = SessionState.OPEN

    def close(self) -> None:
        if self._state is SessionState.CLOSED:
            return
        world = self._world
        self._world = None
        self._state = SessionState.CLOSED
        if world is not None:
            world._close(notify_session=False)
        self._provider._closed(self)

    def __enter__(self) -> MuJoCoSession:
        self._ensure("mujoco.session.enter", ready=True)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
