"""Static, import-safe MuJoCo capability declaration."""

from unirobosim import (
    PHYSICAL_WORLD_SCHEMA_VERSION,
    WORLD_SCHEMA_VERSION,
    CapabilityDeclaration,
    CapabilityId,
    CapabilitySet,
    FrozenMap,
    ProviderDescriptor,
)

CAPABILITIES = CapabilitySet(
    (
        CapabilityDeclaration(
            CapabilityId("profile.core-robotics@1"),
            FrozenMap(
                {
                    "coordinate_system": "right-handed-z-up",
                    "quaternion_order": "xyzw",
                    "array_layout": "batch-first",
                }
            ),
        ),
        CapabilityDeclaration(
            CapabilityId("asset.formats@1"),
            FrozenMap(
                {
                    "rigid_body": ["application/xml", "model/vnd.mujoco.mjcf+xml"],
                    "articulation": [
                        "application/xml",
                        "model/vnd.mujoco.mjcf+xml",
                        "model/vnd.urdf+xml",
                    ],
                }
            ),
        ),
        CapabilityDeclaration(CapabilityId("world.multi-environment@1")),
        CapabilityDeclaration(CapabilityId("state.rigid_body@1")),
        CapabilityDeclaration(CapabilityId("control.rigid_body.wrench@1")),
        CapabilityDeclaration(CapabilityId("contact.binary@1")),
        CapabilityDeclaration(CapabilityId("contact.net_normal_force@1")),
        CapabilityDeclaration(CapabilityId("state.articulation@1")),
        CapabilityDeclaration(CapabilityId("state.articulation.axis-units@1")),
        CapabilityDeclaration(CapabilityId("control.articulation.position@1")),
        CapabilityDeclaration(CapabilityId("control.articulation.position.axis-units@1")),
        CapabilityDeclaration(CapabilityId("control.articulation.velocity@1")),
        CapabilityDeclaration(CapabilityId("control.articulation.effort@1")),
        CapabilityDeclaration(
            CapabilityId("sensor.camera@1"),
            FrozenMap({"schedule": "synchronous", "pose_frame": "environment-local-world"}),
        ),
        CapabilityDeclaration(CapabilityId("sensor.camera.rgb@1")),
        CapabilityDeclaration(CapabilityId("sensor.camera.depth@1")),
        CapabilityDeclaration(
            CapabilityId("debug.sink.native_overlay@1"),
            FrozenMap({"endpoint": "mjvScene-compatible", "portable_store": True}),
            limitations=("headless worlds retain debug geometry for Studio instead of opening a native viewer",),
        ),
        CapabilityDeclaration(CapabilityId("scene.snapshot@1")),
        CapabilityDeclaration(CapabilityId("scene.delta@1")),
        CapabilityDeclaration(CapabilityId("scene.command.pose@1")),
        CapabilityDeclaration(
            CapabilityId("scene.command.drag@1"),
            FrozenMap({"entity_kinds": ["rigid_body"], "modes": ["kinematic"]}),
            limitations=("constraint drag is not exposed in the initial adapter",),
        ),
        CapabilityDeclaration(CapabilityId("render.browser-scene@1")),
        CapabilityDeclaration(
            CapabilityId("planning.scene@2"),
            FrozenMap(
                {
                    "authority_thread": "synchronous",
                    "axis_convention": "right_handed_z_up",
                    "geometry_read_limit_bytes": 64 * 1024 * 1024,
                    "resource_layout": "catalog-pinned-v1",
                    "single_representation_per_geometry": True,
                    "representation_fallback": False,
                }
            ),
        ),
    )
)

_CAMERA_CAPABILITY_IDS = frozenset(
    {
        CapabilityId("sensor.camera@1"),
        CapabilityId("sensor.camera.rgb@1"),
        CapabilityId("sensor.camera.depth@1"),
    }
)
CAMERA_CAPABILITIES = CapabilitySet(
    declaration for declaration in CAPABILITIES if declaration.capability in _CAMERA_CAPABILITY_IDS
)
PHYSICS_ONLY_CAPABILITIES = CapabilitySet(
    declaration for declaration in CAPABILITIES if declaration.capability not in _CAMERA_CAPABILITY_IDS
)

DESCRIPTOR = ProviderDescriptor(
    "google-deepmind.mujoco",
    "UniRoboSim MuJoCo",
    "0.9.1",
    "v0alpha5",
    CAPABILITIES,
    (WORLD_SCHEMA_VERSION, PHYSICAL_WORLD_SCHEMA_VERSION),
    FrozenMap({"mujoco": "3.11.0", "python": "3.12"}),
)


def descriptor_for_config(config: object) -> ProviderDescriptor:
    """Remove camera claims when native camera allocation is disabled."""

    if bool(getattr(config, "enable_cameras", True)):
        return DESCRIPTOR
    return ProviderDescriptor(
        DESCRIPTOR.provider_id,
        DESCRIPTOR.display_name,
        DESCRIPTOR.version,
        DESCRIPTOR.contract_version,
        PHYSICS_ONLY_CAPABILITIES,
        DESCRIPTOR.supported_world_schema_versions,
        DESCRIPTOR.metadata,
    )
