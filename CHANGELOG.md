# Changelog

## 0.9.3 - 2026-08-26

- Validate component-owned articulation drive profiles at the adapter boundary and
  compile stiffness/damping into immutable arrays scoped by entity path.
- Subdivide coarse logical steps into bounded native MuJoCo integration steps while
  preserving the public logical tick, time, camera cadence, and viewer cadence.
- Add deterministic, extreme-step, reset, camera, omission-path, and same-joint-name
  multi-entity regression coverage.

## 0.9.2 - 2026-08-26

- Return RGB camera channels through Core's compact immutable byte storage so
  FastSim simulation queries receive tightly packed RGB24 frames without expanding
  every pixel into Python integers.
- Retain the scalar tuple fallback when running with UniRoboSim Core 0.9.
- Make the DROID acceptance asset portable through explicit, config, environment,
  and backward-compatible user-local discovery instead of a packaged host path.

## 0.9.1 - 2026-08-26

- Add the canonical `visible`, `headless`, and `headless-physics` factory profiles
  while preserving the zero-argument headless camera default.
- Make backend discovery import-safe and omit camera capabilities when cameras are
  disabled.
- Reject camera entities before native world or renderer allocation in the
  physics-only profile.
- Reject invalid provider configuration types instead of replacing false-like
  values with defaults.
- Keep the packaged DROID acceptance hook compatible with FastSim's launch-profile
  discovery and simulation-query runtime bundle.
- Verify the same v0alpha5 provider against UniRoboSim Core 0.9 and 0.10, and
  declare that tested dependency range.

## 0.7.0 - 2026-08-19

- Publish the verified MuJoCo 3.11 adapter in the coordinated UniRoboSim 0.7 release.
- Align distribution, importable and provider versions.
- Declare the shared `v0alpha4` Runtime contract and Core `>=0.7.0,<0.8` compatibility.
