# Changelog

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
