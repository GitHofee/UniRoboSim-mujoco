from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from unirobosim_mujoco._droid_asset import DROID_ASSET_ENV, resolve_droid_asset_path
from unirobosim_mujoco.droid_acceptance import create_backend_run


def _asset(root: Path, name: str) -> Path:
    result = root / name
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text("<robot name='droid'/>", encoding="utf-8")
    return result.resolve()


def test_explicit_config_and_environment_precedence(tmp_path: Path) -> None:
    explicit = _asset(tmp_path, "explicit.urdf")
    configured = _asset(tmp_path, "configured.urdf")
    environment = _asset(tmp_path, "environment.urdf")

    assert (
        resolve_droid_asset_path(
            explicit,
            configured_asset_path=configured,
            environ={DROID_ASSET_ENV: str(environment)},
        )
        == explicit
    )
    assert (
        resolve_droid_asset_path(
            configured_asset_path=configured,
            environ={DROID_ASSET_ENV: str(environment)},
        )
        == configured
    )
    assert resolve_droid_asset_path(environ={DROID_ASSET_ENV: str(environment)}) == environment


def test_existing_legacy_user_path_is_the_last_fallback(tmp_path: Path) -> None:
    expected = _asset(
        tmp_path,
        "projects/gen_data/data/robots/droid/droid_mujoco.urdf",
    )

    assert resolve_droid_asset_path(environ={}, home=tmp_path) == expected


def test_missing_asset_has_actionable_portable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as caught:
        resolve_droid_asset_path(environ={}, home=tmp_path)

    message = str(caught.value)
    assert "asset_path" in message and "robot.asset_path" in message
    assert DROID_ASSET_ENV in message and "/home/" not in message


def test_selected_path_must_be_a_regular_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_droid_asset_path(tmp_path / "missing.urdf", environ={})


def test_acceptance_asset_path_is_optional_keyword_only() -> None:
    parameter = inspect.signature(create_backend_run).parameters["asset_path"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None
