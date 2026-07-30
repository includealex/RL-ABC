"""Early, dependency-free setup checks for RL-ABC entry points."""

from __future__ import annotations

import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict


SUPPORTED_PYTHON = (3, 10)

TESTED_PACKAGES = {
    "torch": "torch",
    "gymnasium": "gymnasium",
    "stable-baselines3": "stable-baselines3",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "scikit-learn",
}


class SetupError(RuntimeError):
    """Raised when required RL-ABC inputs or external tools are unavailable."""


def require_supported_python() -> None:
    """Fail before importing scientific packages on an unsupported Python."""
    running = sys.version_info[:2]
    if running != SUPPORTED_PYTHON:
        expected = ".".join(map(str, SUPPORTED_PYTHON))
        actual = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        raise SystemExit(
            f"ERROR: RL-ABC requires Python {expected}.x; found Python {actual}. "
            "Create the environment with `python3.10 -m venv .venv`, activate it, "
            "and install `requirements.txt`."
        )


def dependency_versions() -> Dict[str, str]:
    """Return installed versions for the principal runtime dependencies."""
    installed: Dict[str, str] = {}
    for label, distribution in TESTED_PACKAGES.items():
        try:
            installed[label] = version(distribution)
        except PackageNotFoundError:
            installed[label] = "not installed"
    return installed


def _resolve_executable(elegant_path: str, executable: str) -> Path | None:
    if elegant_path:
        candidate = Path(elegant_path).expanduser() / executable
        if candidate.is_file():
            return candidate.resolve()
        return None

    located = shutil.which(executable)
    return Path(located).resolve() if located else None


def validate_runtime_config(config: Any, project_dir: Path) -> Dict[str, Path]:
    """Validate required inputs and external Elegant/SDDS tools."""
    project_dir = project_dir.resolve()
    errors = []
    resolved: Dict[str, Path] = {}

    input_beamline = Path(config.input_beamline_file)
    if not input_beamline.is_absolute():
        input_beamline = project_dir / input_beamline

    input_beam = Path(str(config.input_beam_file))
    if input_beam.suffix.lower() != ".ele":
        input_beam = input_beam.with_suffix(".ele")
    if not input_beam.is_absolute():
        input_beam = project_dir / input_beam

    for label, path in (
        ("beamline input", input_beamline),
        ("Elegant command input", input_beam),
    ):
        if path.is_file():
            resolved[label] = path.resolve()
        else:
            errors.append(
                f"Missing {label}: {path}. Keep `machine.lte` and `track.ele` "
                "in the same directory as `train.py`, or update `config.json`."
            )

    elegant_path = str(config.elegant_path).strip()
    required_tools = ["sdds2stream", "sddsquery", "sddsprocess"]
    if not config.override_dynamic_command:
        required_tools.insert(0, "elegant")

    for executable in required_tools:
        path = _resolve_executable(elegant_path, executable)
        if path is None:
            location = Path(elegant_path).expanduser() if elegant_path else "PATH"
            errors.append(f"Cannot find `{executable}` in {location}.")
        else:
            resolved[executable] = path

    sdds_path = Path(str(config.sdds_path)).expanduser()
    if not sdds_path.is_absolute():
        sdds_path = project_dir / sdds_path
    if sdds_path.is_file():
        resolved["defns.rpn"] = sdds_path.resolve()
    else:
        errors.append(f"Cannot find the SDDS definitions file: {sdds_path}.")

    if errors:
        formatted = "\n".join(f"  - {message}" for message in errors)
        raise SetupError(f"RL-ABC setup check failed:\n{formatted}")

    return resolved
