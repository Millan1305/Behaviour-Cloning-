"""Locate the repository root (works from source tree, symlink-install, or via ROBOT_CLONING_ROOT)."""
import os
from pathlib import Path


def _is_root(p: Path) -> bool:
    return (p / "config" / "h1_joints.yaml").exists() and (p / "scripts").is_dir()


def repo_root() -> Path:
    env = os.environ.get("ROBOT_CLONING_ROOT")
    if env and _is_root(Path(env)):
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in list(here.parents):
        if _is_root(parent):
            return parent
    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        if _is_root(parent):
            return parent
    raise RuntimeError(
        "Cannot locate the robot_cloning repository root. Run from inside the repo "
        "or `export ROBOT_CLONING_ROOT=~/robot_cloning`."
    )


_ROOT = None


def root() -> Path:
    global _ROOT
    if _ROOT is None:
        _ROOT = repo_root()
    return _ROOT


def default_model_path() -> Path:
    return root() / "models" / "bc_model.pt"


def default_reference_path() -> Path:
    return root() / "models" / "reference_trajectories.npz"
