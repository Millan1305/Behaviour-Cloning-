"""Make `robot_cloning` importable from scripts without colcon and give friendly errors."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "robot_cloning"))


def need(module, hint="Activate the venv: source ~/robot_cloning/.venv/bin/activate (created by ./install_dependencies.sh)"):
    try:
        return __import__(module)
    except ImportError:
        sys.exit(f"ERROR: python module '{module}' is missing.\n{hint}")
