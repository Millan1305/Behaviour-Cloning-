#!/usr/bin/env bash
# Master dependency installer for Ubuntu 24.04 + ROS 2 Jazzy + Gazebo Harmonic.
# - never runs `apt upgrade`, only installs packages that are missing
# - stops with a clear error on any failure
set -Eeuo pipefail
trap 'echo -e "\n\033[1;31mERROR:\033[0m install_dependencies.sh failed at line $LINENO: $BASH_COMMAND" >&2' ERR
source "$(dirname "${BASH_SOURCE[0]}")/scripts/common.sh"
check_dir
cd "$ROOT"

. /etc/os-release
[ "${ID:-}" = "ubuntu" ] && [ "${VERSION_ID:-}" = "24.04" ] || die "This project targets Ubuntu 24.04 (found ${PRETTY_NAME:-unknown})."
[ "$(id -u)" -ne 0 ] || die "Do not run as root; the script calls sudo only where needed."
command -v sudo >/dev/null || die "sudo is required."

# ---------------------------------------------------------------- ROS 2 Jazzy
if [ ! -f /opt/ros/jazzy/setup.bash ]; then
  info "ROS 2 Jazzy not found -> installing ros-jazzy-ros-base (official apt source)"
  sudo apt-get update
  sudo apt-get install -y software-properties-common curl locales
  sudo add-apt-repository -y universe
  sudo locale-gen en_US en_US.UTF-8
  sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
  export LANG=en_US.UTF-8
  V=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
  [ -n "$V" ] || die "Could not determine ros-apt-source version (no internet / GitHub API rate limit?)."
  curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${V}/ros2-apt-source_${V}.${VERSION_CODENAME}_all.deb"
  sudo dpkg -i /tmp/ros2-apt-source.deb
  sudo apt-get update
  sudo apt-get install -y ros-jazzy-ros-base
else
  ok "ROS 2 Jazzy present"
fi

# ---------------------------------------------------------------- apt packages
PKGS=(
  ros-jazzy-ros-base
  ros-jazzy-ros-gz            # ros_gz_sim, ros_gz_bridge (pulls Gazebo Harmonic)
  ros-jazzy-gz-ros2-control   # ros2_control <-> Gazebo Sim
  ros-jazzy-ros2-control
  ros-jazzy-ros2-controllers  # position_controllers, joint_state_broadcaster
  ros-jazzy-controller-manager
  ros-jazzy-robot-state-publisher
  ros-jazzy-xacro
  python3-colcon-common-extensions python3-rosdep python3-pip python3-venv python3-yaml python3-numpy
  git curl v4l-utils libgl1 libglib2.0-0
)
MISSING=()
for p in "${PKGS[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || MISSING+=("$p"); done
if [ ${#MISSING[@]} -gt 0 ]; then
  info "Installing missing apt packages: ${MISSING[*]}"
  sudo apt-get update
  sudo apt-get install -y "${MISSING[@]}"
else
  ok "all apt packages already installed"
fi
command -v gz >/dev/null || die "'gz' (Gazebo) not found after install. Try: sudo apt-get install ros-jazzy-ros-gz"

# ---------------------------------------------------------------- Python venv
# --system-site-packages: rclpy & other ROS Python packages are installed by apt for the SYSTEM python3.12
# and cannot be pip-installed, so the venv must be able to see them.
if [ ! -f .venv/bin/activate ]; then
  info "Creating .venv (python3 -m venv --system-site-packages .venv)"
  python3 -m venv --system-site-packages .venv
fi
set +u; source .venv/bin/activate; set -u
python -m pip install --upgrade pip
python -m pip install "numpy==1.26.4"
if ! python -c "import torch" 2>/dev/null; then
  info "Installing PyTorch (CPU build; no CUDA needed)"
  python -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu
else
  ok "PyTorch present"
fi
python -m pip install -r requirements.txt

# ---------------------------------------------------------------- verify
ros_source
python - <<'PY'
import importlib, sys
bad = []
for m in ("numpy", "torch", "cv2", "mediapipe", "rclpy", "yaml"):
    try:
        mod = importlib.import_module(m); print(f"  OK  {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        bad.append(f"{m}: {e}"); print(f"  FAIL {m}: {e}")
sys.exit(1 if bad else 0)
PY
info "Dependencies installed."
