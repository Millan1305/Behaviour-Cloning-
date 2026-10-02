#!/usr/bin/env bash
# Build the ROS 2 workspace.  Run from anywhere:  ./build.sh
set -Eeuo pipefail
trap 'echo -e "\n\033[1;31mERROR:\033[0m build.sh failed at line $LINENO: $BASH_COMMAND" >&2' ERR
source "$(dirname "${BASH_SOURCE[0]}")/scripts/common.sh"
check_dir
# colcon must use the SYSTEM python (ROS entry-point shebangs), so remove any active venv from PATH
if [ -n "${VIRTUAL_ENV:-}" ]; then
  PATH="$(echo "$PATH" | tr ':' '\n' | grep -v "^${VIRTUAL_ENV}/bin$" | paste -sd:)"; export PATH; unset VIRTUAL_ENV
fi
ros_source
cd "$ROOT"
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then sudo rosdep init; fi
rosdep update --rosdistro jazzy || echo "WARNING: rosdep update failed (offline?) - continuing with cached data"
rosdep install --from-paths src --ignore-src -r -y --rosdistro jazzy
colcon build --symlink-install --base-paths src
set +u; source install/setup.bash; set -u
ros2 pkg prefix robot_cloning >/dev/null || die "robot_cloning package not found after build"
info "Build OK. In every new terminal run:  source /opt/ros/jazzy/setup.bash && source $ROOT/install/setup.bash"
