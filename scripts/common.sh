# shared helpers (sourced by the other scripts)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROBOT_CLONING_ROOT="$ROOT"
die()  { echo -e "\n\033[1;31mERROR:\033[0m $*" >&2; exit 1; }
info() { echo -e "\033[1;34m[robot_cloning]\033[0m $*"; }
ok()   { echo -e "\033[1;32m  OK\033[0m  $*"; }
ros_source() {
  [ -f /opt/ros/jazzy/setup.bash ] || die "ROS 2 Jazzy not found at /opt/ros/jazzy. Run ./install_dependencies.sh"
  set +u; source /opt/ros/jazzy/setup.bash; set -u
}
ws_source() {
  [ -f "$ROOT/install/setup.bash" ] || die "Workspace not built yet. Run ./build.sh first."
  set +u; source "$ROOT/install/setup.bash"; set -u
}
venv_activate() {
  [ -f "$ROOT/.venv/bin/activate" ] || die "Python venv missing ($ROOT/.venv). Run ./install_dependencies.sh"
  set +u; source "$ROOT/.venv/bin/activate"; set -u
}
check_dir() {
  [ -f "$ROOT/config/h1_joints.yaml" ] || die "Wrong repository layout: $ROOT"
}
