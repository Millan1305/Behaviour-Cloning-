#!/usr/bin/env bash
# Start Gazebo + Unitree H1 + ROS 2 nodes + Behavior Cloning, then keyboard control.
#   ./run.sh                    keyboard-triggered learned behaviours
#   ./run.sh --camera           also start webcam pose detector (press 5 = mirror your pose)
#   ./run.sh --video clip.mp4   pose detector on a pre-recorded video (no webcam)
#   ./run.sh --headless         Gazebo server without GUI
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/common.sh"
check_dir
CAM=false; VIDEO=""; HEADLESS=false
while [ $# -gt 0 ]; do
  case "$1" in
    --camera) CAM=true;; --video) CAM=true; VIDEO="${2:?--video needs a path}"; shift;;
    --headless) HEADLESS=true;; -h|--help) sed -n '2,8p' "$0"; exit 0;;
    *) die "unknown option $1";;
  esac; shift
done
ros_source; ws_source
[ -f "$ROOT/third_party/unitree_ros/robots/h1_description/urdf/h1.urdf" ] || die "H1 description missing. Run ./scripts/fetch_h1.sh (or ./setup.sh)"
command -v gz >/dev/null || die "Gazebo ('gz') not found. Run ./install_dependencies.sh"
if [ ! -f "$ROOT/models/bc_model.pt" ] || [ ! -f "$ROOT/models/reference_trajectories.npz" ]; then
  info "No trained model found -> training on sample data first"; "$ROOT/train.sh"
fi
if [ "$HEADLESS" = false ] && [ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
  echo "WARNING: no display detected; using --headless. (Gazebo GUI needs a desktop session.)"; HEADLESS=true
fi
if [ "$CAM" = true ] && [ -z "$VIDEO" ] && ! ls /dev/video* >/dev/null 2>&1; then
  echo "WARNING: no webcam (/dev/video*) found -> camera disabled. Use --video file.mp4, or keyboard control."; CAM=false
fi
mkdir -p "$ROOT/log"; LOG="$ROOT/log/demo_launch.log"
info "Launching (log: $LOG) ..."
LARGS=(use_camera:=$CAM headless:=$HEADLESS); [ -n "$VIDEO" ] && LARGS+=("video:=$VIDEO")
setsid ros2 launch robot_cloning behavior_cloning_demo.launch.py "${LARGS[@]}" >"$LOG" 2>&1 &
LPID=$!
cleanup() { trap - EXIT INT TERM; kill -INT -- -$LPID 2>/dev/null || true; sleep 3; kill -TERM -- -$LPID 2>/dev/null || true; pkill -f "gz sim" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

ready=0
for i in $(seq 1 90); do
  kill -0 $LPID 2>/dev/null || { tail -n 30 "$LOG"; die "launch exited early - see $LOG"; }
  if ros2 control list_controllers 2>/dev/null | grep -q "forward_position_controller.*active"; then ready=1; break; fi
  sleep 2
done
[ $ready = 1 ] || { tail -n 30 "$LOG"; die "H1 controllers did not become active within 180 s (Gazebo/H1 spawn problem). See $LOG and README Troubleshooting."; }
echo "  [x] Gazebo started          ($(pgrep -f 'gz sim' >/dev/null && echo running))"
echo "  [x] H1 spawned + ros2_control controllers active"
grep -q "Behavior Cloning model loaded" "$LOG" && echo "  [x] Behavior Cloning model loaded" || echo "  [ ] BC model status unknown (see $LOG)"
echo "  [x] ROS 2 nodes: $(ros2 node list 2>/dev/null | tr '\n' ' ')"
echo "  Available actions: stand, raise_hand, wave, bend   (5 = pose mirror, needs --camera/--video)"
PYTHONPATH="$ROOT/src/robot_cloning:${PYTHONPATH:-}" python3 -m robot_cloning.action_manager
