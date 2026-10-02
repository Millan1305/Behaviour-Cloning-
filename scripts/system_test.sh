#!/usr/bin/env bash
# Checks the whole installation.  Run after ./setup.sh.   Exit code = number of failed checks.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
FAIL=0
chk() { if eval "$2" >/dev/null 2>&1; then echo -e "  \033[32mPASS\033[0m $1"; else echo -e "  \033[31mFAIL\033[0m $1"; FAIL=$((FAIL+1)); fi; }
warn_chk() { if eval "$2" >/dev/null 2>&1; then echo -e "  \033[32mPASS\033[0m $1"; else echo -e "  \033[33mWARN\033[0m $1 (optional)"; fi; }
cd "$ROOT"
. /etc/os-release
echo "== System";    chk "Ubuntu 24.04" '[ "$VERSION_ID" = "24.04" ]'
echo "== ROS 2";     chk "/opt/ros/jazzy installed" '[ -f /opt/ros/jazzy/setup.bash ]'
set +u; source /opt/ros/jazzy/setup.bash 2>/dev/null; [ -f install/setup.bash ] && source install/setup.bash; set -u
chk "ros2 CLI" 'command -v ros2'
for p in ros_gz_sim ros_gz_bridge gz_ros2_control controller_manager joint_state_broadcaster position_controllers robot_state_publisher; do
  chk "ROS package $p" "ros2 pkg prefix $p"; done
chk "robot_cloning package built/discovered" 'ros2 pkg prefix robot_cloning'
echo "== Gazebo";    chk "gz command" 'command -v gz'; chk "gz sim version" 'gz sim --versions'
echo "== H1";        chk "H1 URDF cloned" '[ -f third_party/unitree_ros/robots/h1_description/urdf/h1.urdf ]'
chk "Gazebo URDF generation" 'python3 scripts/make_h1_gz_urdf.py'
echo "== Python";    chk "venv exists" '[ -f .venv/bin/activate ]'
PY=.venv/bin/python; [ -x $PY ] || PY=python3
for m in numpy cv2 mediapipe torch rclpy yaml; do chk "python import $m" "$PY -c 'import $m'"; done
warn_chk "CUDA (optional)" "$PY -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'"
warn_chk "webcam /dev/video*" 'ls /dev/video*'
echo "== Data/model"; chk "sample dataset" 'ls data/sample/*.csv'
chk "trained model" '[ -f models/bc_model.pt ]'; chk "reference trajectories" '[ -f models/reference_trajectories.npz ]'
echo "== Software-only tests (no Gazebo needed)"
chk "pytest suite" "$PY -m pytest -q tests"
chk "BC inference self-check (all 4 actions)" "$PY scripts/run_inference.py --check"
echo "== Live ROS graph (only if ./run.sh is running)"
if ros2 node list 2>/dev/null | grep -q behavior_cloning; then
  chk "node /behavior_cloning" 'ros2 node list | grep -q behavior_cloning'
  chk "node /h1_controller" 'ros2 node list | grep -q h1_controller'
  chk "topic /robot_cloning/joint_targets" 'ros2 topic list | grep -q /robot_cloning/joint_targets'
  chk "topic /forward_position_controller/commands" 'ros2 topic list | grep -q /forward_position_controller/commands'
  chk "controllers active" 'ros2 control list_controllers | grep -q "forward_position_controller.*active"'
else echo "  (skipped: start ./run.sh in another terminal to test the live graph)"; fi
echo; [ $FAIL -eq 0 ] && echo "ALL CHECKS PASSED" || echo "$FAIL CHECK(S) FAILED"
exit $FAIL
