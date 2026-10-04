# Humanoid Robot Behavior Cloning Using ROS 2 and Gazebo

Supervised **Behavior Cloning**: human pose (MediaPipe) -> PyTorch network -> Unitree H1 joint targets -> ROS 2 / ros2_control -> Gazebo Harmonic.
Target: **Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic, CPU only (no GPU needed)**.

> **Verification status (read this first).** The repository was written in an offline sandbox without ROS, Gazebo, PyTorch or a webcam.
> Verified there: data generation, feature extraction, retargeting, preprocessing, controller logic, URDF-to-Gazebo conversion, shell syntax.
> **NOT yet verified on a real machine:** `./setup.sh`, colcon build, Gazebo launch, H1 spawn/motion, MediaPipe, PyTorch training.
> Run `./scripts/system_test.sh` after setup; see *Known limitations* and *Troubleshooting*.

## QUICK START — COPY AND PASTE

Fresh Ubuntu 24.04, current directory: `~` (replace `<YOUR_REPO_URL>` with the URL where you uploaded this repo).

```bash
sudo apt-get update && sudo apt-get install -y git curl
cd ~
git clone <YOUR_REPO_URL> robot_cloning
cd ~/robot_cloning
chmod +x setup.sh && ./setup.sh          # installs everything, clones H1, builds, trains first model
```
When it prints `SETUP COMPLETE`:
```bash
cd ~/robot_cloning
./run.sh                                  # Gazebo + H1 + BC; press 1/2/3/4 to trigger learned behaviours, q quits
```
New terminals: `source /opt/ros/jazzy/setup.bash && source ~/robot_cloning/install/setup.bash` (`run.sh` does this itself).

## Architecture

```
 Webcam/video -> pose_detector (MediaPipe) -> /robot_cloning/pose_features ─┐
                                           -> /robot_cloning/pose_landmarks -> demo_recorder -> data/demonstrations/*.csv
 keyboard -> action_manager -> /robot_cloning/action_cmd ─┐                                        │ preprocess + train
                                                          v                                        v
                                   behavior_cloning (PyTorch BCNet, models/bc_model.pt) <── models/
                                                          │ /robot_cloning/joint_targets
                                                          v
                          h1_controller (sign/limits/smoothing) ── /forward_position_controller/commands
                                                          v
                         ros2_control (gz_ros2_control) in Gazebo Harmonic  ->  Unitree H1
```
## 🎥 Behavior Cloning Demo

The following demonstration shows the complete behavior cloning pipeline:

<table>
<tr>
<th align="center">👤 Human Demonstration</th>
<th align="center">🤖 Unitree H1 Robot</th>
</tr>

<tr>
<td align="center">

<a href="./my_clip.mp4">
<img src="./human_demo.gif" width="400">
</a>

<br>

<a href="./my_clip.mp4">▶️ Watch Human Demo</a>

</td>

<td align="center">

<a href="./robot_copy.mp4">
<img src="./robot_demo.gif" width="400">
</a>

<br>

<a href="./robot_copy.mp4">▶️ Watch Robot Demo</a>

</td>
</tr>

<tr>
<td align="center">
Human performs the action
</td>
<td align="center">
H1 copies the demonstrated action
</td>
</tr>
</table>



### Pipeline

Human Action → MediaPipe Pose Detection → PyTorch Behavior Cloning → Joint Retargeting → ROS 2 → Gazebo → Unitree H1

**What is learned:** `BCNet` (MLP, 12 pose features -> 8 robot joint targets + action-class head) trained with MSE + cross-entropy on (pose, robot-target) pairs, validated on held-out *episodes*.
The robot-target labels come from kinematic retargeting of the demonstrated human pose (`common.retarget`).
**Two modes:** *action mode* (keys 1-4) feeds a recorded demonstration's pose sequence through the network; *pose mode* (key 5, needs camera) feeds your live pose through the network so H1 mirrors you.

## Requirements
Hardware: x86_64 PC, 8 GB RAM+, a webcam (optional; video-file fallback exists), a GPU is **not** required (OpenGL-capable for the Gazebo GUI; `--headless` otherwise).
Software (installed by `install_dependencies.sh`): Ubuntu 24.04, ROS 2 Jazzy (`ros-jazzy-ros-base`), Gazebo Harmonic (via `ros-jazzy-ros-gz`), `ros-jazzy-gz-ros2-control`, `ros-jazzy-ros2-control(-ers)`, colcon, rosdep, Python 3.12, numpy 1.26.4, torch 2.5.1 (CPU), mediapipe 0.10.21 (brings OpenCV). Version pins are untested choices.

External repository (fetched by `scripts/fetch_h1.sh`): `https://github.com/unitreerobotics/unitree_ros` — only `robots/h1_description` (URDF + meshes), because it is the official Unitree H1 model. The commit is recorded in `third_party/H1_COMMIT.txt` on the first fetch (commit that file to freeze it; override with `H1_REF=<sha|tag>`). I could not look up a known-good commit offline.

## Install / Setup / Build (separately)
```bash
cd ~/robot_cloning
./install_dependencies.sh      # apt (only missing pkgs, no upgrade), .venv, pip
./scripts/fetch_h1.sh          # H1 URDF + meshes
./build.sh                     # rosdep + colcon build --symlink-install
```
Python venv: `.venv` is created with `--system-site-packages` because `rclpy` comes from apt for the system Python and cannot be pip-installed. Manual use: `source ~/robot_cloning/.venv/bin/activate`.

## Run
```bash
cd ~/robot_cloning
./run.sh                    # keyboard control
./run.sh --camera           # + webcam pose node (key 5 = mirror you)
./run.sh --video clip.mp4   # pose from a video file (no webcam)
./run.sh --headless         # no Gazebo GUI
```
Keys: `1` stand · `2` raise_hand · `3` wave · `4` bend · `5` pose-mirror · `q` quit. Without the keyboard node: `ros2 topic pub --once /robot_cloning/action_cmd std_msgs/msg/String "{data: wave}"`.
Launch file directly: `ros2 launch robot_cloning behavior_cloning_demo.launch.py` (needs `export ROBOT_CLONING_ROOT=~/robot_cloning`).

## Camera setup / dataset collection / training / inference
```bash
cd ~/robot_cloning
python3 scripts/test_camera.py            # inside .venv; stand 2-3 m away, full body visible, good light
./record_demo.sh wave --episodes 5        # 3 s countdown, 4 s each; repeat for stand, raise_hand, bend
./record_demo.sh wave --video my_wave.mp4 # no webcam
./train.sh                                # sample + your demos;  ./train.sh --sources user  = only yours
python3 scripts/run_inference.py --action wave   # ROS-free inference printout
python3 scripts/run_inference.py --check         # software-only test of all 4 actions
./run.sh
```
Perform the action with your **right** arm (the sample data does). Start/end in a relaxed standing pose. Your own demos are used for the keyboard-mode reference trajectories if present.
Dataset (`data/demonstrations/*.csv`, `data/sample/*.csv`): `episode,timestamp,action,lm{0..11}_{x,y,z,v}`; processed into `data/processed/dataset.npz` (features, robot targets, labels, episode ids).

**The bundled sample data is SYNTHETIC** (`scripts/generate_sample_data.py`, parametric stick-figure human, same pipeline as real data). It lets the project run before you record anything; it is not real human motion.

## Available actions / ROS topics
`stand`, `raise_hand`, `wave`, `bend`. Topics: see `docs/TOPICS.md`. Nodes: `pose_detector`, `demo_recorder`, `behavior_cloning`, `h1_controller`, `action_manager`.

## Gazebo usage
`run.sh` starts Gazebo (`empty.sdf`) and spawns H1 from `generated/h1_gz.urdf` (made at launch from the Unitree URDF by `urdf_tools.py`: ROS1 tags stripped, mesh URIs made absolute, `ros2_control` + `gz_ros2_control` added, pelvis fixed to the world at 1.05 m). Logs: `log/demo_launch.log`.
Under VMs/WSL: `export LIBGL_ALWAYS_SOFTWARE=1` or use `--headless`.

## Troubleshooting
```bash
cd ~/robot_cloning && ./scripts/system_test.sh           # full diagnostic
tail -n 80 log/demo_launch.log                           # launch errors
ros2 control list_controllers                            # want: joint_state_broadcaster + forward_position_controller active
ros2 topic echo /robot_cloning/joint_targets --once      # is the network producing output?
ros2 topic hz /forward_position_controller/commands      # ~50 Hz
ls /dev/video*; groups | grep video                      # webcam
gz sim --versions                                        # Gazebo
```
- *"Workspace not built"* -> `./build.sh`. *"ROS not sourced"* -> `source /opt/ros/jazzy/setup.bash`. *"model not found"* -> `./train.sh`.
- *Limb moves the wrong way* -> flip `sign` in `config/h1_joints.yaml` (no retraining).
- *Spawner timeout / robot not appearing* -> check `log/demo_launch.log`; plugin name/URDF issues are the most likely first-run problems (see limitations).
- *`ModuleNotFoundError: torch/mediapipe`* -> `source .venv/bin/activate` or re-run `./install_dependencies.sh`.
- Optional NVIDIA GPU: not used or required (CPU torch). To try CUDA, replace the torch wheel in `.venv` with the CUDA build from pytorch.org.

## Project structure
`src/robot_cloning/` ROS 2 package (nodes, `common.py`, `dataset.py`, `model.py`, `h1_logic.py`, `urdf_tools.py`, launch) · `scripts/` CLI tools · `config/h1_joints.yaml` · `data/` · `models/` · `tests/` · `docs/` · `third_party/` (fetched H1) · `generated/` (URDF/controller YAML, auto-made).

## Known limitations (honest list)
1. **Untested end-to-end on Gazebo/Jazzy** — see the verification note. Expect to debug first-run issues (URDF/plugin/joint sign/axis).
2. **H1 is fixed in the air at the pelvis.** Free-standing bipedal balance needs a locomotion/balance controller, which is out of scope. Consequently the legs/hips do not support the robot.
3. **`bend` = hip + knee flexion.** H1 has no torso pitch joint, and with a fixed pelvis hip flexion swings the legs forward rather than tilting the torso.
4. H1 joint names/limits/signs in `config/` are from my memory of the standard H1 URDF; limits are overridden by the real URDF at launch, signs may need flipping.
5. Labels are produced by kinematic retargeting, so the network learns a smooth approximation of that mapping plus action classification (no teleoperated robot data).
6. Pose mode uses MediaPipe *world* landmarks assuming y-down/x-right axes; verify with `scripts/test_camera.py`.
7. Sample data is synthetic; `stand` and rest frames of other actions are visually identical, so action-classification accuracy < 100% is expected.

## Future improvements
More joints (torso yaw, shoulder pitch/yaw, ankles), temporal models (windowed/GRU), a balance controller + free-standing H1, real teleoperation data, Gazebo-side camera input.
