"""Gazebo Harmonic + Unitree H1 + ros2_control + Behavior Cloning nodes.

ros2 launch robot_cloning behavior_cloning_demo.launch.py [use_camera:=true] [video:=/path.mp4] [headless:=true]
"""
import os
import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, LogInfo,
                            OpaqueFunction, RegisterEventHandler)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _root():
    env = os.environ.get("ROBOT_CLONING_ROOT")
    if env and (Path(env) / "config" / "h1_joints.yaml").exists():
        return Path(env)
    for p in Path(__file__).resolve().parents:
        if (p / "config" / "h1_joints.yaml").exists():
            return p
    raise RuntimeError("Set ROBOT_CLONING_ROOT=~/robot_cloning (repo root) before launching.")


def _launch_setup(context):
    root = _root()
    sys.path.insert(0, str(root / "src" / "robot_cloning"))
    from robot_cloning.h1_logic import load_yaml
    from robot_cloning.urdf_tools import build_gz_urdf

    h1_dir = root / "third_party" / "unitree_ros" / "robots" / "h1_description"
    cfg = load_yaml(root / "config" / "h1_joints.yaml")
    if not (h1_dir / "urdf" / "h1.urdf").exists():
        raise RuntimeError(f"H1 description not found at {h1_dir}. Run ./setup.sh (or ./scripts/fetch_h1.sh) first.")
    gen = build_gz_urdf(h1_dir / "urdf" / "h1.urdf", root / "generated", h1_dir, cfg.get("pelvis_height", 1.05))
    urdf_text = Path(gen["urdf"]).read_text()

    venv_py = root / ".venv" / "bin" / "python"
    py = str(venv_py) if venv_py.exists() else "python3"
    env = dict(os.environ, ROBOT_CLONING_ROOT=str(root),
               PYTHONPATH=f"{root / 'src' / 'robot_cloning'}:{os.environ.get('PYTHONPATH', '')}")

    def pynode(module, params=None):
        cmd = [py, "-u", "-m", f"robot_cloning.{module}"]
        if params:
            cmd += ["--ros-args"] + [x for k, v in params.items() for x in ("-p", f"{k}:={v}")]
        return ExecuteProcess(cmd=cmd, output="screen", additional_env=env, name=module)

    gz_args = "-r -v2 empty.sdf" + (" -s" if LaunchConfiguration("headless").perform(context) == "true" else "")
    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": gz_args}.items())
    rsp = Node(package="robot_state_publisher", executable="robot_state_publisher", output="screen",
               parameters=[{"robot_description": urdf_text}])
    spawn = Node(package="ros_gz_sim", executable="create", output="screen",
                 arguments=["-topic", "robot_description", "-name", "h1"])
    jsb = Node(package="controller_manager", executable="spawner", output="screen",
               arguments=["joint_state_broadcaster", "--controller-manager-timeout", "120"])
    fpc = Node(package="controller_manager", executable="spawner", output="screen",
               arguments=["forward_position_controller", "--controller-manager-timeout", "120"])

    actions = [
        LogInfo(msg=f"[robot_cloning] H1 URDF generated: {gen['urdf']} ({len(gen['joint_order'])} joints, base '{gen['base_link']}')"),
        gz, rsp, spawn,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb, fpc])),
        pynode("behavior_cloning", {"model_path": str(root / "models" / "bc_model.pt")}),
        pynode("h1_controller"),
    ]
    if LaunchConfiguration("use_camera").perform(context) == "true":
        video = LaunchConfiguration("video").perform(context)
        params = {"source": "video" if video else "camera",
                  "camera_index": LaunchConfiguration("camera_index").perform(context),
                  "show_preview": LaunchConfiguration("show_preview").perform(context)}
        if video:
            params["video_path"] = video
        actions.append(pynode("pose_detector", params))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_camera", default_value="false", description="start webcam/video pose detector"),
        DeclareLaunchArgument("video", default_value="", description="pre-recorded video instead of webcam"),
        DeclareLaunchArgument("camera_index", default_value="0"),
        DeclareLaunchArgument("show_preview", default_value="false"),
        DeclareLaunchArgument("headless", default_value="false", description="run Gazebo server without GUI"),
        OpaqueFunction(function=_launch_setup),
    ])
