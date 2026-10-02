from setuptools import find_packages, setup

package_name = "robot_cloning"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/behavior_cloning_demo.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="robot_cloning",
    maintainer_email="maintainer@example.com",
    description="Humanoid behavior cloning on ROS 2 Jazzy + Gazebo Harmonic",
    license="MIT",
    entry_points={
        "console_scripts": [
            "pose_detector = robot_cloning.pose_detector:main",
            "demo_recorder = robot_cloning.demo_recorder:main",
            "behavior_cloning = robot_cloning.behavior_cloning:main",
            "h1_controller = robot_cloning.h1_controller:main",
            "action_manager = robot_cloning.action_manager:main",
        ],
    },
)
