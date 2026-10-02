# ROS 2 interfaces

| Topic | Type | Publisher -> Subscriber | Meaning |
|---|---|---|---|
| `/robot_cloning/pose_landmarks` | `std_msgs/Float32MultiArray` (48) | pose_detector -> demo_recorder | 12 landmarks x [x,y,z,visibility] |
| `/robot_cloning/pose_features` | `std_msgs/Float32MultiArray` (12) | pose_detector -> behavior_cloning | joint angles etc. (`common.FEATURE_NAMES`) |
| `/robot_cloning/action_cmd` | `std_msgs/String` | action_manager -> behavior_cloning | `stand`, `raise_hand`, `wave`, `bend`, `pose` |
| `/robot_cloning/joint_targets` | `sensor_msgs/JointState` | behavior_cloning -> h1_controller | 8 canonical joints predicted by the network |
| `/robot_cloning/predicted_action` | `std_msgs/String` | behavior_cloning | action-classification head output |
| `/robot_cloning/status` | `std_msgs/String` | behavior_cloning | mode/action status |
| `/forward_position_controller/commands` | `std_msgs/Float64MultiArray` (all H1 joints) | h1_controller -> ros2_control in Gazebo | position commands |
| `/joint_states` | `sensor_msgs/JointState` | joint_state_broadcaster | simulated H1 state |

Manual trigger: `ros2 topic pub --once /robot_cloning/action_cmd std_msgs/msg/String "{data: wave}"`
