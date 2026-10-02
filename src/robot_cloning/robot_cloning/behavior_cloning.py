"""ROS 2 node: Behavior Cloning inference.

Pose features --(trained PyTorch network)--> robot joint targets.

Two input modes (switch with /robot_cloning/action_cmd):
  action mode (default): a recorded demonstration's pose-feature sequence for the chosen action
                         (stand/raise_hand/wave/bend) is fed through the NETWORK; the network's
                         output drives the robot.  Nothing is hard-coded: change the data/model and
                         the robot behaviour changes.
  pose mode ("pose"):    live pose features from /robot_cloning/pose_features (webcam) are fed
                         through the network (the robot mirrors you).

Subscribes: /robot_cloning/action_cmd (String), /robot_cloning/pose_features (Float32MultiArray)
Publishes : /robot_cloning/joint_targets (sensor_msgs/JointState), /robot_cloning/predicted_action (String),
            /robot_cloning/status (String)
"""
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, String

from . import paths
from .common import ACTIONS


class BehaviorCloningNode(Node):
    def __init__(self):
        super().__init__("behavior_cloning")
        self.declare_parameter("model_path", "")
        self.declare_parameter("reference_path", "")
        self.declare_parameter("initial_mode", "stand")
        self.declare_parameter("loop_actions", False)
        self.declare_parameter("pose_timeout", 0.7)
        mp_ = self.get_parameter("model_path").value or str(paths.default_model_path())
        rp_ = self.get_parameter("reference_path").value or str(paths.default_reference_path())
        try:
            from .model import BehaviorCloningPolicy
            self.policy = BehaviorCloningPolicy(mp_)
        except FileNotFoundError:
            self.get_logger().fatal(f"Model not found: {mp_}. Train one with ./train.sh")
            raise SystemExit(1)
        except ImportError as e:
            self.get_logger().fatal(f"PyTorch not importable ({e}). Use the project venv (./install_dependencies.sh).")
            raise SystemExit(1)
        try:
            d = np.load(rp_)
        except FileNotFoundError:
            self.get_logger().fatal(f"Reference trajectories not found: {rp_}. Run ./train.sh")
            raise SystemExit(1)
        self.ref = {a: (d[f"ref_{a}"], float(d[f"dt_{a}"])) for a in ACTIONS if f"ref_{a}" in d.files}
        m = self.policy.meta
        self.get_logger().info(
            f"Behavior Cloning model loaded: {mp_} (val joint RMSE {m.get('val_joint_rmse_rad', float('nan')):.3f} rad, "
            f"val action acc {m.get('val_action_acc', float('nan')) * 100:.1f}%). Available actions: {list(self.ref)}")
        self.pub_j = self.create_publisher(JointState, "/robot_cloning/joint_targets", 10)
        self.pub_a = self.create_publisher(String, "/robot_cloning/predicted_action", 10)
        self.pub_s = self.create_publisher(String, "/robot_cloning/status", 10)
        self.create_subscription(String, "/robot_cloning/action_cmd", self.on_cmd, 10)
        self.create_subscription(Float32MultiArray, "/robot_cloning/pose_features", self.on_pose, 10)
        self.mode, self.action, self.k, self.t_next = "action", "stand", 0, 0.0
        self.last_pose_t, self.last_pose = 0.0, None
        self.set_mode(self.get_parameter("initial_mode").value)
        self.create_timer(0.005, self.tick)

    def set_mode(self, cmd):
        cmd = cmd.strip()
        if cmd == "pose":
            self.mode = "pose"
            self.status("mode=pose (mirroring live webcam pose)")
        elif cmd in self.ref:
            self.mode, self.action, self.k, self.t_next = "action", cmd, 0, 0.0
            self.status(f"mode=action action={cmd}")
        else:
            self.get_logger().warn(f"Unknown command '{cmd}'. Valid: {list(self.ref) + ['pose']}")

    def status(self, s):
        self.get_logger().info(s)
        self.pub_s.publish(String(data=s))

    def on_cmd(self, msg):
        self.set_mode(msg.data)

    def on_pose(self, msg):
        self.last_pose, self.last_pose_t = np.array(msg.data, dtype=np.float32), time.time()
        if self.mode == "pose":
            self.infer(self.last_pose)

    def infer(self, feats):
        joints, act, _ = self.policy.predict(feats)
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(self.policy.joint_names)
        js.position = [float(v) for v in joints]
        self.pub_j.publish(js)
        self.pub_a.publish(String(data=act))

    def tick(self):
        if self.mode != "action":
            return
        now = time.time()
        if now < self.t_next:
            return
        seq, dt = self.ref[self.action]
        self.infer(seq[self.k])
        self.t_next = now + dt
        self.k += 1
        if self.k >= len(seq):
            if self.action == "stand" or self.get_parameter("loop_actions").value:
                self.k = 0
            else:
                self.status(f"action '{self.action}' finished -> stand")
                self.set_mode("stand")


def main(args=None):
    rclpy.init(args=args)
    try:
        node = BehaviorCloningNode()
    except SystemExit:
        if rclpy.ok():
            rclpy.shutdown()
        raise
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
