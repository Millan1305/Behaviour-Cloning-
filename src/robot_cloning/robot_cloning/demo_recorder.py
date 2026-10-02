"""ROS 2 node: records /robot_cloning/pose_landmarks for N seconds into data/demonstrations/<action>_<ts>.csv.

  ros2 run robot_cloning demo_recorder --ros-args -p action:=wave -p duration:=4.0
(needs the pose_detector node running).  The standalone alternative is ./record_demo.sh <action>.
"""
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from . import paths
from .common import ACTIONS, N_LM
from .dataset import write_demo_csv


class DemoRecorder(Node):
    def __init__(self):
        super().__init__("demo_recorder")
        self.declare_parameter("action", "wave")
        self.declare_parameter("duration", 4.0)
        self.declare_parameter("countdown", 3.0)
        self.declare_parameter("output_dir", "")
        self.action = self.get_parameter("action").value
        if self.action not in ACTIONS:
            raise SystemExit(f"Unknown action '{self.action}'. Valid: {ACTIONS}")
        self.duration = float(self.get_parameter("duration").value)
        out = self.get_parameter("output_dir").value
        self.out_dir = Path(out) if out else paths.root() / "data" / "demonstrations"
        self.t0 = time.time() + float(self.get_parameter("countdown").value)
        self.ts, self.lm, self.done = [], [], False
        self.create_subscription(Float32MultiArray, "/robot_cloning/pose_landmarks", self.cb, 10)
        self.create_timer(0.5, self.status)
        self.get_logger().info(f"Recording '{self.action}' for {self.duration}s after countdown - get ready!")

    def status(self):
        left = self.t0 - time.time()
        if left > 0:
            self.get_logger().info(f"Starting in {left:.1f}s ...")
        elif not self.done and not self.ts:
            self.get_logger().warn("Recording but no landmarks received - is pose_detector running and seeing you?")

    def cb(self, msg):
        now = time.time()
        if now < self.t0 or self.done:
            return
        self.ts.append(now - self.t0)
        self.lm.append(np.array(msg.data, dtype=float).reshape(N_LM, 4))
        if now - self.t0 >= self.duration:
            self.done = True
            path = self.out_dir / f"{self.action}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            write_demo_csv(path, [dict(episode=0, action=self.action, t=np.array(self.ts), lm=np.stack(self.lm))])
            self.get_logger().info(f"Saved {len(self.ts)} frames to {path}")


def main(args=None):
    rclpy.init(args=args)
    node = DemoRecorder()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
