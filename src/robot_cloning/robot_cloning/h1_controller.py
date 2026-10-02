"""ROS 2 node: H1 controller bridge.

Subscribes /robot_cloning/joint_targets (canonical BC joints, JointState), maps them onto H1 joints
(config/h1_joints.yaml), clamps to URDF limits, smooths + rate-limits, and publishes the full joint
vector to the ros2_control position controller running inside Gazebo:
  /forward_position_controller/commands  (std_msgs/Float64MultiArray)
"""
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from . import paths
from .h1_logic import load_mapper


class H1Controller(Node):
    def __init__(self):
        super().__init__("h1_controller")
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("command_topic", "/forward_position_controller/commands")
        self.mapper = load_mapper(paths.root())
        self.pub = self.create_publisher(Float64MultiArray, self.get_parameter("command_topic").value, 10)
        self.create_subscription(JointState, "/robot_cloning/joint_targets", self.on_targets, 10)
        rate = float(self.get_parameter("rate_hz").value)
        self.dt, self.last = 1.0 / rate, time.time()
        self.create_timer(self.dt, self.tick)
        self.get_logger().info(f"H1 controller ready: {len(self.mapper.joint_order)} joints -> "
                               f"{self.get_parameter('command_topic').value}")
        self.warned = False

    def on_targets(self, msg):
        unknown = self.mapper.set_canonical_targets(msg.name, msg.position)
        if unknown and not self.warned:
            self.get_logger().warn(f"Ignoring joints without mapping in config/h1_joints.yaml: {unknown}")
            self.warned = True

    def tick(self):
        now = time.time()
        dt, self.last = min(now - self.last, 0.1), now
        self.pub.publish(Float64MultiArray(data=[float(v) for v in self.mapper.step(dt)]))


def main(args=None):
    rclpy.init(args=args)
    node = H1Controller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
