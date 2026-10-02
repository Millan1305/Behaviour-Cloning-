"""ROS 2 node: camera/video -> MediaPipe Pose -> landmarks + normalised features.

Publishes
  /robot_cloning/pose_landmarks  std_msgs/Float32MultiArray  (12 landmarks x [x,y,z,visibility] = 48)
  /robot_cloning/pose_features   std_msgs/Float32MultiArray  (12 features, see common.FEATURE_NAMES)
"""
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from .common import landmarks_to_features
from .vision import PoseSource, PoseSourceError


class PoseDetectorNode(Node):
    def __init__(self):
        super().__init__("pose_detector")
        self.declare_parameter("source", "camera")
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("video_path", "")
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("show_preview", False)
        self.declare_parameter("min_visibility", 0.5)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        try:
            self.src = PoseSource(g("source"), g("camera_index"), g("video_path"))
        except PoseSourceError as e:
            self.get_logger().error(f"Pose detector disabled: {e}")
            self.src = None
            return
        self.min_vis = float(g("min_visibility"))
        self.preview = bool(g("show_preview"))
        self.pub_lm = self.create_publisher(Float32MultiArray, "/robot_cloning/pose_landmarks", 10)
        self.pub_ft = self.create_publisher(Float32MultiArray, "/robot_cloning/pose_features", 10)
        self.create_timer(1.0 / float(g("rate_hz")), self.tick)
        self.get_logger().info("Pose detector started (MediaPipe, CPU). Publishing /robot_cloning/pose_features")

    def tick(self):
        lm, frame, ok = self.src.read()
        if not ok:
            self.get_logger().warn("No more frames from source", throttle_duration_sec=5.0)
            return
        if lm is not None and lm[:, 3].min() >= self.min_vis:
            self.pub_lm.publish(Float32MultiArray(data=[float(v) for v in lm.reshape(-1)]))
            self.pub_ft.publish(Float32MultiArray(data=[float(v) for v in landmarks_to_features(lm)]))
        elif lm is None:
            self.get_logger().info("No person detected", throttle_duration_sec=5.0)
        if self.preview and frame is not None:
            import cv2
            cv2.imshow("robot_cloning pose", self.src.draw(cv2.flip(frame, 1), "pose detector"))
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = PoseDetectorNode()
    try:
        if node.src is not None:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.src is not None:
            node.src.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
