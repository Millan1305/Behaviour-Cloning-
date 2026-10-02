"""ROS 2 node: keyboard / CLI action selector.

  1 stand   2 raise_hand   3 wave   4 bend   5 pose (mirror webcam)   q quit
Publishes /robot_cloning/action_cmd (std_msgs/String).
Non-interactive:  python3 -m robot_cloning.action_manager --action wave
"""
import argparse
import select
import sys
import time

import rclpy
from std_msgs.msg import String

from .common import ACTION_KEYS

KEYS = dict(ACTION_KEYS, **{"5": "pose"})


def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", help="publish one action and exit")
    ns, ros_args = ap.parse_known_args(sys.argv[1:] if args is None else args)
    rclpy.init(args=ros_args)
    node = rclpy.create_node("action_manager")
    pub = node.create_publisher(String, "/robot_cloning/action_cmd", 10)

    def send(a):
        for _ in range(5):                      # repeat: first message can be lost during discovery
            pub.publish(String(data=a))
            time.sleep(0.1)
        print(f"-> {a}", flush=True)

    if ns.action:
        time.sleep(0.5)
        send(ns.action)
        node.destroy_node()
        rclpy.shutdown()
        return
    if not sys.stdin.isatty():
        print("action_manager needs an interactive terminal (or use --action NAME).")
        return
    import termios
    import tty
    print("\nKeyboard control:  1=stand  2=raise_hand  3=wave  4=bend  5=pose(webcam mirror)  q=quit\n", flush=True)
    fd, old = sys.stdin.fileno(), termios.tcgetattr(sys.stdin.fileno())
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            if select.select([sys.stdin], [], [], 0.1)[0]:
                k = sys.stdin.read(1)
                if k == "q":
                    break
                if k in KEYS:
                    send(KEYS[k])
            rclpy.spin_once(node, timeout_sec=0)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
