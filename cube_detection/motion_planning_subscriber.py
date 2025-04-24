#!/usr/bin/env python3
#
# motion_planning_subscriber.py
#
# Receives /cube/position (camera frame),
# optionally transforms into robot base frame,
# publishes both the position and its Euclidean distance.

import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float32


class MotionPlanningSubscriber(Node):
    def _init_(self):
        super()._init_("motion_planning_subscriber")

        self.create_subscription(Point, "/cube/position", self.cb_pos, 10)
        self.pub_pos = self.create_publisher(Point, "/motion_planning/cube_position", 10)
        self.pub_dist = self.create_publisher(Float32, "/cube/distance", 10)

        self.get_logger().info("MotionPlanningSubscriber started.")

    # ──────────────────────────────────────────────────────────────
    def cb_pos(self, msg: Point):
        if any(math.isnan(v) for v in (msg.x, msg.y, msg.z)) or msg.z < 0:
            self.get_logger().warn("Invalid cube position received; ignoring.")
            return

        # transform if necessary
        rx, ry, rz = self.camera_to_robot(msg.x, msg.y, msg.z)
        dist = math.sqrt(rx * 2 + ry * 2 + rz ** 2)

        # publish
        self.pub_pos.publish(Point(x=rx, y=ry, z=rz))
        self.pub_dist.publish(Float32(data=float(dist)))

        self.get_logger().info(
            f"Robot-frame cube ({rx:.3f}, {ry:.3f}, {rz:.3f}) m – "
            f"dist {dist:.3f} m"
        )

    # ──────────────────────────────────────────────────────────────
    def camera_to_robot(self, x_c: float, y_c: float, z_c: float):
        """
        Example: rotate 180° about Z and translate +0.2 m in Z.
        Adjust to your Kaya’s calibration, or simply return (x_c, y_c, z_c).
        """
        theta = math.pi
        Rz = np.array(
            [[math.cos(theta), -math.sin(theta), 0],
             [math.sin(theta),  math.cos(theta), 0],
             [0,                0,               1]],
            float,
        )
        vec = Rz @ np.array([x_c, y_c, z_c])
        vec[2] += 0.2
        return tuple(vec.tolist())


def main(args=None):
    rclpy.init(args=args)
    node = MotionPlanningSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if _name_ == "_main_":
    main()