#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import math

class MotionPlanningSubscriber(Node):
    """
    A ROS2 node that subscribes to /cube/position and republishes
    the same position to /motion_planning/cube_position.
    
    If you want to filter or transform the position data before
    sending it on to motion planning, you can do so here.
    """

    def __init__(self):
        super().__init__('motion_planning_subscriber')

        # Subscribe to the cube's position
        self.subscription = self.create_subscription(
            Point,
            '/cube/position',
            self.position_callback,
            10
        )

        # Publisher to forward the cube's position to the motion planning node
        self.publisher = self.create_publisher(
            Point,
            '/motion_planning/cube_position',
            10
        )

        self.get_logger().info("MotionPlanningSubscriber node started.")

    def position_callback(self, msg: Point):
        """
        Called whenever a new position is received on /cube/position.
        We pass it along to /motion_planning/cube_position (possibly after
        some validation checks).
        """

        # Example check: ignore NaN or extreme values
        if any(math.isnan(val) for val in [msg.x, msg.y, msg.z]):
            self.get_logger().warn("Received NaN position values; ignoring.")
            return

        # If desired, you can clamp or reject obviously invalid data, e.g. negative distance:
        if msg.z < 0:
            self.get_logger().warn(f"Received negative Z={msg.z:.2f}, ignoring as invalid.")
            return

        # Log the received cube position
        self.get_logger().info(
            f"Received cube position: x={msg.x:.3f}, y={msg.y:.3f}, z={msg.z:.3f}"
        )

        # Forward the cube's position to /motion_planning/cube_position
        self.publisher.publish(msg)
        self.get_logger().info("Republished cube position to /motion_planning/cube_position")

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

if __name__ == '__main__':
    main()
