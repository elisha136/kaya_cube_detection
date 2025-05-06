#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
from std_msgs.msg import Float32

import math
import numpy as np

class MotionPlanningSubscriber(Node):
    """
    A ROS2 node that subscribes to /cube/position (in the camera frame),
    applies a transform to the 'robot_base_frame' (if needed), calculates the
    Euclidean distance, and republishes both the 3D position and distance.
    """

    def __init__(self):
        super().__init__('motion_planning_subscriber')

        # Declare optional parameters
        self.declare_parameter('z_offset', 0.2)  # default: 20 cm above camera

        # Fetch parameter
        self.z_offset = self.get_parameter('z_offset').get_parameter_value().double_value

        # Subscribe to the cube's position in camera frame
        self.subscription = self.create_subscription(
            Point,
            '/cube/position',
            self.position_callback,
            10
        )

        # Publish the cube position in robot_base_frame
        self.publisher_pos = self.create_publisher(
            Point,
            '/motion_planning/cube_position',
            10
        )

        # Publish the distance (Float32) from the robot base
        self.publisher_dist = self.create_publisher(
            Float32,
            '/cube/distance',
            10
        )

        self.get_logger().info("MotionPlanningSubscriber node started.")

    def position_callback(self, msg: Point):
        """
        1. Transform the position from the camera frame to the robot base frame
        2. Compute distance from the robot base
        3. Publish both the new position and the distance
        """
        # Basic checks for invalid data
        if any(math.isnan(val) for val in [msg.x, msg.y, msg.z]):
            self.get_logger().warn("Received NaN position values; ignoring.")
            return
        if msg.z < 0:
            self.get_logger().warn(f"Received negative Z={msg.z:.2f}, ignoring as invalid.")
            return

        # Current position in camera frame
        cam_x = msg.x
        cam_y = msg.y
        cam_z = msg.z

        # 1. (Optional) transform to robot_base_frame
        rob_x, rob_y, rob_z = self.camera_to_robot_transform(cam_x, cam_y, cam_z)

        # 2. Compute distance from robot base (0,0,0) in robot frame
        distance = math.sqrt(rob_x**2 + rob_y**2 + rob_z**2)

        # 3. Publish the new position in the robot frame
        new_pos = Point(x=rob_x, y=rob_y, z=rob_z)
        self.publisher_pos.publish(new_pos)

        # 4. Publish the distance on /cube/distance
        dist_msg = Float32()
        dist_msg.data = float(distance)
        self.publisher_dist.publish(dist_msg)

        # Log it
        self.get_logger().info(
            f"Camera frame pos=({cam_x:.3f}, {cam_y:.3f}, {cam_z:.3f}) => "
            f"Robot frame pos=({rob_x:.3f}, {rob_y:.3f}, {rob_z:.3f}); "
            f"Distance={distance:.3f} m"
        )

    def camera_to_robot_transform(self, x_c: float, y_c: float, z_c: float):
        """
        Example transform from the camera's optical frame to the robot base frame.
        If the camera frame is already the robot base frame, just do:
            return (x_c, y_c, z_c)

        Otherwise, apply known rotation/translation. For example:
         - Rotate around Z by 180 deg
         - Translate +z_offset meters in Z
        """
        # Example rotation around Z by 180 deg
        theta = math.pi  # 180 deg
        Rz = np.array([
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta),  math.cos(theta), 0],
            [0,               0,               1]
        ], dtype=float)

        cam_vec = np.array([x_c, y_c, z_c], dtype=float).reshape(3,1)
        robot_vec = Rz @ cam_vec

        # Example translation: +z_offset m in Z
        robot_vec[2] += self.z_offset

        return (float(robot_vec[0]), float(robot_vec[1]), float(robot_vec[2]))

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
