#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Pose
from std_msgs.msg import Float32

from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException, TransformException
from tf2_geometry_msgs import PointStamped, do_transform_point

import math


class MotionPlanningSubscriber(Node):
    """ROS 2 node that transforms cube position, computes distance, and publishes both.

    Subscribes to /cube/position (or user-defined topic), transforms the position
    to the robot base frame (using tf2), calculates distance to the origin,
    and republishes both position and distance.

    Attributes:
        tf_buffer (Buffer): Buffer for storing transforms.
        tf_listener (TransformListener): Listener to fill the buffer.
    """

    def __init__(self):
        """Initializes the MotionPlanningSubscriber node."""
        super().__init__('motion_planning_subscriber')

        # Declare configurable parameters with defaults
        self.declare_parameter('input_topic', '/cube/position')
        self.declare_parameter('output_position_topic', '/motion_planning/cube_position')
        self.declare_parameter('output_distance_topic', '/cube/distance')
        self.declare_parameter('source_frame', 'camera_frame')
        self.declare_parameter('target_frame', 'robot_base_frame')

        # Retrieve parameters
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.output_position_topic = self.get_parameter('output_position_topic').get_parameter_value().string_value
        self.output_distance_topic = self.get_parameter('output_distance_topic').get_parameter_value().string_value
        self.source_frame = self.get_parameter('source_frame').get_parameter_value().string_value
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value

        # Set up tf2 listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Set up subscriptions and publishers
        self.subscription = self.create_subscription(Point, self.input_topic, self.position_callback, 10)
        self.publisher_pos = self.create_publisher(Pose, self.output_position_topic, 10)
        self.publisher_dist = self.create_publisher(Float32, self.output_distance_topic, 10)

        self.get_logger().info(f"MotionPlanningSubscriber started. Listening on {self.input_topic}.")

    def position_callback(self, msg: Point):
        """Handles incoming position messages, transforms them, computes distance, and publishes results.

        Args:
            msg (Point): Incoming position message in the source frame.
        """
        # Validate incoming data
        if any(math.isnan(val) for val in [msg.x, msg.y, msg.z]):
            self.get_logger().warn("Received NaN position values; ignoring.")
            return
        if msg.z < 0:
            self.get_logger().warn(f"Received negative Z={msg.z:.2f}, ignoring as invalid.")
            return

        # Wrap point in PointStamped with source frame
        stamped_point = PointStamped()
        stamped_point.header.frame_id = self.source_frame
        stamped_point.point = msg

        try:
            # Lookup transform from source to target frame
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                rclpy.time.Time()
            )

            # Transform point to target frame
            transformed_point = do_transform_point(stamped_point, transform).point

        except (LookupException, ExtrapolationException, TransformException) as e:
            self.get_logger().warn(f"Transform lookup failed: {e}")
            return

        # Compute distance to origin in target frame
        distance = math.sqrt(
            transformed_point.x ** 2 +
            transformed_point.y ** 2 +
            transformed_point.z ** 2
        )

        # Create Pose with identity orientation (no rotation)
        pose_msg = Pose()
        pose_msg.position = transformed_point
        pose_msg.orientation.w = 1.0  # Identity quaternion

        # Publish transformed pose and distance
        self.publisher_pos.publish(pose_msg)
        self.publisher_dist.publish(Float32(data=float(distance)))

        # Log results
        self.get_logger().info(
            f"Transformed position: ({transformed_point.x:.3f}, {transformed_point.y:.3f}, {transformed_point.z:.3f}) "
            f"in '{self.target_frame}'; Distance={distance:.3f} m"
        )


def main(args=None):
    """Entry point for the ROS 2 node."""
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

