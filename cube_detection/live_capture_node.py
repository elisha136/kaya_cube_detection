#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import pyrealsense2 as rs
import numpy as np
import cv2
import threading

from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Quaternion
from cv_bridge import CvBridge


class LiveCaptureNode(Node):
    """
    A ROS2 node that uses one RealSense pipeline for color, depth, gyro, and accel data.
    Frames are retrieved in a dedicated thread (via wait_for_frames()) and published
    on separate topics.

    Topics published:
      - /rgb_frame       (sensor_msgs/Image) Color images
      - /depth_frame     (sensor_msgs/Image) Depth images
      - /camera/imu      (sensor_msgs/Imu)    IMU data (gyro or accel)
    """

    def __init__(self):
        super().__init__('live_capture_node')
        self.get_logger().info("Initializing LiveCaptureNode...")

        # --- Publishers ---
        self.color_publisher = self.create_publisher(Image, 'rgb_frame', 10)
        self.depth_publisher = self.create_publisher(Image, 'depth_frame', 10)
        self.imu_publisher   = self.create_publisher(Imu, 'camera/imu', 10)

        # Use CvBridge for converting OpenCV images to ROS sensor_msgs/Image
        self.bridge = CvBridge()

        # --- Configure RealSense ---
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable color + depth streams
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Enable gyro + accel streams
        # By default, the D435i’s IMU can run at 200/400 Hz (gyro) and 63/250 Hz (accel).
        # If you want to force specific rates:
        #   self.config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
        #   self.config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63)
        self.config.enable_stream(rs.stream.gyro)
        self.config.enable_stream(rs.stream.accel)

        try:
            _profile = self.pipeline.start(self.config)
            self.get_logger().info("RealSense pipeline started successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to start RealSense pipeline: {e}")
            return

        # This event signals when we should stop the thread
        self.stop_event = threading.Event()
        # Start a dedicated thread to read framesets from the camera
        self.publish_thread = threading.Thread(target=self.frame_publisher_loop, daemon=True)
        self.publish_thread.start()

    def frame_publisher_loop(self):
        """
        Runs in a separate thread. Continuously calls wait_for_frames(), which returns
        a frameset containing whichever frames arrived since the last call.
        We iterate over each frame in the frameset (color, depth, gyro, accel)
        and publish them accordingly.
        """
        self.get_logger().info("Starting frame publisher loop in a separate thread...")

        while not self.stop_event.is_set():
            try:
                # Wait up to 500 ms for a frameset
                frameset = self.pipeline.wait_for_frames(timeout_ms=500)
            except RuntimeError as e:
                # Usually this means no frames arrived in time. Continue or handle error.
                self.get_logger().warn(f"No frames received in 500 ms: {e}")
                continue

            if not frameset:
                continue

            # Iterate over the frames in this frameset
            for frame in frameset:
                stream_type = frame.profile.stream_type()
                # --------------------------------------
                # COLOR FRAME
                # --------------------------------------
                if stream_type == rs.stream.color:
                    color_image = np.asanyarray(frame.get_data())
                    color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
                    color_msg.header.stamp = self.get_clock().now().to_msg()
                    color_msg.header.frame_id = 'camera_color_frame'
                    self.color_publisher.publish(color_msg)
                    self.get_logger().debug("Published color frame.")

                # --------------------------------------
                # DEPTH FRAME
                # --------------------------------------
                elif stream_type == rs.stream.depth:
                    depth_image = np.asanyarray(frame.get_data())
                    depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding='mono16')
                    depth_msg.header.stamp = self.get_clock().now().to_msg()
                    depth_msg.header.frame_id = 'camera_depth_frame'
                    self.depth_publisher.publish(depth_msg)
                    self.get_logger().debug("Published depth frame.")

                # --------------------------------------
                # IMU MOTION FRAMES (ACCEL OR GYRO)
                # --------------------------------------
                elif frame.is_motion_frame():
                    motion_frame = frame.as_motion_frame()
                    motion_data = motion_frame.get_motion_data()

                    # We'll publish each motion frame as an Imu message.
                    imu_msg = Imu()
                    imu_msg.header.stamp = self.get_clock().now().to_msg()
                    imu_msg.header.frame_id = 'camera_imu_frame'

                    # If it's gyro data
                    if stream_type == rs.stream.gyro:
                        imu_msg.angular_velocity.x = motion_data.x
                        imu_msg.angular_velocity.y = motion_data.y
                        imu_msg.angular_velocity.z = motion_data.z

                    # If it's accel data
                    elif stream_type == rs.stream.accel:
                        imu_msg.linear_acceleration.x = motion_data.x
                        imu_msg.linear_acceleration.y = motion_data.y
                        imu_msg.linear_acceleration.z = motion_data.z

                    # By default, set orientation to identity.
                    imu_msg.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

                    # Publish
                    self.imu_publisher.publish(imu_msg)
                    self.get_logger().debug(f"Published IMU data ({stream_type}).")

                # If you had other stream types (e.g., infrared), you could handle them here as well.

        self.get_logger().info("Exiting frame publisher loop...")

    def destroy_node(self):
        """
        Overridden to ensure we stop the pipeline and the background thread cleanly
        when the node is destroyed (e.g., on shutdown).
        """
        self.get_logger().info("Stopping RealSense pipeline...")
        self.stop_event.set()     # Signal the thread to stop
        if self.publish_thread.is_alive():
            self.publish_thread.join(timeout=2.0)
        self.pipeline.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LiveCaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
