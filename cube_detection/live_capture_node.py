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
    A ROS2 node that uses one RealSense pipeline for color and depth.
    (No IMU streams, since D435 doesn't have IMU onboard.)
    Frames are retrieved in a dedicated thread (via wait_for_frames())
    and published on separate topics:
      - /rgb_frame   (sensor_msgs/Image)
      - /depth_frame (sensor_msgs/Image)
    """

    def __init__(self):
        super().__init__('live_capture_node')
        self.get_logger().info("Initializing LiveCaptureNode...")

        # Publishers
        self.color_publisher = self.create_publisher(Image, 'rgb_frame', 10)
        self.depth_publisher = self.create_publisher(Image, 'depth_frame', 10)
        # If you had a D435i, you would also publish IMU:
        # self.imu_publisher = self.create_publisher(Imu, 'camera/imu', 10)

        self.bridge = CvBridge()

        # Prepare RealSense pipeline
        self.stop_event = threading.Event()
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable color + depth
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # D435i example (comment out for D435):
        # self.config.enable_stream(rs.stream.gyro)
        # self.config.enable_stream(rs.stream.accel)

        try:
            _profile = self.pipeline.start(self.config)
            self.get_logger().info("RealSense pipeline started successfully (D435).")
        except Exception as e:
            self.get_logger().error(f"Failed to start RealSense pipeline: {e}")
            self.pipeline = None
            return

        # Start thread to publish frames
        self.publish_thread = threading.Thread(target=self.frame_publisher_loop, daemon=True)
        self.publish_thread.start()

    def frame_publisher_loop(self):
        self.get_logger().info("Starting frame publisher loop...")
        while not self.stop_event.is_set():
            try:
                frameset = self.pipeline.wait_for_frames(timeout_ms=500)
            except RuntimeError as e:
                self.get_logger().warn(f"No frames received in 500 ms: {e}")
                continue

            if not frameset:
                continue

            for frame in frameset:
                stream_type = frame.profile.stream_type()

                # Color frame
                if stream_type == rs.stream.color:
                    color_image = np.asanyarray(frame.get_data())
                    color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
                    color_msg.header.stamp = self.get_clock().now().to_msg()
                    color_msg.header.frame_id = 'camera_color_frame'
                    self.color_publisher.publish(color_msg)
                    self.get_logger().debug("Published color frame.")

                # Depth frame
                elif stream_type == rs.stream.depth:
                    depth_image = np.asanyarray(frame.get_data())
                    depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding='mono16')
                    depth_msg.header.stamp = self.get_clock().now().to_msg()
                    depth_msg.header.frame_id = 'camera_depth_frame'
                    self.depth_publisher.publish(depth_msg)
                    self.get_logger().debug("Published depth frame.")

                # IMU data (uncomment if D435i):
                # elif frame.is_motion_frame():
                #     motion_frame = frame.as_motion_frame()
                #     motion_data = motion_frame.get_motion_data()
                #     imu_msg = Imu()
                #     imu_msg.header.stamp = self.get_clock().now().to_msg()
                #     imu_msg.header.frame_id = 'camera_imu_frame'
                #     if stream_type == rs.stream.gyro:
                #         imu_msg.angular_velocity.x = motion_data.x
                #         imu_msg.angular_velocity.y = motion_data.y
                #         imu_msg.angular_velocity.z = motion_data.z
                #     elif stream_type == rs.stream.accel:
                #         imu_msg.linear_acceleration.x = motion_data.x
                #         imu_msg.linear_acceleration.y = motion_data.y
                #         imu_msg.linear_acceleration.z = motion_data.z
                #     imu_msg.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                #     self.imu_publisher.publish(imu_msg)
                #     self.get_logger().debug(f"Published IMU data ({stream_type}).")

        self.get_logger().info("Exiting frame publisher loop...")

    def destroy_node(self):
        self.get_logger().info("Stopping RealSense pipeline...")
        self.stop_event.set()  # stop the thread
        if hasattr(self, 'publish_thread') and self.publish_thread.is_alive():
            self.publish_thread.join(timeout=2.0)
        if self.pipeline:
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
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()
