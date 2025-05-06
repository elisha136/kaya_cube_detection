#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import numpy as np
import cv2
import threading

from sensor_msgs.msg import Image
from geometry_msgs.msg import Quaternion
from cv_bridge import CvBridge

class LiveCaptureNode(Node):
    """
    A ROS2 node that uses one RealSense pipeline for color and depth.
    Frames are aligned so depth matches color resolution (640x480).
    Aligned frames are published on:
      - /rgb_frame   (sensor_msgs/Image)
      - /depth_frame (sensor_msgs/Image)
      Author: Adimalara
    """

    def __init__(self):
        super().__init__('live_capture_node')
        self.get_logger().info("Initializing LiveCaptureNode...")

        # Publishers
        self.color_publisher = self.create_publisher(Image, 'rgb_frame', 10)
        self.depth_publisher = self.create_publisher(Image, 'depth_frame', 10)

        self.bridge = CvBridge()

        # Prepare RealSense pipeline
        self.stop_event = threading.Event()
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable color + depth at 640x480 resolution
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        try:
            _profile = self.pipeline.start(self.config)
            self.get_logger().info("RealSense pipeline started successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to start RealSense pipeline: {e}")
            self.pipeline = None
            return

        # Align depth → color so every (u,v) pixel lines up
        self.align = rs.align(rs.stream.color)

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

            # Project depth onto the 640×480 color plane
            aligned_frames = self.align.process(frameset)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # Confirm both have the same resolution
            if color_frame.get_width() != depth_frame.get_width() or color_frame.get_height() != depth_frame.get_height():
                self.get_logger().warn("Color and depth frame size mismatch; skipping frame.")
                continue

            # Convert color frame to ROS Image
            color_image = np.asanyarray(color_frame.get_data())
            color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
            color_msg.header.stamp = self.get_clock().now().to_msg()
            color_msg.header.frame_id = 'camera_link'  # unified frame_id
            self.color_publisher.publish(color_msg)

            # Convert aligned depth frame to ROS Image
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding='mono16')
            depth_msg.header.stamp = self.get_clock().now().to_msg()
            depth_msg.header.frame_id = 'camera_link'  # unified frame_id
            self.depth_publisher.publish(depth_msg)

            self.get_logger().debug("Published aligned color+depth frames.")

        self.get_logger().info("Exiting frame publisher loop...")

    def destroy_node(self):
        self.get_logger().info("Stopping RealSense pipeline...")
        self.stop_event.set()
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
