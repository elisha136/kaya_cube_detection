#!/usr/bin/env python3
#
# live_capture_node.py  – RealSense RGB + depth publisher (640×480 @ 30 Hz)

import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import numpy as np
import cv2
import threading

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class LiveCaptureNode(Node):
    """
    Publishes aligned RGB and depth frames:
       /rgb_frame   (sensor_msgs/Image, bgr8)
       /depth_frame (sensor_msgs/Image, mono16)
    """

    def _init_(self):
        super()._init_("live_capture_node")

        # ── publishers
        self.color_pub = self.create_publisher(Image, "rgb_frame", 10)
        self.depth_pub = self.create_publisher(Image, "depth_frame", 10)
        self.bridge = CvBridge()

        # ── RealSense pipeline
        self.stop_evt = threading.Event()
        pipe_cfg = rs.config()
        pipe_cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipe_cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.pipeline = rs.pipeline()
        try:
            self.pipeline.start(pipe_cfg)
            self.get_logger().info("RealSense pipeline started.")
        except Exception as exc:
            self.get_logger().error(f"Cannot start RealSense pipeline: {exc}")
            raise SystemExit(1)

        self.align = rs.align(rs.stream.color)
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    # ──────────────────────────────────────────────────────────────
    def _loop(self):
        while not self.stop_evt.is_set():
            try:
                frames = self.pipeline.wait_for_frames(500)
            except RuntimeError:
                continue
            frames = self.align.process(frames)
            c, d = frames.get_color_frame(), frames.get_depth_frame()
            if not c or not d:
                continue

            stamp = self.get_clock().now().to_msg()
            color_img = np.asanyarray(c.get_data())
            depth_img = np.asanyarray(d.get_data())

            self._pub_img(color_img, self.color_pub, "bgr8", stamp, "camera_color_frame")
            self._pub_img(depth_img, self.depth_pub, "mono16", stamp, "camera_depth_frame")

    def _pub_img(self, arr, pub, enc, stamp, frame_id):
        msg = self.bridge.cv2_to_imgmsg(arr, enc)
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        pub.publish(msg)

    # ──────────────────────────────────────────────────────────────
    def destroy_node(self):
        self.stop_evt.set()
        if self.worker.is_alive():
            self.worker.join()
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


if _name_ == "_main_":
    main()