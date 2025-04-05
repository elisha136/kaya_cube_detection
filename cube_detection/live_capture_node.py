#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np  # Import numpy

class LiveCaptureNode(Node):
    def __init__(self):
        super().__init__('live_capture_node')
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)  # Ensure topic matches cube_detection_node
        self.bridge = CvBridge()

        # Configure RealSense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Configure color stream
        self.pipeline.start(config)

        self.timer = self.create_timer(1/30, self.timer_callback)  # 30 Hz

    def timer_callback(self):
        try:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                return

            # Convert RealSense frame to numpy array
            frame = np.asanyarray(color_frame.get_data())

            # Convert OpenCV image to ROS2 Image message
            ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.publisher_.publish(ros_image)
        except Exception as e:
            self.get_logger().error(f"Error capturing/publishing image: {e}")

    def destroy_node(self):
        # Stop the RealSense pipeline
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
        node.destroy_node()  # Ensure pipeline is stopped
        rclpy.shutdown()

if __name__ == '__main__':
    main()