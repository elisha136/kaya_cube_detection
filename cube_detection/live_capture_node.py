import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import pyrealsense2 as rs
import numpy as np

class IntelPublisher(Node):
    def __init__(self):
        super().__init__("intel_publisher")
        self.intel_publisher_rgb = self.create_publisher(Image, "rgb_frame", 10)

        timer_period = 0.05  # 20 Hz
        self.br_rgb = CvBridge()

        try:
            # Initialize RealSense pipeline
            self.pipe = rs.pipeline()
            self.cfg = rs.config()
            self.cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipe.start(self.cfg)
            self.timer = self.create_timer(timer_period, self.timer_callback)
        except Exception as e:
            self.get_logger().error(f"Intel RealSense is not connected: {e}")

    def timer_callback(self):
        try:
            # Wait for a coherent pair of frames
            frames = self.pipe.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                self.get_logger().warn("No color frame received")
                return

            # Convert RealSense frame to numpy array
            color_image = np.asanyarray(color_frame.get_data())

            # Publish the frame as a ROS2 Image message
            self.intel_publisher_rgb.publish(self.br_rgb.cv2_to_imgmsg(color_image, encoding="bgr8"))
            self.get_logger().info("Publishing RGB frame")
        except Exception as e:
            self.get_logger().error(f"Error in timer_callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    intel_publisher = IntelPublisher()
    try:
        rclpy.spin(intel_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        intel_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
