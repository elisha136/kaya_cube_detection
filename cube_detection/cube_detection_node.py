#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np

class CubeDetectionNode(Node):
    def __init__(self):
        super().__init__('cube_detection_node')

        # Load the custom-trained YOLO model
        MODEL_PATH = "/home/elisha/Cubeproject/Dataset/results/cube_detection_exp3/weights/best.pt"
        self.model = YOLO(MODEL_PATH)

        # ROS2 subscription and publishers
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',  # Ensure this matches the live capture node's topic
            self.image_callback,
            10
        )
        self.cube_position_publisher = self.create_publisher(Point, '/cube/position', 10)
        self.motion_planning_publisher = self.create_publisher(Point, '/motion_planning/cube_position', 10)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        try:
            # Convert ROS2 Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Perform cube detection using YOLO
            x, y, z = self.detect_cube(cv_image)

            # Publish cube position to both topics
            position = Point(x=float(x), y=float(y), z=float(z))
            self.cube_position_publisher.publish(position)
            self.motion_planning_publisher.publish(position)
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def detect_cube(self, image):
        try:
            # Run inference using YOLO
            results = self.model(image)

            # Extract the first detected cube's bounding box (if any)
            for result in results:
                for box in result.boxes.xyxy:  # Bounding box coordinates
                    x1, y1, x2, y2 = map(int, box[:4])  # Ensure correct slicing
                    # Calculate the center of the bounding box
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    cz = 0  # Assuming z is constant or derived from depth
                    return cx, cy, cz

            # If no cube is detected, log and return default position
            self.get_logger().info("No cube detected.")
            return 0, 0, 0
        except Exception as e:
            self.get_logger().error(f"Error during YOLO inference: {e}")
            return 0, 0, 0

def main(args=None):
    rclpy.init(args=args)
    node = CubeDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()