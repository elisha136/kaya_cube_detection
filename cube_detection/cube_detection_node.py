#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

class CubeDetectionNode(Node):
    def __init__(self):
        super().__init__('cube_detection_node')

        # Load the custom-trained YOLO model
        MODEL_PATH = "/home/manulab/projects/Cubeproject/Dataset/results/cube_detection_exp3/weights/best.pt"
        self.model = YOLO(MODEL_PATH)

        # ROS2 subscription and publishers
        self.subscription_rgb = self.create_subscription(
            Image,
            'rgb_frame',  # Subscribing to the RGB frames
            self.rgb_frame_callback,
            10
        )
        self.cube_position_publisher = self.create_publisher(Point, '/cube/position', 10)
        self.motion_planning_publisher = self.create_publisher(Point, '/motion_planning/cube_position', 10)
        self.bridge = CvBridge()

    def rgb_frame_callback(self, data):
        try:
            self.get_logger().info("Receiving RGB frame")
            # Convert ROS2 Image message to an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            # Perform cube detection using YOLO
            detections = self.detect_cubes(cv_image)

            if detections:
                for (cx, cy, cz) in detections:
                    position = Point(x=float(cx), y=float(cy), z=float(cz))
                    self.cube_position_publisher.publish(position)
                    self.motion_planning_publisher.publish(position)
            else:
                self.get_logger().info("No cube detected.")

            # Optionally display the frame with bounding boxes
            cv2.imshow("RGB Frame with Detection", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error processing RGB frame: {e}")

    def detect_cubes(self, image):
        try:
            results = self.model(image)
            detections = []
            # Iterate over all results (each result may have multiple detections)
            for result in results:
                for box in result.boxes:
                    # Extract bounding box coordinates and confidence
                    coords = box.xyxy[0].cpu().numpy()  # Expected format: [x1, y1, x2, y2]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, coords)
                    
                    # Draw the bounding box on the image for debugging
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Calculate the center of the bounding box
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    cz = 0  # For now, depth is assumed constant
                    
                    detections.append((cx, cy, cz, conf))
            
            # Sort detections by confidence (highest first)
            detections.sort(key=lambda x: x[3], reverse=True)
            # Limit the number of detections to 2 if there are more
            if len(detections) > 2:
                detections = detections[:2]
            
            # Remove the confidence value from the returned list
            return [(cx, cy, cz) for cx, cy, cz, conf in detections]
        except Exception as e:
            self.get_logger().error(f"Error during YOLO inference: {e}")
            return []

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
