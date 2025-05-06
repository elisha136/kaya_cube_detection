#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
import os
import cv2
import numpy as np
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose

from ultralytics import YOLO
import pyrealsense2 as rs

class CubeDetectionNode(Node):
    """
ROS 2 Cube Detection Node with YOLOv8 and Intel RealSense D435

This node:
 1. Subscribes to /rgb_frame and /depth_frame topics.
 2. Runs YOLOv8 inference on RGB frames to detect cubes.
 3. Estimates 3D positions of cubes using aligned depth values from the RealSense camera.
 4. Applies an Extended Kalman Filter (EKF) for 3D tracking of detected cubes.
 5. Publishes tracked positions as a geometry_msgs/PoseArray to /cube/positions.

Author: Adimalara
"""   
    def __init__(self):
        super().__init__('cube_detection_node')

        # -----------------------------
        # Declare Parameters
        # -----------------------------
        self.declare_parameter('model_path', '')
        self.declare_parameter('show_detections', True)

        # -----------------------------
        # Resolve model path
        # -----------------------------
        user_model_path = self.get_parameter('model_path') \
                              .get_parameter_value().string_value
        if user_model_path:
            model_path = user_model_path
        else:
            # Now correctly use ROS-2’s share directory lookup
            pkg_share = get_package_share_directory('cube_detection')
            model_path = os.path.join(pkg_share, 'models', 'best.pt')

        self.show_detections = self.get_parameter('show_detections') \
                                   .get_parameter_value().bool_value

        # -----------------------------
        # Load YOLO Model
        # -----------------------------
        self.get_logger().info(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)



        # -----------------------------
        # Camera Intrinsics (calibrated for 640x480)
        # -----------------------------
        self.fx = 458.314
        self.fy = 609.756
        self.cx = 324.789
        self.cy = 252.507

        self.depth_width = 640
        self.depth_height = 480

        # -----------------------------
        #  Subscribers & Publishers
        # -----------------------------
        self.bridge = CvBridge()

        self.subscription_rgb = self.create_subscription(Image, 'rgb_frame', self.rgb_frame_callback, 10)
        self.subscription_depth = self.create_subscription(Image, 'depth_frame', self.depth_frame_callback, 10)

        self.cube_positions_pub = self.create_publisher(PoseArray, '/cube/positions', 10)

        # -----------------------------
        #  State and EKF Setup
        # -------------------------------
        self.latest_depth_image = None
        self.last_timestamp = None

        self.max_cubes = 5
        self.states = [np.zeros(6, dtype=np.float32) for _ in range(self.max_cubes)]
        self.Ps = [np.eye(6, dtype=np.float32) for _ in range(self.max_cubes)]

        self.process_noise = np.eye(6, dtype=np.float32) * 0.2
        self.measurement_noise = np.eye(3, dtype=np.float32) * 0.3

        self.get_logger().info("CubeDetectionNode initialized successfully with RealSense distance and EKF tracking.")

    def depth_frame_callback(self, msg: Image):
        """Callback to store the latest depth frame."""
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')
        except Exception as e:
            self.get_logger().error(f"Error converting depth image: {e}")

    def rgb_frame_callback(self, msg: Image):
        """Main callback: runs YOLO, calculates 3D positions, applies EKF, publishes PoseArray."""
        try:
            color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting color image: {e}")
            return

        if self.latest_depth_image is None:
            self.get_logger().info("No depth frame yet; skipping detection.")
            return

        current_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_timestamp is None:
            dt = 0.1
        else:
            dt = current_timestamp - self.last_timestamp
            dt = max(0.01, min(dt, 0.5))
        self.last_timestamp = current_timestamp

        detections_2d = self.detect_cubes_2d(color_image)
        if not detections_2d:
            self.get_logger().info("No cube detected.")
            return

        poses = PoseArray()
        poses.header.stamp = msg.header.stamp
        poses.header.frame_id = 'camera_link'

        for i, (cx_px, cy_px, conf) in enumerate(detections_2d[:self.max_cubes]):
            if not (0 <= cx_px < self.depth_width and 0 <= cy_px < self.depth_height):
                continue

            # Retrieve raw depth and convert to meters
            depth_m = self.latest_depth_image[cy_px, cx_px] * 0.001
            if depth_m <= 0:
                continue

            # Project 2D center to 3D coordinate using pinhole model
            X_m = (cx_px - self.cx) * depth_m / self.fx
            Y_m = (cy_px - self.cy) * depth_m / self.fy
            Z_m = depth_m

            measurement = np.array([X_m, Y_m, Z_m], dtype=np.float32)

            self.ekf_predict(i, dt)
            self.ekf_update(i, measurement)

            pose = Pose()
            pose.position.x = float(self.states[i][0])
            pose.position.y = float(self.states[i][1])
            pose.position.z = float(self.states[i][2])
            poses.poses.append(pose)

            self.get_logger().info(f"Cube {i}: x={pose.position.x:.2f}, y={pose.position.y:.2f}, z={pose.position.z:.2f}, conf={conf:.2f}")

            # Visualization
            if self.show_detections:
                cv2.circle(color_image, (cx_px, cy_px), 5, (0,255,0), -1)
                cv2.putText(color_image, f"{conf:.2f} {Z_m:.2f}m", (cx_px+5, cy_px-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        if poses.poses:
            self.cube_positions_pub.publish(poses)

        if self.show_detections:
            cv2.imshow("YOLO + Depth", color_image)
            cv2.waitKey(1)

    def detect_cubes_2d(self, color_image: np.ndarray):
        """YOLO inference. Returns list of (cx, cy, confidence) sorted by confidence."""
        results = self.model(color_image)
        detections = []
        for result in results:
            for box in result.boxes:
                coords = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, coords)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detections.append((cx, cy, conf))
        detections.sort(key=lambda x: x[2], reverse=True)
        return detections

    def ekf_predict(self, idx, dt):
        """Performs EKF prediction for a given cube index."""
        F = np.array([
            [1, 0, 0, dt, 0,  0 ],
            [0, 1, 0, 0,  dt, 0 ],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0 ],
            [0, 0, 0, 0,  1,  0 ],
            [0, 0, 0, 0,  0,  1 ]
        ], dtype=np.float32)

        self.states[idx] = F @ self.states[idx]
        self.Ps[idx] = F @ self.Ps[idx] @ F.T + self.process_noise

    def ekf_update(self, idx, measurement: np.ndarray):
        """Performs EKF update using a new 3D position measurement."""
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)

        z = measurement.reshape((3, 1))
        x_pred = self.states[idx].reshape((6, 1))

        y_k = z - (H @ x_pred)
        S = H @ self.Ps[idx] @ H.T + self.measurement_noise
        K = self.Ps[idx] @ H.T @ np.linalg.inv(S)

        x_new = x_pred + K @ y_k
        self.states[idx] = x_new.flatten()

        I = np.eye(6, dtype=np.float32)
        self.Ps[idx] = (I - K @ H) @ self.Ps[idx]

def main(args=None):
    """Entry point for ROS 2 node."""
    rclpy.init(args=args)
    node = CubeDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
