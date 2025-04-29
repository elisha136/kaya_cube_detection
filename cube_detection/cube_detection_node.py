#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseArray, Pose

from ultralytics import YOLO

class CubeDetectionNode(Node):
    """
    A ROS 2 node that:
     1) Subscribes to /rgb_frame (color) and /depth_frame (depth) -- both 640x480 after alignment
     2) Runs YOLO on the color image to detect cubes
     3) Uses the color camera's pinhole model + bounding box center pixel (median depth) to get 3D positions
     4) Tracks 3D positions over time using an Extended Kalman Filter (constant velocity in 3D)
     5) Publishes all filtered 3D positions as a PoseArray
    """

    def __init__(self):
        super().__init__('cube_detection_node')

        # -----------------------------
        # 1. Declare ROS Parameters
        # -----------------------------
        self.declare_parameter('model_path', '/home/manulab/projects/Cubeproject/Dataset/results/cube_detection_exp3/weights/best.pt')
        self.declare_parameter('show_detections', True)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.show_detections = self.get_parameter('show_detections').get_parameter_value().bool_value

        # -----------------------------
        # 2. YOLO Setup
        # -----------------------------
        self.model = YOLO(model_path)
        self.get_logger().info(f"Loaded YOLO model from: {model_path}")

        # -----------------------------
        # 3. Camera Intrinsic Params (calibrated/scaled for 640x480)
        # -----------------------------
        self.fx = 458.314
        self.fy = 609.756
        self.cx = 324.789
        self.cy = 252.507

        self.depth_width = 640
        self.depth_height = 480

        # -----------------------------
        # 4. ROS Subscribers and Publishers
        # -----------------------------
        self.bridge = CvBridge()

        self.subscription_rgb = self.create_subscription(Image, 'rgb_frame', self.rgb_frame_callback, 10)
        self.subscription_depth = self.create_subscription(Image, 'depth_frame', self.depth_frame_callback, 10)

        self.cube_positions_pub = self.create_publisher(PoseArray, '/cube/positions', 10)

        # Store latest depth image and last timestamp
        self.latest_depth_image = None
        self.last_timestamp = None

        # EKF filter state for each cube (for now: track only top N cubes)
        self.max_cubes = 5  # Max number of cubes to track
        self.states = [np.zeros(6, dtype=np.float32) for _ in range(self.max_cubes)]
        self.Ps = [np.eye(6, dtype=np.float32) for _ in range(self.max_cubes)]

        # Covariance parameters
        self.process_noise = np.eye(6, dtype=np.float32) * 0.2
        self.measurement_noise = np.eye(3, dtype=np.float32) * 0.3

        self.get_logger().info("CubeDetectionNode initialized successfully with dynamic dt, median depth, and multi-cube tracking.")

    def depth_frame_callback(self, msg: Image):
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')
        except Exception as e:
            self.get_logger().error(f"Error converting depth image: {e}")

    def rgb_frame_callback(self, msg: Image):
        try:
            color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting color image: {e}")
            return

        if self.latest_depth_image is None:
            self.get_logger().info("No depth frame yet; skipping detection.")
            return

        # Calculate dynamic dt
        current_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_timestamp is None:
            dt = 0.1
        else:
            dt = current_timestamp - self.last_timestamp
            dt = max(0.01, min(dt, 0.5))  # Clamp dt to reasonable range
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

            depth_m = self.get_median_depth(cx_px, cy_px)
            if depth_m is None:
                continue

            X_m = (cx_px - self.cx) * depth_m / self.fx
            Y_m = (cy_px - self.cy) * depth_m / self.fy
            Z_m = depth_m

            measurement = np.array([X_m, Y_m, Z_m], dtype=np.float32)

            # Predict and update EKF for this cube
            self.ekf_predict(i, dt)
            self.ekf_update(i, measurement)

            pose = Pose()
            pose.position.x = float(self.states[i][0])
            pose.position.y = float(self.states[i][1])
            pose.position.z = float(self.states[i][2])
            poses.poses.append(pose)

            self.get_logger().info(f"Cube {i}: x={pose.position.x:.2f}, y={pose.position.y:.2f}, z={pose.position.z:.2f}, conf={conf:.2f}")

            # Optional visualization
            if self.show_detections:
                cv2.circle(color_image, (cx_px, cy_px), 5, (0,255,0), -1)
                cv2.putText(color_image, f"{conf:.2f}", (cx_px+5, cy_px-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Publish all cube positions
        if poses.poses:
            self.cube_positions_pub.publish(poses)

        if self.show_detections:
            cv2.imshow("YOLO + Depth", color_image)
            cv2.waitKey(1)

    def detect_cubes_2d(self, color_image: np.ndarray):
        """
        YOLO inference. Returns list of (cx, cy, confidence), sorted descending by confidence.
        """
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

    def get_median_depth(self, cx_px, cy_px, window_size=5):
        """
        Returns the median depth around the (cx, cy) pixel.
        """
        half_w = window_size // 2
        x1 = max(cx_px - half_w, 0)
        y1 = max(cy_px - half_w, 0)
        x2 = min(cx_px + half_w, self.depth_width-1)
        y2 = min(cy_px + half_w, self.depth_height-1)

        depth_window = self.latest_depth_image[y1:y2+1, x1:x2+1]
        valid_depths = depth_window[depth_window > 0]

        if valid_depths.size == 0:
            return None

        return np.median(valid_depths) / 1000.0  # Convert mm to meters

    def ekf_predict(self, idx, dt):
        """
        Constant velocity predict step for cube idx.
        """
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
        """
        Measurement update step for cube idx.
        """
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
# This is a ROS 2 node for detecting cubes in a color image and estimating their 3D positions using depth data.
# It uses YOLO for object detection and an Extended Kalman Filter for tracking.
# The node subscribes to color and depth images, processes them, and publishes the detected cube positions.
# The code includes dynamic dt calculation, median depth extraction, and multi-cube tracking.
# The node is designed to be run in a ROS 2 environment and requires the ultralytics YOLO library and OpenCV.
# The node is initialized with camera intrinsic parameters and can visualize detections if configured.
# The EKF is used to predict and update the state of each detected cube, allowing for smooth tracking over time.
# The code is structured to handle multiple cubes, with a maximum limit set for tracking.
# The node is designed to be modular and can be extended for additional functionality as needed.
