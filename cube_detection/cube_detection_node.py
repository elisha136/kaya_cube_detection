#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# ROS message types
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point

# YOLO and OpenCV
from ultralytics import YOLO
import cv2

# NumPy for math
import numpy as np

# CvBridge to convert ROS Image <-> OpenCV
from cv_bridge import CvBridge


class CubeDetectionNode(Node):
    """
    A ROS 2 node that:
      1) Subscribes to an aligned color image (rgb_frame) and a matching depth image (depth_frame).
      2) Uses YOLO for object detection in the color image.
      3) Converts the bounding box center pixel -> 3D coordinates (X, Y, Z) using the pinhole camera model.
      4) Passes (X, Y, Z) to an Extended Kalman Filter (constant velocity in 3D).
      5) Publishes the filtered 3D position to:
            /cube/position
            /motion_planning/cube_position

    Assumes only one cube is of interest (highest confidence detection).
    """
    def __init__(self):
        super().__init__('cube_detection_node')

        # ---------------------------------
        # 1. YOLO Setup
        # ---------------------------------
        # Load your custom YOLO model
        MODEL_PATH = "/home/manulab/projects/Cubeproject/Dataset/results/cube_detection_exp3/weights/best.pt"
        self.model = YOLO(MODEL_PATH)
        self.get_logger().info(f"Loaded YOLO model from: {MODEL_PATH}")

        # ---------------------------------
        # 2. Camera Intrinsic Parameters
        # ---------------------------------
        # Replace these with your camera's calibration data.
        # For example, a 640x480 stream might have something like:
        #   fx ~ 615.0, fy ~ 615.0, cx ~ 320.0, cy ~ 240.0
        # Values below are placeholders -- you *must* use real ones for good accuracy.
        self.fx = 615.0
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0

        # Depth image resolution
        # (We assume color and depth are both 640x480 and aligned)
        self.depth_width = 640
        self.depth_height = 480

        # ---------------------------------
        # 3. ROS Subscribers & Publishers
        # ---------------------------------
        self.bridge = CvBridge()

        # Subscribe to color
        self.subscription_rgb = self.create_subscription(
            Image, 'rgb_frame', self.rgb_frame_callback, 10
        )

        # Subscribe to depth
        self.subscription_depth = self.create_subscription(
            Image, 'depth_frame', self.depth_frame_callback, 10
        )

        # Publishers: 3D position of the detected cube
        self.cube_position_pub = self.create_publisher(Point, '/cube/position', 10)
        self.motion_planning_pub = self.create_publisher(Point, '/motion_planning/cube_position', 10)

        # Storage for latest depth image
        self.latest_depth_image = None

        # ---------------------------------
        # 4. EKF State Initialization
        # ---------------------------------
        # State: [x, y, z, vx, vy, vz] (in meters and m/s)
        self.state = np.zeros(6, dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 1.0  # initial covariance

        # Noise covariances (tune as needed)
        self.process_noise = np.eye(6, dtype=np.float32) * 0.2
        self.measurement_noise = np.eye(3, dtype=np.float32) * 0.3

        # EKF nominal time step (s). Could be improved by measuring actual dt from timestamps.
        self.dt = 0.1

        # For optional display
        self.show_detections = True

        self.get_logger().info("CubeDetectionNode initialized successfully.")

    # ---------------------------------------------------------------------
    # Depth Subscriber: store the latest depth frame
    # ---------------------------------------------------------------------
    def depth_frame_callback(self, msg):
        """
        Convert the depth image to a NumPy array. Typically 16-bit where
        each pixel is in millimeters.
        """
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')
            self.latest_depth_image = depth_image
        except Exception as e:
            self.get_logger().error(f"Error converting depth image: {e}")

    # ---------------------------------------------------------------------
    # Color Subscriber: YOLO detection + EKF
    # ---------------------------------------------------------------------
    def rgb_frame_callback(self, msg):
        """
        - Convert color image to OpenCV
        - Run YOLO detection
        - Retrieve depth from matching pixel
        - Convert (pixel_x, pixel_y, depth) -> 3D (X, Y, Z)
        - EKF update, then publish filtered position
        """
        try:
            color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting color image: {e}")
            return

        if self.latest_depth_image is None:
            # We haven't received a depth frame yet
            self.get_logger().info("No depth frame yet; skipping 3D projection.")
            self.ekf_predict()  # Optionally just do a predict step
            return

        # 1) YOLO detection (2D)
        detections_2d = self.detect_cubes_2d(color_image)
        if not detections_2d:
            # No detections -> EKF predict-only
            self.get_logger().info("No cube detected.")
            self.ekf_predict()
            return

        # We'll handle the highest-confidence detection (first in the sorted list)
        cx_px, cy_px, conf = detections_2d[0]

        # 2) Check bounds
        if not (0 <= cx_px < self.depth_width and 0 <= cy_px < self.depth_height):
            self.get_logger().info("Bounding box center out of depth image bounds.")
            self.ekf_predict()
            return

        # 3) Get raw depth in mm
        raw_depth = self.latest_depth_image[cy_px, cx_px]
        if raw_depth == 0:
            self.get_logger().info("Invalid depth (0) at detection center.")
            self.ekf_predict()
            return

        # 4) Convert to meters
        depth_m = raw_depth / 1000.0

        # 5) Pinhole camera model -> from pixel to 3D
        #    X = (x_px - cx)*Z / fx
        #    Y = (y_px - cy)*Z / fy
        #    Z = depth_m
        X_m = (cx_px - self.cx) * depth_m / self.fx
        Y_m = (cy_px - self.cy) * depth_m / self.fy
        Z_m = depth_m

        # 6) EKF update with measured [X_m, Y_m, Z_m]
        z_measurement = np.array([X_m, Y_m, Z_m], dtype=np.float32)

        # Predict, then update
        self.ekf_predict()
        self.ekf_update(z_measurement)

        # 7) Publish filtered position
        filtered_x = self.state[0]
        filtered_y = self.state[1]
        filtered_z = self.state[2]

        point_msg = Point(x=float(filtered_x), y=float(filtered_y), z=float(filtered_z))
        self.cube_position_pub.publish(point_msg)
        self.motion_planning_pub.publish(point_msg)
        self.get_logger().info(
            f"Published filtered cube position: x={filtered_x:.2f}, y={filtered_y:.2f}, z={filtered_z:.2f} (conf={conf:.2f})"
        )

        # 8) Optional display
        if self.show_detections:
            # Draw circle at detection center
            cv2.circle(color_image, (cx_px, cy_px), 5, (0, 255, 0), -1)
            # Optionally label with confidence
            cv2.putText(color_image, f"{conf:.2f}", (cx_px+5, cy_px-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow("YOLO + Depth", color_image)
            cv2.waitKey(1)

    # ---------------------------------------------------------------------
    # YOLO Inference
    # ---------------------------------------------------------------------
    def detect_cubes_2d(self, color_image):
        """
        Runs YOLO on the color image.
        Returns a list of (center_x_pixel, center_y_pixel, confidence),
        sorted by descending confidence.
        """
        results = self.model(color_image)
        detections = []
        for result in results:
            for box in result.boxes:
                # box.xyxy = [x1, y1, x2, y2]
                coords = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, coords)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detections.append((cx, cy, conf))

        # Sort by confidence descending
        detections.sort(key=lambda x: x[2], reverse=True)

        return detections

    # ---------------------------------------------------------------------
    # EKF Predict
    # ---------------------------------------------------------------------
    def ekf_predict(self):
        """
        Constant velocity model in 3D:
          state = [x, y, z, vx, vy, vz]^T
        F = [[1, 0, 0, dt, 0,  0 ],
             [0, 1, 0, 0,  dt, 0 ],
             [0, 0, 1, 0,  0,  dt],
             [0, 0, 0, 1,  0,  0 ],
             [0, 0, 0, 0,  1,  0 ],
             [0, 0, 0, 0,  0,  1 ]]
        """
        dt = self.dt
        F = np.array([
            [1, 0, 0, dt, 0,  0 ],
            [0, 1, 0, 0,  dt, 0 ],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0 ],
            [0, 0, 0, 0,  1,  0 ],
            [0, 0, 0, 0,  0,  1 ]
        ], dtype=np.float32)

        # Predict step
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.process_noise

    # ---------------------------------------------------------------------
    # EKF Update
    # ---------------------------------------------------------------------
    def ekf_update(self, measurement):
        """
        measurement = [x, y, z] in meters
        H = [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0]]
        """
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)

        z = measurement.reshape((3, 1))
        x_pred = self.state.reshape((6, 1))

        # Innovation
        y_k = z - (H @ x_pred)

        # Innovation covariance
        S = H @ self.P @ H.T + self.measurement_noise

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        x_new = x_pred + K @ y_k
        self.state = x_new.flatten()

        # Update covariance
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ H) @ self.P

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
