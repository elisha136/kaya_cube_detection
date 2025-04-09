#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point

from ultralytics import YOLO

class CubeDetectionNode(Node):
    """
    A ROS 2 node that:
     1) Subscribes to /rgb_frame (color) and /depth_frame (depth)
     2) Runs YOLO on the color image to detect cubes
     3) Uses pinhole camera model + the bounding box center pixel to get 3D position
     4) Feeds that 3D position into an Extended Kalman Filter (constant velocity in 3D)
     5) Publishes the filtered 3D position to:
         /cube/position
         /motion_planning/cube_position
    """

    def __init__(self):
        super().__init__('cube_detection_node')

        # -----------------------------
        # 1. YOLO Setup
        # -----------------------------
        MODEL_PATH = "/home/manulab/projects/Cubeproject/Dataset/results/cube_detection_exp3/weights/best.pt"
        self.model = YOLO(MODEL_PATH)
        self.get_logger().info(f"Loaded YOLO model from: {MODEL_PATH}")

        # -----------------------------
        # 2. Camera Intrinsic Params
        # -----------------------------
        # From pyrealsense2 for 640x480:
        self.fx = 611.0853
        self.fy = 609.7556
        self.cx = 326.3857
        self.cy = 252.5073

        # Depth resolution must match color if they're aligned
        self.depth_width = 640
        self.depth_height = 480

        # -----------------------------
        # 3. ROS Subscribers/Pubs
        # -----------------------------
        self.bridge = CvBridge()

        self.subscription_rgb = self.create_subscription(
            Image, 'rgb_frame', self.rgb_frame_callback, 10
        )
        self.subscription_depth = self.create_subscription(
            Image, 'depth_frame', self.depth_frame_callback, 10
        )

        # Publish the 3D position of the detected cube
        self.cube_position_pub = self.create_publisher(Point, '/cube/position', 10)
        self.motion_planning_pub = self.create_publisher(Point, '/motion_planning/cube_position', 10)

        # Latest depth image in memory
        self.latest_depth_image = None

        # -----------------------------
        # 4. EKF State Initialization
        # -----------------------------
        # State: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6, dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 1.0  # Covariance

        # Covariances (tune as needed)
        self.process_noise = np.eye(6, dtype=np.float32) * 0.2
        self.measurement_noise = np.eye(3, dtype=np.float32) * 0.3

        # Nominal dt (seconds)
        self.dt = 0.1

        # Show bounding boxes with OpenCV
        self.show_detections = True

        self.get_logger().info("CubeDetectionNode initialized successfully.")

    def depth_frame_callback(self, msg: Image):
        """Receive depth frame as 16-bit mm."""
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')
            self.latest_depth_image = depth_image
        except Exception as e:
            self.get_logger().error(f"Error converting depth image: {e}")

    def rgb_frame_callback(self, msg: Image):
        """Perform YOLO detection, get depth, run EKF, publish 3D position."""
        try:
            color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting color image: {e}")
            return

        if self.latest_depth_image is None:
            self.get_logger().info("No depth frame yet; skipping detection.")
            self.ekf_predict()
            return

        # 1) YOLO detect
        detections_2d = self.detect_cubes_2d(color_image)
        if not detections_2d:
            self.get_logger().info("No cube detected.")
            self.ekf_predict()
            return

        # Highest confidence detection
        cx_px, cy_px, conf = detections_2d[0]

        # 2) Check bounds
        if not (0 <= cx_px < self.depth_width and 0 <= cy_px < self.depth_height):
            self.get_logger().info("BBox center out of depth bounds.")
            self.ekf_predict()
            return

        # 3) Get depth (mm) at that pixel
        raw_depth = self.latest_depth_image[cy_px, cx_px]
        if raw_depth == 0:
            self.get_logger().info("Invalid depth (0).")
            self.ekf_predict()
            return

        depth_m = raw_depth / 1000.0

        # 4) Pinhole camera model
        X_m = (cx_px - self.cx) * depth_m / self.fx
        Y_m = (cy_px - self.cy) * depth_m / self.fy
        Z_m = depth_m

        measurement = np.array([X_m, Y_m, Z_m], dtype=np.float32)

        # 5) EKF
        self.ekf_predict()
        self.ekf_update(measurement)

        filtered_x = self.state[0]
        filtered_y = self.state[1]
        filtered_z = self.state[2]

        # 6) Publish
        point_msg = Point(x=float(filtered_x), y=float(filtered_y), z=float(filtered_z))
        self.cube_position_pub.publish(point_msg)
        self.motion_planning_pub.publish(point_msg)

        self.get_logger().info(
            f"Filtered cube pos: x={filtered_x:.2f}, y={filtered_y:.2f}, z={filtered_z:.2f}, conf={conf:.2f}"
        )

        # 7) Optional display
        if self.show_detections:
            cv2.circle(color_image, (cx_px, cy_px), 5, (0,255,0), -1)
            cv2.putText(color_image, f"{conf:.2f}", 
                        (cx_px+5, cy_px-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)
            cv2.imshow("YOLO + Depth", color_image)
            cv2.waitKey(1)

    def detect_cubes_2d(self, color_image: np.ndarray):
        """
        YOLO inference. Returns list of (cx, cy, confidence).
        Sorted descending by confidence.
        """
        results = self.model(color_image)
        detections = []
        for result in results:
            for box in result.boxes:
                coords = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, coords)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detections.append((cx, cy, conf))

        detections.sort(key=lambda x: x[2], reverse=True)
        return detections

    def ekf_predict(self):
        """Constant velocity predict step."""
        dt = self.dt
        F = np.array([
            [1, 0, 0, dt, 0,  0 ],
            [0, 1, 0, 0,  dt, 0 ],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0 ],
            [0, 0, 0, 0,  1,  0 ],
            [0, 0, 0, 0,  0,  1 ]
        ], dtype=np.float32)

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.process_noise

    def ekf_update(self, measurement: np.ndarray):
        """Measurement = [x, y, z]."""
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)

        z = measurement.reshape((3,1))
        x_pred = self.state.reshape((6,1))

        # Innovation
        y_k = z - (H @ x_pred)
        S = H @ self.P @ H.T + self.measurement_noise

        K = self.P @ H.T @ np.linalg.inv(S)

        x_new = x_pred + K @ y_k
        self.state = x_new.flatten()

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
