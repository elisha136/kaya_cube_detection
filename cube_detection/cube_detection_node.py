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
     1) Subscribes to /rgb_frame (color) and /depth_frame (depth) -- both 640x480 after alignment
     2) Runs YOLO on the color image to detect cubes
     3) Uses the color camera's pinhole model + bounding box center pixel to get 3D position
     4) Feeds the 3D position into an Extended Kalman Filter (constant velocity in 3D)
     5) Publishes the filtered 3D position
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
        # 2. Camera Intrinsic Params (COLOR - 640x480)
        # Scaled from calibrated 1920x1080 values
        # -----------------------------
        self.fx = 458.314    # = 1374.942017 * 640 / 1920
        self.fy = 609.756    # = 1371.950195 * 480 / 1080
        self.cx = 324.789    # = 974.367798  * 640 / 1920
        self.cy = 252.507    # = 568.141541  * 480 / 1080

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

        self.cube_position_pub = self.create_publisher(Point, '/cube/position', 10)
        self.motion_planning_pub = self.create_publisher(Point, '/motion_planning/cube_position', 10)

        # Latest depth image in memory
        self.latest_depth_image = None

        # -----------------------------
        # 4. EKF State Initialization
        # -----------------------------
        self.state = np.zeros(6, dtype=np.float32)  # [x, y, z, vx, vy, vz]
        self.P = np.eye(6, dtype=np.float32) * 1.0
        self.process_noise = np.eye(6, dtype=np.float32) * 0.2
        self.measurement_noise = np.eye(3, dtype=np.float32) * 0.3
        self.dt = 0.1

        # Debug: show YOLO detections
        self.show_detections = True

        self.get_logger().info("CubeDetectionNode initialized successfully with 640x480 intrinsics.")

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
            self.ekf_predict()
            return

        detections_2d = self.detect_cubes_2d(color_image)
        if not detections_2d:
            self.get_logger().info("No cube detected.")
            self.ekf_predict()
            return

        cx_px, cy_px, conf = detections_2d[0]
        if not (0 <= cx_px < self.depth_width and 0 <= cy_px < self.depth_height):
            self.get_logger().info("BBox center out of depth bounds.")
            self.ekf_predict()
            return

        raw_depth = self.latest_depth_image[cy_px, cx_px]
        if raw_depth == 0:
            self.get_logger().info("Invalid depth (0).")
            self.ekf_predict()
            return
        depth_m = raw_depth / 1000.0

        X_m = (cx_px - self.cx) * depth_m / self.fx
        Y_m = (cy_px - self.cy) * depth_m / self.fy
        Z_m = depth_m

        measurement = np.array([X_m, Y_m, Z_m], dtype=np.float32)

        self.ekf_predict()
        self.ekf_update(measurement)

        point_msg = Point(
            x=float(self.state[0]),
            y=float(self.state[1]),
            z=float(self.state[2])
        )
        self.cube_position_pub.publish(point_msg)
        self.motion_planning_pub.publish(point_msg)

        self.get_logger().info(
            f"Filtered cube pos: x={point_msg.x:.2f}, y={point_msg.y:.2f}, z={point_msg.z:.2f}, conf={conf:.2f}"
        )

        if self.show_detections:
            cv2.circle(color_image, (cx_px, cy_px), 5, (0,255,0), -1)
            cv2.putText(color_image, f"{conf:.2f}", (cx_px+5, cy_px-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow("YOLO + Depth", color_image)
            cv2.waitKey(1)

    def detect_cubes_2d(self, color_image: np.ndarray):
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

    def ekf_predict(self):
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
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)

        z = measurement.reshape((3, 1))
        x_pred = self.state.reshape((6, 1))
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
