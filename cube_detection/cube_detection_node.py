#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class CubeDetectionNode(Node):
    """
    A ROS 2 node that:
     1) Subscribes to /rgb_frame (color) and /depth_frame (depth) -- aligned and 640x480 resolution.
     2) Runs TensorRT YOLO model to detect cubes in the color image.
     3) Projects bounding box centers into 3D using median depth filtering.
     4) Tracks multiple cubes using separate Extended Kalman Filters.
     5) Publishes all filtered cube positions as a PoseArray.
    """

    def __init__(self):
        super().__init__('cube_detection_node_trt')

        # -----------------------------
        # 1. ROS2 Parameters
        # -----------------------------
        self.declare_parameter('trt_model_path', '/home/manulab/projects/Cubeproject/model.trt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('max_cubes', 5)
        self.declare_parameter('debug_view', True)

        model_path = self.get_parameter('trt_model_path').get_parameter_value().string_value
        self.conf_thres = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.max_cubes = self.get_parameter('max_cubes').get_parameter_value().integer_value
        self.debug_view = self.get_parameter('debug_view').get_parameter_value().bool_value

        # -----------------------------
        # 2. TensorRT Engine Setup
        # -----------------------------
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        with open(model_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        self.in_w, self.in_h = self.engine.get_binding_shape(0)[-1], self.engine.get_binding_shape(0)[-2]
        self.output_shape = self.engine.get_binding_shape(1)

        self.d_input = cuda.mem_alloc(trt.volume(self.engine.get_binding_shape(0)) * np.float32().itemsize)
        self.d_output = cuda.mem_alloc(trt.volume(self.output_shape) * np.float32().itemsize)

        self.bindings = [int(self.d_input), int(self.d_output)]

        # -----------------------------
        # 3. Camera Intrinsic Params (color camera, scaled 640x480)
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

        self.latest_depth_image = None
        self.last_timestamp = None

        # EKF State: Track multiple cubes
        self.states = [np.zeros(6, dtype=np.float32) for _ in range(self.max_cubes)]
        self.Ps = [np.eye(6, dtype=np.float32) for _ in range(self.max_cubes)]

        self.process_noise = np.eye(6, dtype=np.float32) * 0.2
        self.measurement_noise = np.eye(3, dtype=np.float32) * 0.3

        self.get_logger().info("CubeDetectionNodeTRT initialized successfully.")

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

        current_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_timestamp is None:
            dt = 0.1
        else:
            dt = current_timestamp - self.last_timestamp
            dt = max(0.01, min(dt, 0.5))
        self.last_timestamp = current_timestamp

        detections_2d = self.run_inference(color_image)

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

            self.ekf_predict(i, dt)
            self.ekf_update(i, measurement)

            pose = Pose()
            pose.position.x = float(self.states[i][0])
            pose.position.y = float(self.states[i][1])
            pose.position.z = float(self.states[i][2])
            poses.poses.append(pose)

            self.get_logger().info(f"Cube {i}: x={pose.position.x:.2f}, y={pose.position.y:.2f}, z={pose.position.z:.2f}, conf={conf:.2f}")

            if self.debug_view:
                cv2.circle(color_image, (cx_px, cy_px), 5, (0,255,0), -1)
                cv2.putText(color_image, f"{conf:.2f}", (cx_px+5, cy_px-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        if poses.poses:
            self.cube_positions_pub.publish(poses)

        if self.debug_view:
            cv2.imshow("YOLO TRT + Depth", color_image)
            cv2.waitKey(1)

    def run_inference(self, color_image: np.ndarray):
        resized = cv2.resize(color_image, (self.in_w, self.in_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)

        cuda.memcpy_htod(self.d_input, img)
        self.context.execute_v2(self.bindings)

        output = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output, self.d_output)

        output = output.reshape(-1, 6)

        detections = []
        for det in output:
            x1, y1, x2, y2, conf, cls_id = det
            if conf < self.conf_thres or int(cls_id) != 0:
                continue
            sx, sy = 640 / self.in_w, 480 / self.in_h
            cx = int((x1 + x2) / 2 * sx)
            cy = int((y1 + y2) / 2 * sy)
            detections.append((cx, cy, conf))

        detections.sort(key=lambda x: x[2], reverse=True)
        return detections

    def get_median_depth(self, cx_px, cy_px, window_size=5):
        half_w = window_size // 2
        x1 = max(cx_px - half_w, 0)
        y1 = max(cy_px - half_w, 0)
        x2 = min(cx_px + half_w, self.depth_width-1)
        y2 = min(cy_px + half_w, self.depth_height-1)

        depth_window = self.latest_depth_image[y1:y2+1, x1:x2+1]
        valid_depths = depth_window[depth_window > 0]

        if valid_depths.size == 0:
            return None

        return np.median(valid_depths) / 1000.0

    def ekf_predict(self, idx, dt):
        F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        self.states[idx] = F @ self.states[idx]
        self.Ps[idx] = F @ self.Ps[idx] @ F.T + self.process_noise

    def ekf_update(self, idx, measurement: np.ndarray):
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
