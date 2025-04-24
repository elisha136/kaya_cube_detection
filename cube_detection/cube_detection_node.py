#!/usr/bin/env python3
#
# cube_detection_node_trt.py
#
# 1. subscribes to  /rgb_frame  &  /depth_frame
# 2. runs TensorRT cube detector on RGB
# 3. converts the most confident bbox centre to 3-D using aligned depth
# 4. constant-velocity EKF filtering
# 5. publishes:
#       /cube/position              (geometry_msgs/Point, camera frame)
#       /motion_planning/cube_position  (duplicate for next node)

import os
import threading
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import Image

# ── TensorRT / PyCUDA
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_engine(engine_path: Path) -> trt.ICudaEngine:
    if not engine_path.exists():
        raise FileNotFoundError(engine_path)
    with engine_path.open("rb") as f, trt.Runtime(TRT_LOGGER) as rt:
        return rt.deserialize_cuda_engine(f.read())


def alloc_trt_buffers(engine) -> Tuple[list, list, list, cuda.Stream]:
    """
    Returns  (inputs, outputs, bindings, stream)
      • inputs/outputs: list[dict{"host":…, "dev":…}]
      • bindings: List[int]  (device ptrs)
    """
    stream = cuda.Stream()
    inputs, outputs, bindings = [], [], []
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host = cuda.pagelocked_empty(size, dtype)
        dev = cuda.mem_alloc(host.nbytes)
        bindings.append(int(dev))
        (inputs if engine.binding_is_input(binding) else outputs).append(
            {"host": host, "dev": dev}
        )
    return inputs, outputs, bindings, stream


class CubeDetectionTRTNode(Node):
    # ──────────────────────────────────────────────────────────────
    def _init_(self):
        super()._init_("cube_detection_trt_node")

        # ---------- TensorRT -------------------------------------------------
        trt_path = Path.home() / "models" / "best.trt"   # adjust if necessary
        self.engine = load_engine(trt_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.cuda_stream = alloc_trt_buffers(
            self.engine
        )
        self.batch, self.in_c, self.in_h, self.in_w = (1, *self.engine.get_binding_shape(0))
        self.get_logger().info(f"TensorRT engine loaded: {trt_path.name} "
                               f"(expects {self.in_w}×{self.in_h})")

        # ---------- camera intrinsics for 640×480 (scaled) -------------------
        self.fx = 458.314
        self.fy = 609.756
        self.cx = 324.789
        self.cy = 252.507

        # ---------- EKF ------------------------------------------------------
        self.state = np.zeros(6, np.float32)   # [x y z vx vy vz]
        self.P = np.eye(6, dtype=np.float32)
        self.Q = np.eye(6, dtype=np.float32) * 0.2     # process noise
        self.R = np.eye(3, dtype=np.float32) * 0.3     # meas. noise
        self.dt = 0.1

        # ---------- ROS I/O ---------------------------------------------------
        self.bridge = CvBridge()
        self.depth_img = None

        self.sub_rgb = self.create_subscription(
            Image, "rgb_frame", self.cb_rgb, 10
        )
        self.sub_depth = self.create_subscription(
            Image, "depth_frame", self.cb_depth, 10
        )

        self.pub_pos = self.create_publisher(Point, "/cube/position", 10)
        # duplicate topic for next stage
        self.pub_pos_mp = self.create_publisher(
            Point, "/motion_planning/cube_position", 10
        )

        # debug window
        self.declare_parameter("debug_view", True)
        self.debug_view = bool(self.get_parameter("debug_view").get_parameter_value().bool_value)

    # ───────────────── depth callback ─────────────────
    def cb_depth(self, msg: Image):
        try:
            self.depth_img = self.bridge.imgmsg_to_cv2(msg, "mono16")
        except Exception as exc:
            self.get_logger().error(f"depth convert error: {exc}")

    # ───────────────── RGB callback ───────────────────
    def cb_rgb(self, msg: Image):
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            self.get_logger().error(f"rgb convert error: {exc}")
            return

        # no depth yet → only predict
        if self.depth_img is None:
            self._ekf_predict()
            return

        # run TensorRT inference  -------------------------------------------
        detections = self._run_inference(rgb)
        if not detections:
            self._ekf_predict()
            return

        # best detection (highest conf)
        x1, y1, x2, y2, conf, cls_id = detections[0]
        cx_px, cy_px = int((x1 + x2) / 2), int((y1 + y2) / 2)

        if not (0 <= cx_px < 640 and 0 <= cy_px < 480):
            self._ekf_predict()
            return

        depth_mm = int(self.depth_img[cy_px, cx_px])
        if depth_mm == 0:
            self._ekf_predict()
            return

        z_m = depth_mm / 1000.0
        x_m = (cx_px - self.cx) * z_m / self.fx
        y_m = (cy_px - self.cy) * z_m / self.fy

        meas = np.array([x_m, y_m, z_m], np.float32)
        self._ekf_predict()
        self._ekf_update(meas)

        pt = Point(x=float(self.state[0]), y=float(self.state[1]), z=float(self.state[2]))
        self.pub_pos.publish(pt)
        self.pub_pos_mp.publish(pt)

        self.get_logger().info(f"Cube @ ({pt.x:.2f}, {pt.y:.2f}, {pt.z:.2f}) m (conf={conf:.2f})")

        if self.debug_view:
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(rgb, (cx_px, cy_px), 4, (0, 0, 255), -1)
            cv2.putText(rgb, f"{conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
            cv2.imshow("Detections", rgb)
            cv2.waitKey(1)

    # ───────────────── TensorRT helpers ─────────────────
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """resize + BGR→RGB + NCHW + float32[0,1]"""
        resized = cv2.resize(img, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        return np.ascontiguousarray(tensor)

    def _run_inference(self, img: np.ndarray) -> List[Tuple[int, int, int, int, float, int]]:
        host_in = self.inputs[0]["host"]
        np.copyto(host_in, self._preprocess(img).ravel())
        cuda.memcpy_htod_async(self.inputs[0]["dev"], host_in, self.cuda_stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.cuda_stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["dev"], self.cuda_stream)
        self.cuda_stream.synchronize()

        out = self.outputs[0]["host"]

        # YOLOv8 TRT export → (num,6): x1,y1,x2,y2,conf,cls
        detections = out.reshape(-1, 6)
        detections = detections[detections[:, 4] > 0.3]      # conf threshold
        detections = detections[detections[:, 5] == 0]       # cube class id = 0
        detections = detections[np.argsort(-detections[:, 4])]   # sort high→low

        res: List[Tuple[int, int, int, int, float, int]] = []
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            # scale to 640×480
            sx, sy = 640 / self.in_w, 480 / self.in_h
            x1, y1, x2, y2 = [int(v * s) for v, s in zip((x1, y1, x2, y2), (sx, sy, sx, sy))]
            res.append((x1, y1, x2, y2, float(conf), int(cls_id)))
        return res

    # ───────────────── EKF ---------------------------------------------------
    def _ekf_predict(self):
        dt = self.dt
        F = np.array(
            [
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            np.float32,
        )
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q

    def _ekf_update(self, z: np.ndarray):
        H = np.hstack((np.eye(3), np.zeros((3, 3), np.float32)))
        y = z - H @ self.state
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(6, np.float32) - K @ H) @ self.P


def main(args=None):
    rclpy.init(args=args)
    node = CubeDetectionTRTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if _name_ == "_main_":
    main()