#!/usr/bin/env python3
"""
rollout_dp3_spot.py
-------------------
Roll out a trained DP3 policy on Boston Dynamics Spot using dual-camera
point-cloud observations.

Observation format (matches training data exactly):
  agentview_point_cloud  : (8192, 6) XYZ + BGR (uint8 range 0-255)
  eye_in_hand_point_cloud: (8192, 6) XYZ + BGR (uint8 range 0-255)
  state                  : (20,) float32
      [ arm_q(7) | ee_pos(3) | ee_rot_flat(9) | gripper(1) ]

Action format (10-dim):
  [ Δx Δy Δz (3) | rot6d (6) | gripper_abs (1) ]

Usage:
python rollout_dp3_spot.py --ckpt ../data/screwdriver/epoch=0100-train_loss=0.001.ckpt \
    --use-spot-hand-camera \
    --control-hz 10 \
    --show

Safety: keep one hand near the E-stop at all times.
"""
from __future__ import annotations

import argparse
import os
import queue
import signal
import sys
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import dill
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# DP3 package path — works whether installed with pip or not
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # scripts/
_DP3_ROOT = os.environ.get(
    "DP3_ROOT",
    os.path.normpath(os.path.join(_THIS_DIR, "..", "3D-Diffusion-Policy"))
)
if os.path.isdir(_DP3_ROOT) and _DP3_ROOT not in sys.path:
    sys.path.insert(0, _DP3_ROOT)
    print(f"[DP3] Added to sys.path: {_DP3_ROOT}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_to_path(p: str) -> None:
    if p not in sys.path:
        sys.path.insert(0, p)


def _get_shape_meta(cfg) -> dict:
    if "policy" in cfg and "shape_meta" in cfg.policy:
        return OmegaConf.to_container(cfg.policy.shape_meta, resolve=True)
    if "task" in cfg and "shape_meta" in cfg.task:
        return OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
    raise KeyError("shape_meta not found in cfg.policy or cfg.task.")


def _get_obs_horizon(cfg) -> int:
    """Extract n_obs_steps from config."""
    for key in ("n_obs_steps", "obs_horizon"):
        if hasattr(cfg, key):
            return int(getattr(cfg, key))
        if "policy" in cfg and hasattr(cfg.policy, key):
            return int(getattr(cfg.policy, key))
    return 2  # sensible default


def _extract_obs_keys(shape_meta: dict) -> Tuple[List[str], List[str]]:
    """Return (pc_keys, lowdim_keys) from shape_meta."""
    obs_meta = shape_meta["obs"]
    pc_keys, lowdim_keys = [], []
    for key, attr in obs_meta.items():
        obs_type = attr.get("type", "low_dim")
        if obs_type == "point_cloud":
            pc_keys.append(key)
        else:
            lowdim_keys.append(key)
    return pc_keys, lowdim_keys


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(x, y, z, w) → (3, 3) rotation matrix."""
    x, y, z, w = q.astype(np.float64)
    tx, ty, tz = 2 * x, 2 * y, 2 * z
    R = np.array([
        [1 - ty*y - tz*z,  tx*y - tz*w,     tx*z + ty*w],
        [tx*y + tz*w,      1 - tx*x - tz*z, ty*z - tx*w],
        [tx*z - ty*w,      ty*z + tx*w,     1 - tx*x - ty*y],
    ], dtype=np.float32)
    return R

def project_rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """
    Projects a 6D rotation representation to the nearest valid 3x3 rotation matrix using Gram-Schmidt.
    rot6d: (6,) array
    Returns: (3, 3) rotation matrix
    """
    a1 = rot6d[:3]
    a2 = rot6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=-1)
    return R


# ---------------------------------------------------------------------------
# Point-cloud helpers (matching training exactly)
# ---------------------------------------------------------------------------

def voxel_grid_sampling(xyz: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Fast, deterministic voxel-grid downsampling.

    Parameters
    ----------
    xyz        : (N, 3) XYZ coordinates
    num_samples: target number of output points

    Returns
    -------
    selected_indices : (≤ num_samples,) int32 indices into xyz
    """
    N = len(xyz)
    if N <= num_samples:
        return np.arange(N, dtype=np.int32)

    pts_min = xyz.min(axis=0)
    pts_max = xyz.max(axis=0)
    bbox = pts_max - pts_min + 1e-6

    voxel_vol = bbox.prod() / num_samples
    vsize = float(np.cbrt(voxel_vol))
    if vsize < 1e-6:
        vsize = bbox.max() / float(np.cbrt(num_samples))

    vidx = np.floor((xyz - pts_min) / vsize).astype(np.int64)
    keys = vidx[:, 0] * 73_856_093 + vidx[:, 1] * 19_349_663 + vidx[:, 2] * 83_492_791

    _, first = np.unique(keys, return_index=True)
    sel = np.sort(first)

    if len(sel) >= num_samples:
        return sel[:num_samples].astype(np.int32)

    # pad with remaining points uniformly
    mask = np.ones(N, dtype=bool)
    mask[sel] = False
    rest = np.where(mask)[0]
    need = num_samples - len(sel)
    if len(rest) > 0:
        step = max(1, len(rest) // need)
        extra = rest[::step][:need]
        sel = np.concatenate([sel, extra])
    return sel[:num_samples].astype(np.int32)


def rgbd_to_pointcloud(
    color_bgr: np.ndarray,
    depth_u16: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float = 0.001,
    min_depth_m: float = 0.01,
    max_depth_m: float = float('inf'),
    num_points: int = 8192,
) -> np.ndarray:
    """
    Convert a BGR-D frame to an (N, 6) point cloud matching the training format.

    IMPORTANT: Colors are kept in BGR order to match the training data
    (CameraStreamer uses rs.format.bgr8, and convert_spot_hdf5_to_zarr.py
    stores these BGR values directly without conversion).

    Colors are kept in uint8 range [0, 255] (NOT divided by 255) to match the
    values stored in the zarr training dataset.

    Parameters
    ----------
    color_bgr  : (H, W, 3) BGR image (from CameraStreamer or Spot hand camera)
    depth_u16  : (H, W) uint16 depth, raw units → meters via depth_scale
    fx, fy     : focal lengths in pixels (276.0 to match training)
    cx, cy     : principal point in pixels
    depth_scale: multiply raw depth by this to get metres (default: 0.001)
    Returns
    -------
    (num_points, 6) float32 array: [X, Y, Z, B, G, R]
    """
    h, w = depth_u16.shape
    depth_m = depth_u16.astype(np.float32) * depth_scale

    us, vs = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))

    z = depth_m
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy

    pts_3d = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Keep BGR uint8 values (range 0-255, matching training)
    colors = color_bgr.reshape(-1, 3).astype(np.float32)

    valid = (z.ravel() > min_depth_m) & (z.ravel() < max_depth_m)
    pts_3d = pts_3d[valid]
    colors = colors[valid]

    n = len(pts_3d)
    if n == 0:
        return np.zeros((num_points, 6), dtype=np.float32)

    if n >= num_points:
        idxs = voxel_grid_sampling(pts_3d, num_points)
        pts_3d = pts_3d[idxs]
        colors = colors[idxs]
        # Pad to exactly num_points if voxel grid returned fewer
        if len(pts_3d) < num_points:
            # Zero-pad remainder (matching training data convention)
            pad = num_points - len(pts_3d)
            pts_3d = np.vstack([pts_3d, np.zeros((pad, 3), dtype=np.float32)])
            colors = np.vstack([colors, np.zeros((pad, 3), dtype=np.float32)])
    else:
        # Fewer than num_points valid points — zero-pad (matching training)
        pad = num_points - n
        pts_3d = np.vstack([pts_3d, np.zeros((pad, 3), dtype=np.float32)])
        colors = np.vstack([colors, np.zeros((pad, 3), dtype=np.float32)])

    pc = np.concatenate([pts_3d, colors], axis=1).astype(np.float32)
    return pc[:num_points]


# ---------------------------------------------------------------------------
# Camera manager
# ---------------------------------------------------------------------------

class DualCameraManager:
    """
    Manages the dual-camera setup:
      - agentview: external RealSense camera (fixed workspace view)
      - eye_in_hand: Spot built-in gripper camera OR second RealSense

    Camera parameters are fixed at 320 × 240, fx=fy=276, cx=160, cy=120
    to match the training data conversion exactly.
    """

    # Intrinsics used during training data conversion
    # (must match convert_spot_hdf5_to_zarr.py exactly)
    IMG_W: int = 320
    IMG_H: int = 240
    FX: float = 276.0
    FY: float = 276.0
    CX: float = 160.0
    CY: float = 120.0
    DEPTH_SCALE: float = 0.001   # uint16 → metres

    def __init__(
        self,
        spot_images,                        # SpotImages instance
        agentview_serial: Optional[str],
        hand_serial: Optional[str] = None,
        use_spot_hand_camera: bool = True,
        fps: int = 30,
        num_points: int = 8192,
    ) -> None:
        """
        Parameters
        ----------
        spot_images          : SpotImages instance (from SpotRobotController)
        agentview_serial     : RealSense serial for external camera
                               (None => auto-select first available device)
        hand_serial          : RealSense serial for hand camera (if not using
                               Spot's built-in camera)
        use_spot_hand_camera : Use Spot's built-in gripper camera for eye_in_hand
        fps                  : Capture frame rate for RealSense cameras
        num_points           : Points per cloud (must match training, default 8192)
        """
        try:
            from camera_streamer import CameraStreamer
        except Exception:
            from spot_teleop.camera_streamer import CameraStreamer
        try:
            from spot_images import CameraSource
        except Exception:
            from spot_teleop.spot_images import CameraSource

        self.spot_images = spot_images
        self.num_points = num_points
        self.use_spot_hand_camera = use_spot_hand_camera

        # --- Agentview (external RealSense) ---------------------------------
        if agentview_serial:
            print(f"[DualCameraManager] Starting agentview RealSense "
                  f"(serial={agentview_serial}, {self.IMG_W}x{self.IMG_H}@{fps}fps)")
        else:
            print(f"[DualCameraManager] Starting agentview RealSense "
                  f"(auto-select, {self.IMG_W}x{self.IMG_H}@{fps}fps)")
        self._agentview = CameraStreamer(
            device_serial=agentview_serial,
            width=self.IMG_W,
            height=self.IMG_H,
            fps=fps,
        )
        ok = self._agentview.start()
        if not ok:
            serial_info = f"serial={agentview_serial}" if agentview_serial else "auto-selected device"
            raise RuntimeError(
                f"[DualCameraManager] Failed to start agentview camera "
                f"({serial_info}). Check connection."
            )

        # --- Eye-in-hand (Spot gripper OR external RealSense) ---------------
        if use_spot_hand_camera:
            print("[DualCameraManager] Using Spot built-in gripper camera "
                  "(hand_color_image + hand_depth_in_hand_color_frame)")
            self._hand_cam = None  # use spot_images directly
            self._hand_cam_sources = [CameraSource("hand", ["visual", "depth_registered"])]
        else:
            if hand_serial is None:
                raise ValueError(
                    "hand_serial is required when use_spot_hand_camera=False"
                )
            print(f"[DualCameraManager] Starting hand RealSense "
                  f"(serial={hand_serial}, {self.IMG_W}x{self.IMG_H}@{fps}fps)")
            self._hand_cam = CameraStreamer(
                device_serial=hand_serial,
                width=self.IMG_W,
                height=self.IMG_H,
                fps=fps,
                align_depth_to_color=True,
            )
            ok = self._hand_cam.start()
            if not ok:
                raise RuntimeError(
                    f"[DualCameraManager] Failed to start hand RealSense "
                    f"(serial={hand_serial}). Check connection."
                )

        print("[DualCameraManager] Camera initialisation complete.")

    def capture(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture one frame from each camera and return point clouds.

        Returns
        -------
        agentview_pc   : (num_points, 6) float32 [X, Y, Z, R, G, B]
        eye_in_hand_pc : (num_points, 6) float32 [X, Y, Z, R, G, B]
        """
        agentview_pc = self._capture_agentview()
        eye_in_hand_pc = self._capture_eye_in_hand()
        return agentview_pc, eye_in_hand_pc

    def get_display_images(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return (agentview_bgr, hand_bgr) for on-screen display (cv2.imshow expects BGR)."""
        try:
            color_ag, _ = self._get_realsense_frames(self._agentview, "agentview")
        except RuntimeError:
            color_ag = np.zeros((self.IMG_H, self.IMG_W, 3), dtype=np.uint8)

        if self.use_spot_hand_camera:
            try:
                from utils.spot_utils import image_to_cv
            except Exception:
                from spot_teleop.utils.spot_utils import image_to_cv
            imgs = self.spot_images.get_images_by_cameras(self._hand_cam_sources)
            if imgs:
                hand_bgr = image_to_cv(imgs[0].image_response)
                if hand_bgr.shape[:2] != (self.IMG_H, self.IMG_W):
                    hand_bgr = cv2.resize(hand_bgr, (self.IMG_W, self.IMG_H),
                                          interpolation=cv2.INTER_AREA)
            else:
                hand_bgr = np.zeros((self.IMG_H, self.IMG_W, 3), dtype=np.uint8)
        else:
            hand_bgr, _ = self._hand_cam.get_latest()
            if hand_bgr is None:
                hand_bgr = np.zeros((self.IMG_H, self.IMG_W, 3), dtype=np.uint8)

        return color_ag, hand_bgr

    def stop(self) -> None:
        """Stop all camera threads."""
        self._agentview.stop()
        if self._hand_cam is not None:
            self._hand_cam.stop()
        print("[DualCameraManager] Cameras stopped.")

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _get_realsense_frames(self, cam, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Retry-wait for valid frames from a RealSense camera (up to 2s).

        Returns BGR color (matching CameraStreamer's bgr8 format) and uint16 depth.
        Training data was stored in BGR order — DO NOT convert to RGB.
        """
        deadline = time.monotonic() + 2.0
        while True:
            color_bgr, depth = cam.get_latest()
            if color_bgr is not None and depth is not None:
                return color_bgr, depth
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"[DualCameraManager] Timed out waiting for frames from {name} camera"
                )
            time.sleep(0.01)

    def _capture_agentview(self) -> np.ndarray:
        try:
            color_bgr, depth_u16 = self._get_realsense_frames(self._agentview, "agentview")
        except RuntimeError as e:
            print(f"[DualCameraManager] WARNING: {e}, using zeros")
            return np.zeros((self.num_points, 6), dtype=np.float32)

        # Resize to training resolution if needed
        if color_bgr.shape[:2] != (self.IMG_H, self.IMG_W):
            color_bgr = cv2.resize(color_bgr, (self.IMG_W, self.IMG_H),
                                   interpolation=cv2.INTER_AREA)
        if depth_u16.shape[:2] != (self.IMG_H, self.IMG_W):
            depth_u16 = cv2.resize(depth_u16, (self.IMG_W, self.IMG_H),
                                   interpolation=cv2.INTER_NEAREST)

        # Hardcode depth_scale=0.001 to match convert_spot_hdf5_to_zarr.py
        # (training data was created with this fixed scale, NOT the camera's actual scale)
        return rgbd_to_pointcloud(
            color_bgr, depth_u16,
            self.FX, self.FY, self.CX, self.CY,
            depth_scale=self.DEPTH_SCALE,
            num_points=self.num_points,
        )

    def _capture_eye_in_hand(self) -> np.ndarray:
        if self.use_spot_hand_camera:
            return self._capture_spot_hand_camera()
        else:
            try:
                color_bgr, depth_u16 = self._get_realsense_frames(self._hand_cam, "hand")
            except RuntimeError as e:
                print(f"[DualCameraManager] WARNING: {e}, using zeros")
                return np.zeros((self.num_points, 6), dtype=np.float32)

            if color_bgr.shape[:2] != (self.IMG_H, self.IMG_W):
                color_bgr = cv2.resize(color_bgr, (self.IMG_W, self.IMG_H),
                                       interpolation=cv2.INTER_AREA)
            if depth_u16.shape[:2] != (self.IMG_H, self.IMG_W):
                depth_u16 = cv2.resize(depth_u16, (self.IMG_W, self.IMG_H),
                                       interpolation=cv2.INTER_NEAREST)

            return rgbd_to_pointcloud(
                color_bgr, depth_u16,
                self.FX, self.FY, self.CX, self.CY,
                depth_scale=self.DEPTH_SCALE,
                num_points=self.num_points,
            )

    def _capture_spot_hand_camera(self) -> np.ndarray:
        """Get point cloud from Spot's built-in gripper camera."""
        try:
            from utils.spot_utils import image_to_cv
        except Exception:
            from spot_teleop.utils.spot_utils import image_to_cv
        from bosdyn.api import image_pb2

        imgs = self.spot_images.get_images_by_cameras(self._hand_cam_sources)
        if imgs is None or len(imgs) < 2:
            print("[DualCameraManager] WARNING: Spot hand camera images unavailable")
            return np.zeros((self.num_points, 6), dtype=np.float32)

        # imgs[0] = visual (BGR), imgs[1] = depth_registered (uint16)
        color_bgr = image_to_cv(imgs[0].image_response)
        depth_raw = image_to_cv(imgs[1].image_response)   # uint16 array

        if color_bgr.ndim == 2:   # safety: convert gray to BGR
            color_bgr = cv2.cvtColor(color_bgr, cv2.COLOR_GRAY2BGR)

        # Training data stored Spot hand frames in BGR (from image_to_cv) — keep BGR.

        # Resize Spot hand camera to match training resolution (320x240)
        if color_bgr.shape[:2] != (self.IMG_H, self.IMG_W):
            color_bgr = cv2.resize(color_bgr, (self.IMG_W, self.IMG_H),
                                   interpolation=cv2.INTER_AREA)
        if depth_raw.shape[:2] != (self.IMG_H, self.IMG_W):
            depth_raw = cv2.resize(depth_raw, (self.IMG_W, self.IMG_H),
                                   interpolation=cv2.INTER_NEAREST)

        return rgbd_to_pointcloud(
            color_bgr, depth_raw,
            self.FX, self.FY, self.CX, self.CY,
            depth_scale=self.DEPTH_SCALE,
            num_points=self.num_points,
        )


# ---------------------------------------------------------------------------
# Observation collector
# ---------------------------------------------------------------------------

class SpotObservationCollector:
    """
    Collects robot observations matching the training data format exactly.

    State vector (20-dim, float32):
        [0:7]   arm_q        – 7 arm joint positions (radians)
                               order: sh0, sh1, el0, el1, wr0, wr1, f1x
        [7:10]  ee_pos       – EE position (x, y, z) in BODY frame (metres)
        [10:19] ee_rot_flat  – EE rotation matrix (3×3) flattened row-major
        [19]    gripper      – gripper openness ∈ [0, 1]
    """

    ARM_JOINT_ORDER = [
        "arm0.sh0", "arm0.sh1", "arm0.el0",
        "arm0.el1", "arm0.wr0", "arm0.wr1", "arm0.f1x",
    ]

    def __init__(self, controller, camera_manager: DualCameraManager) -> None:
        self.controller = controller
        self.camera_manager = camera_manager

    def get_state(self) -> np.ndarray:
        """
        Extract 20-dim robot state vector.

        Returns
        -------
        state : (20,) float32
        """
        robot_state = self.controller.current_state()
        kin = robot_state.kinematic_state

        # --- 1. Arm joint angles (7) ----------------------------------------
        joint_map = {j.name: j.position.value for j in kin.joint_states}
        arm_q = np.array(
            [joint_map.get(name, 0.0) for name in self.ARM_JOINT_ORDER],
            dtype=np.float32,
        )

        # --- 2. EE pose in body frame (pos:3, quat:4) -----------------------
        ee_pos, ee_quat = self.controller.current_ee_pose()   # BODY_FRAME_NAME
        ee_pos = ee_pos.astype(np.float32)

        # --- 3. EE rotation matrix (9) --------------------------------------
        R = quat_to_rotmat(ee_quat)              # (3, 3)
        ee_rot_flat = R.flatten().astype(np.float32)  # row-major

        # --- 4. Gripper openness [0, 1] (1) ---------------------------------
        gripper_pct = self.controller.current_gripper()      # 0–100 percentage
        g = np.array([gripper_pct / 100.0], dtype=np.float32)

        state = np.concatenate([arm_q, ee_pos, ee_rot_flat, g])
        assert state.shape == (20,), f"Expected (20,), got {state.shape}"
        return state

    def get_observation(self) -> Dict[str, np.ndarray]:
        """
        Capture one full observation dict.

        Returns
        -------
        obs : dict with keys matching shape_meta:
              'agentview_point_cloud'   : (8192, 6)
              'eye_in_hand_point_cloud' : (8192, 6)
              'state'                  : (20,)
        """
        agentview_pc, eye_in_hand_pc = self.camera_manager.capture()
        state = self.get_state()
        return {
            "agentview_point_cloud": agentview_pc,
            "eye_in_hand_point_cloud": eye_in_hand_pc,
            "state": state,
        }


# ---------------------------------------------------------------------------
# Main rollout loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Roll out a trained DP3 policy on Spot with dual cameras."
    )
    parser.add_argument("--ckpt", required=True,
                        help="Path to .ckpt checkpoint file")
    parser.add_argument("--device", default="cuda:0",
                        help="PyTorch device, e.g. cuda:0 or cpu")
    parser.add_argument("--control-hz", type=float, default=10.0,
                        help="Target control frequency in Hz")
    parser.add_argument("--steps-per-inference", type=int, default=1,
                        help="How many actions to execute per policy call")
    parser.add_argument("--max-seconds", type=float, default=180.0,
                        help="Maximum rollout duration in seconds")
    # Camera
    parser.add_argument("--agentview-serial", default=None,
                        help="RealSense serial for agentview camera "
                             "(None => auto-select; or set SPOT_AGENTVIEW_REALSENSE_SERIAL env var)")
    parser.add_argument("--hand-serial", default=None,
                        help="RealSense serial for hand camera "
                             "(ignored if --use-spot-hand-camera)")
    parser.add_argument("--use-spot-hand-camera", action="store_true",
                        help="Use Spot's built-in gripper camera for eye_in_hand")
    parser.add_argument("--camera-fps", type=int, default=30,
                        help="RealSense capture frame rate")
    parser.add_argument("--num-points", type=int, default=8192,
                        help="Points per cloud (must match training)")
    # Robot
    parser.add_argument("--reset-pose", dest="reset_pose", action="store_true",
                        default=True,
                        help="Move arm to init pose before starting (default: enabled)")
    parser.add_argument("--no-reset-pose", dest="reset_pose", action="store_false",
                        help="Disable arm reset at startup")
    parser.add_argument("--init-pose", type=float, nargs=7,
                        default=[0.55, 0.0, 0.55, 0.0, 0.5, 0.0, 0.8660254],
                        metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
                        help="Startup arm pose [x,y,z,qx,qy,qz,qw] (used when --reset-pose)")
    parser.add_argument("--dock", action="store_true",
                        help="Dock robot after rollout ends")
    # Display
    parser.add_argument("--show", action="store_true",
                        help="Show live camera feed (requires display)")
    parser.add_argument("--start-paused", action="store_true",
                        help="Start in paused mode (preview cameras before policy actuation)")
    args = parser.parse_args()


    # --- import Spot SDK modules AFTER path is set --------------------------
    try:
        from spot_controller import SpotRobotController
    except Exception:
        from spot_teleop.spot_controller import SpotRobotController
    from bosdyn.client.frame_helpers import get_a_tform_b, BODY_FRAME_NAME

    # -----------------------------------------------------------------------
    # Load DP3 checkpoint
    # -----------------------------------------------------------------------
    print(f"\n[1/5] Loading checkpoint: {args.ckpt}")

    # Add DP3 source to sys.path for diffusion_policy_3d imports
    dp3_path = os.path.normpath(os.path.join(_THIS_DIR, "..", "3D-Diffusion-Policy"))
    if dp3_path not in sys.path:
        sys.path.insert(0, dp3_path)

    import copy

    payload = torch.load(open(args.ckpt, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    OmegaConf.set_struct(cfg, False)

    # Instantiate model directly from config (avoids importing train.py / wandb)
    policy = hydra.utils.instantiate(cfg.policy)
    ema_model = None
    if cfg.training.use_ema:
        ema_model = copy.deepcopy(policy)

    # Load saved weights
    policy.load_state_dict(payload["state_dicts"]["model"])
    if ema_model is not None and "ema_model" in payload["state_dicts"]:
        ema_model.load_state_dict(payload["state_dicts"]["ema_model"])

    if cfg.training.use_ema and ema_model is not None:
        policy = ema_model
    policy.eval().to(args.device)

    shape_meta = _get_shape_meta(cfg)
    obs_horizon = _get_obs_horizon(cfg)
    pc_keys, lowdim_keys = _extract_obs_keys(shape_meta)

    print(f"  obs_horizon     : {obs_horizon}")
    print(f"  point-cloud keys: {pc_keys}")
    print(f"  low-dim keys    : {lowdim_keys}")

    # -----------------------------------------------------------------------
    # Connect to Spot
    # -----------------------------------------------------------------------
    robot_ip = os.environ.get("SPOT_ROBOT_IP", "192.168.80.3")
    user     = os.environ.get("BOSDYN_CLIENT_USERNAME", "user")
    password = os.environ.get("BOSDYN_CLIENT_PASSWORD", "password")

    print(f"\n[2/5] Connecting to Spot at {robot_ip} ...")
    controller = SpotRobotController(
        robot_ip, user, password,
        default_exec_time=0.3,
    )
    print("  Connected.")

    print("  Undocking ...")
    controller.undock()
    print("  Waiting 4.0s after undock ...")
    time.sleep(1.0)

    if args.reset_pose:
        print(f"  Resetting arm to init pose: {args.init_pose}")
        controller.reset_pose(pose=list(args.init_pose))
        print("  Opening gripper after reset pose ...")
        try:
            controller.send_gripper(1.0)
            time.sleep(0.5)
        except Exception as e:
            print(f"  [warn] Failed to open gripper after reset pose: {e}")

    # Helper: immediately stop all robot motion and hold current pose
    def _stop_robot_motion():
        try:
            controller.move_base_with_velocity(0.0, 0.0, 0.0)
        except Exception as e:
            print(f"  [warn] Failed to send zero base velocity: {e}")
        try:
            ee_pos, ee_quat = controller.current_ee_pose(BODY_FRAME_NAME)
            grip = float(controller.current_gripper()) / 100.0
            controller.move_arm_to(ee_pos, ee_quat, gripper=grip,
                                   frame_name=BODY_FRAME_NAME)
        except Exception as e:
            print(f"  [warn] Failed to send arm hold command: {e}")

    # -----------------------------------------------------------------------
    # Initialise cameras
    # -----------------------------------------------------------------------
    # Resolve agentview serial: CLI arg > env var > None (auto-select)
    agentview_serial = args.agentview_serial or os.environ.get("SPOT_AGENTVIEW_REALSENSE_SERIAL")
    if agentview_serial:
        print(f"  Agentview RealSense serial: {agentview_serial}")
    else:
        print("  Agentview RealSense: auto-select (no serial specified)")

    print("\n[3/5] Initialising cameras ...")
    camera_manager = DualCameraManager(
        spot_images=controller.spot_images,
        agentview_serial=agentview_serial,
        hand_serial=args.hand_serial,
        use_spot_hand_camera=args.use_spot_hand_camera,
        fps=args.camera_fps,
        num_points=args.num_points,
    )

    obs_collector = SpotObservationCollector(controller, camera_manager)

    # -----------------------------------------------------------------------
    # Observation history buffers
    # -----------------------------------------------------------------------
    print("\n[4/5] Filling observation history buffers ...")
    history: Dict[str, deque] = {
        key: deque(maxlen=obs_horizon)
        for key in pc_keys + lowdim_keys
    }

    # seed history with obs_horizon frames
    for i in range(obs_horizon):
        obs_now = obs_collector.get_observation()
        for k in pc_keys + lowdim_keys:
            if k in obs_now:
                history[k].append(obs_now[k])
        time.sleep(0.05)

    # -----------------------------------------------------------------------
    # Control loop setup
    # -----------------------------------------------------------------------
    dt = 1.0 / args.control_hz
    plan: Optional[np.ndarray] = None
    plan_idx = 0
    step_count = 0
    start_time = time.monotonic()
    is_paused = bool(args.start_paused)
    stop_flag = False
    dock_requested = bool(args.dock)

    dp3_path = "/home/akash/UNH/demoGen_research/3D-Diffusion-Policy/3D-Diffusion-Policy"
    if dp3_path not in sys.path:
        sys.path.append(dp3_path)

    # Import here so sys.path is already set up regardless of install method
    from diffusion_policy_3d.common.pytorch_util import dict_apply

    # --- Keyboard / terminal command handling (from reference script) ------
    stdin_cmd_queue: queue.Queue = queue.Queue()

    def _handle_keypress(key_stroke: int):
        nonlocal stop_flag, is_paused, dock_requested
        if key_stroke < 0:
            return
        key_stroke = key_stroke & 0xFF
        key_lower = key_stroke
        if key_stroke != 27 and 0 <= key_stroke <= 255:
            key_lower = ord(chr(key_stroke).lower())

        if key_stroke == 27 or key_lower == ord('q'):
            name = "ESC" if key_stroke == 27 else "Q"
            print(f"\n[{name}] Quitting rollout and sending hold command...")
            stop_flag = True
            _stop_robot_motion()
        elif key_lower == ord('s'):
            print("\n[S] Emergency stop (hold current pose)!")
            stop_flag = True
            _stop_robot_motion()
        elif key_lower == ord('a'):
            is_paused = not is_paused
            print(f"\n[A] Arm actuation {'PAUSED' if is_paused else 'RESUMED'}")
        elif key_lower == ord('r'):
            if is_paused:
                is_paused = False
                print("\n[R] Policy RESUMED")
            else:
                print("\n[R] Policy already running")
        elif key_lower == ord('p'):
            if not is_paused:
                is_paused = True
                print("\n[P] Policy PAUSED (camera preview continues)")
            else:
                print("\n[P] Policy already paused")
        elif key_lower == ord('d'):
            print("\n[D] Stop rollout, stow arm, and dock at exit...")
            dock_requested = True
            stop_flag = True
            _stop_robot_motion()
        elif key_lower == ord('u'):
            print("\n[U] Undock request...")
            try:
                controller.undock()
            except Exception as e:
                print(f"  [warn] Undock failed: {e}")

    def _handle_terminal_command(cmd: str):
        cmd = (cmd or "").strip().lower()
        if not cmd:
            return
        cmd_map = {
            "q": ord("q"), "quit": ord("q"), "exit": ord("q"),
            "s": ord("s"), "stop": ord("s"), "estop": ord("s"),
            "a": ord("a"), "pause": ord("a"), "resume": ord("a"),
            "p": ord("p"),
            "r": ord("r"), "run": ord("r"), "start": ord("r"),
            "d": ord("d"), "dock": ord("d"),
            "u": ord("u"), "undock": ord("u"),
        }
        if cmd in cmd_map:
            _handle_keypress(cmd_map[cmd])
        elif cmd in ("h", "help", "?"):
            print("[cmds] q=quit, s=stop/hold, a=toggle, p=pause, r=resume, d=dock+stop, u=undock")
        else:
            print(f"[cmds] Unknown '{cmd}'. Use q/s/a/p/r/d/u (then Enter).")

    def _drain_terminal_commands():
        while True:
            try:
                cmd = stdin_cmd_queue.get_nowait()
            except queue.Empty:
                break
            _handle_terminal_command(cmd)

    def _stdin_reader():
        while True:
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if line == "":
                break
            stdin_cmd_queue.put(line)

    # --- Signal handling for graceful shutdown ---
    old_sig_handlers = {}

    def _signal_handler(signum, _frame):
        nonlocal stop_flag, dock_requested
        stop_flag = True
        dock_requested = True
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = str(signum)
        print(f"\n[{signame}] Stopping rollout and docking at exit...")
        try:
            _stop_robot_motion()
        except Exception:
            pass

    for _sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            old_sig_handlers[_sig] = signal.getsignal(_sig)
            signal.signal(_sig, _signal_handler)
        except Exception:
            pass

    # Start terminal command reader thread
    stdin_thread = None
    if sys.stdin is not None and sys.stdin.isatty():
        stdin_thread = threading.Thread(target=_stdin_reader, daemon=True)
        stdin_thread.start()

    # -----------------------------------------------------------------------
    # Start rollout
    # -----------------------------------------------------------------------
    print("\n[5/5] Starting rollout ...")
    print("=" * 60)
    print("  Q / ESC / Ctrl+C : quit")
    print("  S                : emergency stop (hold pose)")
    print("  A                : toggle pause/resume")
    print("  P                : pause")
    print("  R                : resume")
    print("  D                : dock + stop")
    print("  U                : undock")
    if args.show:
        print("  (Click the OpenCV window before pressing keys)")
    print("  Terminal fallback: q/s/a/p/r/d/u + Enter")
    print("=" * 60)

    try:
        while not stop_flag:
            loop_start = time.monotonic()

            # ---- timeout check -------------------------------------------
            elapsed = loop_start - start_time
            if elapsed > args.max_seconds:
                print(f"\n[rollout] Timeout ({args.max_seconds:.0f}s). Stopping.")
                break

            # ---- keyboard & terminal polling -----------------------------
            _drain_terminal_commands()
            if args.show:
                key = cv2.pollKey() & 0xFF
                if key != 255:  # 255 means no key pressed
                    _handle_keypress(key)
            if stop_flag:
                break

            # ---- capture current observation -----------------------------
            obs_now = obs_collector.get_observation()
            for k in pc_keys + lowdim_keys:
                if k in obs_now:
                    history[k].append(obs_now[k])

            # ---- if paused, just show camera feed and continue -----------
            if is_paused:
                if args.show:
                    color_ag, color_hand = camera_manager.get_display_images()
                    disp_ag   = cv2.resize(color_ag,   (320, 240))
                    disp_hand = cv2.resize(color_hand, (320, 240))
                    row = np.hstack([disp_ag, disp_hand])
                    cv2.putText(row, "Agentview",   (10,  25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    cv2.putText(row, "Eye-in-Hand", (330, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    cv2.putText(row, "PAUSED | q:quit s:hold r:resume d:dock",
                                (10, 230),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)
                    cv2.imshow("DP3 Dual Camera", row)
                    cv2.waitKey(1)
                elapsed_loop = time.monotonic() - loop_start
                time.sleep(max(0.0, dt - elapsed_loop))
                continue

            # ---- inference (if no plan remaining) -----------------------
            if plan is None or plan_idx >= args.steps_per_inference or plan_idx >= len(plan):
                # Build (1, T, ...) observation tensors
                obs_dict = {}
                for k in pc_keys:
                    arr = np.stack(list(history[k]), axis=0).astype(np.float32)
                    obs_dict[k] = torch.from_numpy(arr)
                for k in lowdim_keys:
                    arr = np.stack(list(history[k]), axis=0).astype(np.float32)
                    obs_dict[k] = torch.from_numpy(arr)

                obs_dict = dict_apply(obs_dict,
                                      lambda x: x.unsqueeze(0).to(args.device))

                t0 = time.monotonic()
                with torch.no_grad():
                    result = policy.predict_action(obs_dict)
                inf_ms = (time.monotonic() - t0) * 1000.0

                plan = result["action"][0].cpu().numpy()  # (T_action, 10)
                plan_idx = 0

                print(f"[step {step_count:4d}] Generated {len(plan)}-step plan "
                      f"(inference {inf_ms:.1f} ms)")

            # ---- execute action -----------------------------------------
            action = plan[plan_idx]
            plan_idx += 1

            print(f"[debug] step={step_count} plan_idx={plan_idx-1} action={action}")
            # Project rot6d to valid rotation before sending
            rot6d = action[3:9]
            R = project_rot6d_to_matrix(rot6d)
            # Replace rot6d in action with the projected 6D representation (first two columns of R)
            action[3:6] = R[:, 0]
            action[6:9] = R[:, 1]

            if not is_paused:
                try:
                    controller.apply_action(action)
                    print(f"[debug] action sent to controller.")
                except Exception as e:
                    print(f"[error] controller.apply_action failed: {e}")
            else:
                print("[debug] rollout paused, action not sent.")

            step_count += 1

            # ---- sleep to maintain control frequency --------------------
            elapsed_loop = time.monotonic() - loop_start
            sleep_t = max(0.0, dt - elapsed_loop)
            time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[rollout] Interrupted by Ctrl+C.")
        dock_requested = True
        try:
            _stop_robot_motion()
        except Exception:
            pass
    except Exception as exc:
        import traceback
        print(f"\n[rollout] Unexpected error: {exc}")
        traceback.print_exc()
        dock_requested = True
        try:
            _stop_robot_motion()
        except Exception:
            pass
    finally:
        print("\n[rollout] Cleaning up ...")
        # Stop motion first
        try:
            _stop_robot_motion()
        except Exception:
            pass

        # Dock if requested (D key, --dock flag, signal, or error)
        if dock_requested:
            print("  Stowing arm and docking robot ...")
            try:
                controller.stow_arm()
            except Exception as e:
                print(f"  [warn] stow_arm failed: {e}")
            try:
                controller.dock()
            except Exception as e:
                print(f"  [warn] dock failed: {e}")

        camera_manager.stop()

        # Restore original signal handlers
        for _sig, _handler in old_sig_handlers.items():
            try:
                signal.signal(_sig, _handler)
            except Exception:
                pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[rollout] Done.")


if __name__ == "__main__":
    main()
