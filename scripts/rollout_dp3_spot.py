#!/usr/bin/env python3
"""
rollout_dp3_spot.py
-------------------
Roll out a trained DP3 policy on Boston Dynamics Spot using dual-camera
point-cloud observations.

Observation format (matches training data exactly):
  agentview_point_cloud  : (8192, 6) XYZ + RGB (uint8 range 0-255)
  eye_in_hand_point_cloud: (8192, 6) XYZ + RGB (uint8 range 0-255)
  state                  : (21,) float32
      [ arm_q(7) | ee_pos(3) | ee_rot_flat(9) | gripper_dup(2) ]

Action format (10-dim):
  [ Δx Δy Δz (3) | rot6d (6) | gripper_abs (1) ]

Usage:
  python scripts/rollout_dp3_spot.py \\
      --ckpt data/outputs/.../checkpoints/latest.ckpt \\
      --agentview-serial 153122071664 \\
      --use-spot-hand-camera \\
      --spot-repo /path/to/spot-meta-teleop \\
      --control-hz 10 \\
      --show

Safety: keep one hand near the E-stop at all times.
"""
from __future__ import annotations

import argparse
import os
import sys
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
    min_depth_m: float = 0.05,
    max_depth_m: float = 2.5,
    num_points: int = 8192,
) -> np.ndarray:
    """
    Convert an RGB-D frame to an (N, 6) point cloud matching the training format.

    Colors are kept in uint8 range [0, 255] (NOT divided by 255) to match the
    values stored in the zarr training dataset.

    Parameters
    ----------
    color_bgr  : (H, W, 3) BGR image (OpenCV convention)
    depth_u16  : (H, W) uint16 depth, raw units → meters via depth_scale
    fx, fy     : focal lengths in pixels
    cx, cy     : principal point in pixels
    depth_scale: multiply raw depth by this to get metres (default: 0.001)
    Returns
    -------
    (num_points, 6) float32 array: [X, Y, Z, R, G, B]
    """
    h, w = depth_u16.shape
    depth_m = depth_u16.astype(np.float32) * depth_scale

    us, vs = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))

    z = depth_m
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy

    pts_3d = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Convert BGR → RGB and keep uint8 values (range 0-255, matching training)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    colors = color_rgb.reshape(-1, 3).astype(np.float32)  # uint8 values 0-255

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
            pad = num_points - len(pts_3d)
            pad_idxs = np.random.choice(len(pts_3d), pad, replace=True)
            pts_3d = np.vstack([pts_3d, pts_3d[pad_idxs]])
            colors = np.vstack([colors, colors[pad_idxs]])
    else:
        pad = num_points - n
        pad_idxs = np.random.choice(n, pad, replace=True)
        pts_3d = np.vstack([pts_3d, pts_3d[pad_idxs]])
        colors = np.vstack([colors, colors[pad_idxs]])

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

    Camera parameters are fixed at 320 × 240, fx=fy=320, cx=160, cy=120
    to match the training data conversion exactly.
    """

    # Intrinsics used during training data conversion
    IMG_W: int = 320
    IMG_H: int = 240
    FX: float = 320.0
    FY: float = 320.0
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
        agentview_serial     : RealSense serial for external camera (required)
        hand_serial          : RealSense serial for hand camera (if not using
                               Spot's built-in camera)
        use_spot_hand_camera : Use Spot's built-in gripper camera for eye_in_hand
        fps                  : Capture frame rate for RealSense cameras
        num_points           : Points per cloud (must match training, default 8192)
        """
        if agentview_serial is None:
            raise ValueError("agentview_serial is required")

        from camera_streamer import CameraStreamer
        from spot_images import CameraSource

        self.spot_images = spot_images
        self.num_points = num_points
        self.use_spot_hand_camera = use_spot_hand_camera

        # --- Agentview (external RealSense) ---------------------------------
        print(f"[DualCameraManager] Starting agentview RealSense "
              f"(serial={agentview_serial}, {self.IMG_W}x{self.IMG_H}@{fps}fps)")
        self._agentview = CameraStreamer(
            device_serial=agentview_serial,
            width=self.IMG_W,
            height=self.IMG_H,
            fps=fps,
            align_depth_to_color=True,
        )
        ok = self._agentview.start()
        if not ok:
            raise RuntimeError(
                f"[DualCameraManager] Failed to start agentview camera "
                f"(serial={agentview_serial}). Check connection."
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
        """Return (agentview_bgr, hand_bgr) for on-screen display."""
        color_ag, _ = self._agentview.get_latest()
        if color_ag is None:
            color_ag = np.zeros((self.IMG_H, self.IMG_W, 3), dtype=np.uint8)

        if self.use_spot_hand_camera:
            from utils.spot_utils import image_to_cv
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

    def _capture_agentview(self) -> np.ndarray:
        color_bgr, depth_u16 = self._agentview.get_latest()
        if color_bgr is None or depth_u16 is None:
            print("[DualCameraManager] WARNING: agentview frame is None, using zeros")
            return np.zeros((self.num_points, 6), dtype=np.float32)

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
            color_bgr, depth_u16 = self._hand_cam.get_latest()
            if color_bgr is None or depth_u16 is None:
                print("[DualCameraManager] WARNING: hand RealSense frame is None")
                return np.zeros((self.num_points, 6), dtype=np.float32)
            return rgbd_to_pointcloud(
                color_bgr, depth_u16,
                self.FX, self.FY, self.CX, self.CY,
                depth_scale=self.DEPTH_SCALE,
                num_points=self.num_points,
            )

    def _capture_spot_hand_camera(self) -> np.ndarray:
        """Get point cloud from Spot's built-in gripper camera."""
        from utils.spot_utils import image_to_cv
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

    State vector (21-dim, float32):
        [0:7]   arm_q        – 7 arm joint positions (radians)
                               order: sh0, sh1, el0, el1, wr0, wr1, f1x
        [7:10]  ee_pos       – EE position (x, y, z) in BODY frame (metres)
        [10:19] ee_rot_flat  – EE rotation matrix (3×3) flattened row-major
        [19:21] gripper_dup  – gripper openness ∈ [0, 1] duplicated
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
        Extract 21-dim robot state vector.

        Returns
        -------
        state : (21,) float32
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

        # --- 4. Gripper openness [0, 1] duplicated (2) ----------------------
        gripper_pct = self.controller.current_gripper()      # 0–100 percentage
        g = np.float32(gripper_pct / 100.0)
        gripper_dup = np.array([g, g], dtype=np.float32)

        state = np.concatenate([arm_q, ee_pos, ee_rot_flat, gripper_dup])
        assert state.shape == (21,), f"Expected (21,), got {state.shape}"
        return state

    def get_observation(self) -> Dict[str, np.ndarray]:
        """
        Capture one full observation dict.

        Returns
        -------
        obs : dict with keys matching shape_meta:
              'agentview_point_cloud'   : (8192, 6)
              'eye_in_hand_point_cloud' : (8192, 6)
              'state'                  : (21,)
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
    parser.add_argument("--max-seconds", type=float, default=60.0,
                        help="Maximum rollout duration in seconds")
    # Camera
    parser.add_argument("--agentview-serial", default=None,
                        help="RealSense serial number for agentview camera")
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
    _default_spot_repo = os.environ.get(
        "SPOT_META_TELEOP_DIR",
        # Sibling directory of 3D-Diffusion-Policy (works on any machine
        # when both repos are cloned in the same parent folder)
        os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "spot-meta-teleop"))
    )
    parser.add_argument("--spot-repo",
                        default=_default_spot_repo,
                        help="Path to spot-meta-teleop repo. "
                             "Override with SPOT_META_TELEOP_DIR env var.")
    parser.add_argument("--undock", action="store_true",
                        help="Undock robot before starting")
    parser.add_argument("--reset-pose", action="store_true",
                        help="Move arm to default ready pose before starting")
    parser.add_argument("--dock", action="store_true",
                        help="Dock robot after rollout ends")
    # Display
    parser.add_argument("--show", action="store_true",
                        help="Show live camera feed (requires display)")
    args = parser.parse_args()

    # --- validate spot repo --------------------------------------------------
    if not os.path.isdir(args.spot_repo):
        raise FileNotFoundError(
            f"spot-meta-teleop not found at: {args.spot_repo}\n"
            f"  Set --spot-repo or SPOT_META_TELEOP_DIR env var."
        )
    _add_to_path(args.spot_repo)
    _add_to_path(os.path.join(args.spot_repo, "utils"))

    # --- import Spot SDK modules AFTER path is set --------------------------
    from spot_controller import SpotRobotController
    from bosdyn.client.frame_helpers import BODY_FRAME_NAME

    # -----------------------------------------------------------------------
    # Load DP3 checkpoint
    # -----------------------------------------------------------------------
    print(f"\n[1/5] Loading checkpoint: {args.ckpt}")
    with open(args.ckpt, "rb") as f:
        payload = torch.load(f, pickle_module=dill)

    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
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
        arm_base_frame=BODY_FRAME_NAME,
        default_exec_time=1.0 / args.control_hz,
    )
    print("  Connected.")

    if args.undock:
        print("  Undocking ...")
        controller.undock()

    controller.stand()
    controller.unstow_arm()

    if args.reset_pose:
        print("  Moving arm to ready pose ...")
        controller.reset_pose()
        time.sleep(1.0)

    # -----------------------------------------------------------------------
    # Initialise cameras
    # -----------------------------------------------------------------------
    print("\n[3/5] Initialising cameras ...")
    camera_manager = DualCameraManager(
        spot_images=controller.spot_images,
        agentview_serial=args.agentview_serial,
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
    # Control loop
    # -----------------------------------------------------------------------
    print("\n[5/5] Starting rollout ...")
    print("=" * 60)
    print("  Q / Ctrl+C : quit")
    print("  S          : emergency stop arm")
    print("  A          : pause/resume arm actuation")
    print("=" * 60)

    dt = 1.0 / args.control_hz
    plan: Optional[np.ndarray] = None
    plan_idx = 0
    step_count = 0
    start_time = time.monotonic()
    is_paused = False
    stop_flag = False

    # Import here so sys.path is already set up regardless of install method
    from diffusion_policy_3d.common.pytorch_util import dict_apply

    try:
        while not stop_flag:
            loop_start = time.monotonic()

            # ---- timeout check -------------------------------------------
            elapsed = loop_start - start_time
            if elapsed > args.max_seconds:
                print(f"\n[rollout] Timeout ({args.max_seconds:.0f}s). Stopping.")
                break

            # ---- keyboard polling ----------------------------------------
            if args.show:
                key = cv2.pollKey() & 0xFF
                if key == ord('q'):
                    print("\n[Q] Quitting rollout.")
                    stop_flag = True
                    continue
                elif key == ord('s'):
                    print("\n[S] Emergency stop – stowing arm.")
                    controller.stow_arm()
                    stop_flag = True
                    continue
                elif key == ord('a'):
                    is_paused = not is_paused
                    print(f"\n[A] Arm actuation {'PAUSED' if is_paused else 'RESUMED'}")

            # ---- capture current observation -----------------------------
            obs_now = obs_collector.get_observation()
            for k in pc_keys + lowdim_keys:
                if k in obs_now:
                    history[k].append(obs_now[k])

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

            if not is_paused:
                controller.apply_action(action)

            # ---- optional display ---------------------------------------
            if args.show:
                color_ag, color_hand = camera_manager.get_display_images()
                disp_ag   = cv2.resize(color_ag,   (320, 240))
                disp_hand = cv2.resize(color_hand, (320, 240))
                row = np.hstack([disp_ag, disp_hand])
                cv2.putText(row, "Agentview",   (10,  25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(row, "Eye-in-Hand", (330, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(row,
                            f"step {step_count}  {'PAUSED' if is_paused else ''}",
                            (10, 460),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
                cv2.imshow("DP3 Dual Camera", row)
                cv2.waitKey(1)

            step_count += 1

            # ---- sleep to maintain control frequency --------------------
            elapsed_loop = time.monotonic() - loop_start
            sleep_t = max(0.0, dt - elapsed_loop)
            time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[rollout] Interrupted by Ctrl+C.")
    except Exception as exc:
        import traceback
        print(f"\n[rollout] Unexpected error: {exc}")
        traceback.print_exc()
    finally:
        print("\n[rollout] Cleaning up ...")
        try:
            controller.stow_arm()
        except Exception as e:
            print(f"  stow_arm failed: {e}")

        camera_manager.stop()

        if args.dock:
            print("  Docking robot ...")
            controller.dock()

        cv2.destroyAllWindows()
        print("[rollout] Done.")


if __name__ == "__main__":
    main()
