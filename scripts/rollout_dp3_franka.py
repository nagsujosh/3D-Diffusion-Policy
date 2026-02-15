#!/usr/bin/env python3
"""
DP3 (3D Diffusion Policy) rollout on a real Franka robot via python-deoxys.

Action pipeline (must match data collection):
  1. Policy predicts actions in normalised space (limits → [-1, 1]).
  2. The DP3 normalizer stored inside the checkpoint un-normalises them
     back to the HDF5 action space (pre-scaled SpaceMouse values).
  3. The OSC controller YAML applies action_scale internally:
       translation × 0.05
       rotation    × 1.0
  4. Therefore we send policy output DIRECTLY to the controller — no extra
     scaling is needed.

SMOOTH EXECUTION — Receding Horizon with Temporal Ensemble:
  1. Predict K future actions (K = horizon, e.g. 4).
  2. Execute only `n_exec` actions (default 1 → most reactive).
  3. Re-predict from a fresh observation.

  Overlapping predictions are AVERAGED (temporal ensemble) to reduce noise.
  An optional EMA filter further smooths the final commands.
  Gripper hysteresis prevents chattering.

Observation format (must match training zarr):
  agentview_point_cloud   : (B, T_o, 8192, 6)   — xyz + rgb  (use_pc_color=True)
  eye_in_hand_point_cloud : (B, T_o, 8192, 6)   — xyz + rgb  (use_pc_color=True)
  agent_pos               : (B, T_o, 21)        — joint_states(7) + ee_position(3) + ee_rotation(9) + gripper(2)

Checkpoint loading:
  The DP3 normalizer is part of the model's state_dict (nn.Module param
  buffers), so it is restored automatically via load_payload — no need
  for a separate zarr pass.  An optional --zarr flag is kept so you can
  double-check action statistics at startup.


Run: 
# Dual camera (agentview + eye_in_hand) with Student-t (default: 3 samples)
python scripts/rollout_dp3_franka.py \
    --ckpt path/to/epoch=0340-train_loss=0.145.ckpt \
    --camera_ids 0 1

# With optional sanity check against training zarr
python scripts/rollout_dp3_franka.py \
    --ckpt path/to/latest.ckpt \
    --zarr data/stacking_cups_dual_view.zarr \
    --camera_ids 0 1

# Disable Student-t multi-sample (fastest, single inference)
python scripts/rollout_dp3_franka.py \
    --ckpt path/to/latest.ckpt \
    --camera_ids 0 1 \
    --no_student_t

# More samples for higher robustness (5 forward passes)
python scripts/rollout_dp3_franka.py \
    --ckpt path/to/latest.ckpt \
    --n_samples 5 \
    --use_student_t \
    --camera_ids 0 1

# Full control with all smoothing features
python scripts/rollout_dp3_franka.py \
    --ckpt path/to/latest.ckpt \
    --n_action_steps 2 \
    --max_steps 500 \
    --n_trials 3 \
    --ema_alpha 0.7 \
    --temporal_weight 0.1 \
    --n_samples 3 \
    --use_student_t \
    --student_t_fast \
    --camera_ids 0 1 \
    --num_points 8192 \
    --control_hz 20
"""

import os
import sys
import time
import argparse
import random
import pathlib
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from collections import deque

import cv2
import numpy as np
import torch
import dill
import imageio
import open3d as o3d
from scipy.stats import t as student_t

# ============================================================
# Path setup  — adjust these to your Franka workstation
# ============================================================
sys.path.append("/home/franka_deoxys/deoxys_control/deoxys")
sys.path.append("/home/franka_deoxys/deoxys_vision")

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.config_utils import robot_config_parse_args
from deoxys.experimental.motion_utils import reset_joints_to

from deoxys_vision.networking.camera_redis_interface import CameraRedisSubInterface
from deoxys_vision.utils.camera_utils import assert_camera_ref_convention, get_camera_info

# ============================================================
# DP3 imports — handle both pip-installed and manual paths
# ============================================================
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)

# Try importing directly first (if pip installed with 'pip install -e .')
try:
    from train import TrainDP3Workspace
    from diffusion_policy_3d.policy.dp3 import DP3
    print("[DP3] Imported from pip-installed package")
except ImportError:
    # Fall back to manual path detection
    print("[DP3] Package not found, trying manual path detection...")
    
    DP3_ROOT = os.environ.get("DP3_ROOT")
    if DP3_ROOT is None:
        # Try common locations
        possible_paths = [
            "/your/path/3D-Diffusion-Policy/3D-Diffusion-Policy",
            "/your/path/3D-Diffusion-Policy/3D-Diffusion-Policy",
            str(pathlib.Path(__file__).resolve().parent.parent / "3D-Diffusion-Policy"),
            str(pathlib.Path(__file__).resolve().parent.parent),
        ]
        for path in possible_paths:
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "train.py")):
                DP3_ROOT = path
                break
        
        if DP3_ROOT is None:
            raise RuntimeError(
                "Could not find 3D-Diffusion-Policy installation. "
                "Either pip install it with 'pip install -e .' or set DP3_ROOT environment variable."
            )

    print(f"[DP3] Using DP3_ROOT: {DP3_ROOT}")
    
    # Verify train.py exists
    train_py_path = os.path.join(DP3_ROOT, "train.py")
    if not os.path.exists(train_py_path):
        raise RuntimeError(
            f"Could not find train.py at {train_py_path}. "
            f"Please verify DP3_ROOT points to the correct directory."
        )

    sys.path.insert(0, DP3_ROOT)
    
    # Save original directory and change to DP3_ROOT for imports
    _original_cwd = os.getcwd()
    os.chdir(DP3_ROOT)
    
    from train import TrainDP3Workspace
    from diffusion_policy_3d.policy.dp3 import DP3
    
    # Change back to original directory
    os.chdir(_original_cwd)


# ============================================================
# Deterministic seeding
# ============================================================
def set_global_seeds(seed: int):
    print(f"[SEED] Setting global seed = {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# ============================================================
# Geometry helpers  (must match extract_point_cloud_from_depth
# in scripts/convert_franka_hdf5_to_zarr.py)
# ============================================================
def voxel_grid_sampling(points: np.ndarray, num_samples: int) -> np.ndarray:
    """Deterministic voxel-grid sub-sampling → exactly `num_samples` indices."""
    N = len(points)
    if N <= num_samples:
        return np.arange(N)

    pts_min = points.min(axis=0)
    bbox_size = points.max(axis=0) - pts_min + 1e-6
    voxel_size = np.cbrt(bbox_size.prod() / num_samples)
    if voxel_size < 1e-6:
        voxel_size = bbox_size.max() / np.cbrt(num_samples)

    voxel_idx = np.floor((points - pts_min) / voxel_size).astype(np.int64)
    keys = (
        voxel_idx[:, 0] * 73856093
        + voxel_idx[:, 1] * 19349663
        + voxel_idx[:, 2] * 83492791
    )
    _, first = np.unique(keys, return_index=True)
    selected = np.sort(first)

    if len(selected) >= num_samples:
        selected = selected[:num_samples]
    else:
        remaining = np.ones(N, dtype=bool)
        remaining[selected] = False
        extra = np.where(remaining)[0]
        needed = num_samples - len(selected)
        step = max(1, len(extra) // needed)
        selected = np.concatenate([selected, extra[::step][:needed]])

    return selected[:num_samples].astype(np.int32)


def depth_to_pointcloud(
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    num_points: int = 1024,
    fx: float = 320.0,
    fy: float = 320.0,
    cx: float = 160.0,
    cy: float = 120.0,
    depth_scale: float = 1.0,
    normalize_rgb: bool = True,
) -> np.ndarray:
    """
    Back-project an RGB-D pair into a point cloud of shape (num_points, 6).

    The 6 channels are [x, y, z, r, g, b].
    - If `normalize_rgb` is True  →  colours in [0, 1]   (matches zarr converter).
    - If `normalize_rgb` is False →  colours in [0, 255].

    Returns zeros when the depth image contains no valid pixels.
    """
    H, W = depth_image.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u_f, v_f = u.flatten(), v.flatten()
    d_f = depth_image.flatten().astype(np.float32) * depth_scale

    valid = d_f > 0.01
    if not np.any(valid):
        return np.zeros((num_points, 6), dtype=np.float32)

    u_v, v_v, d_v = u_f[valid], v_f[valid], d_f[valid]
    x = (u_v - cx) * d_v / fx
    y = (v_v - cy) * d_v / fy
    z = d_v

    color = rgb_image.reshape(-1, 3)[valid].astype(np.float32)
    if normalize_rgb:
        color = color / 255.0

    pts = np.column_stack([x, y, z, color]).astype(np.float32)

    out = np.zeros((num_points, 6), dtype=np.float32)
    n = pts.shape[0]
    if n >= num_points:
        idx = voxel_grid_sampling(pts[:, :3], num_points)
        sampled = pts[idx]
        out[: len(sampled)] = sampled
    else:
        out[:n] = pts
    return out


def visualize_pointcloud(pc: np.ndarray, window_name: str = "Point Cloud"):
    """Quick Open3D visualisation of a (N, 6) point cloud."""
    xyz = pc[:, :3]
    rgb = pc[:, 3:6]
    mask = np.any(xyz != 0, axis=1)
    if not np.any(mask):
        print("[O3D] No valid points.")
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[mask].astype(np.float64))
    # Colours may already be [0, 1]; clamp just in case
    colours = np.clip(rgb[mask].astype(np.float64), 0.0, 1.0)
    pcd.colors = o3d.utility.Vector3dVector(colours)
    o3d.visualization.draw_geometries([pcd], window_name=window_name)


# ============================================================
# Build 21-D agent_pos  (must match convert_franka_hdf5_to_zarr)
#   joint_states (7) + ee_position (3) + ee_rotation (9) + gripper_states (2)
# ============================================================
def agent_pos_21_from_robot(robot: FrankaInterface) -> np.ndarray:
    """
    Read the latest Franka state buffers and pack them into the
    21-dim agent_pos vector that DP3 expects:

        [ joint_states (7), ee_position (3), ee_rotation (9), gripper_states (2) ]

    ee_position is extracted from the 4×4 homogeneous transform O_T_EE.
    ee_rotation is the 3×3 rotation matrix flattened.
    gripper_states contains [width, width] for symmetric parallel gripper.
    """
    if len(robot._state_buffer) == 0 or len(robot._gripper_state_buffer) == 0:
        raise RuntimeError("Robot state / gripper buffers are empty — is the robot connected?")

    st = robot._state_buffer[-1]
    gs = robot._gripper_state_buffer[-1]

    # Joint positions (7 DOF arm)
    joint_states = np.array(st.q, dtype=np.float32)[:7]  # (7,)

    # EE pose: libfranka stores O_T_EE column-major (16 floats)
    # Convert to 4×4 matrix (row-major)
    ee_pose_flat = np.array(st.O_T_EE, dtype=np.float32)  # (16,)
    ee_pose_matrix = ee_pose_flat.reshape(4, 4).T  # Transpose from column-major to row-major
    
    # Extract position (3 elements)
    ee_position = ee_pose_matrix[:3, 3]  # (3,)
    
    # Extract rotation matrix and flatten (9 elements)
    ee_rotation = ee_pose_matrix[:3, :3].flatten()  # (9,)

    # Gripper width → symmetric gripper (2 elements)
    width = float(getattr(gs, "width", 0.04))
    gripper_states = np.array([width, width], dtype=np.float32)  # (2,)

    agent_pos = np.concatenate([joint_states, ee_position, ee_rotation, gripper_states])
    assert agent_pos.shape == (21,), f"Expected (21,), got {agent_pos.shape}"
    return agent_pos


# ============================================================
# FrameStack — keeps last T_o observations
# ============================================================
class FrameStack:
    def __init__(self, n_obs_steps: int):
        self.T = n_obs_steps
        self.agentview_pc = deque(maxlen=n_obs_steps)
        self.eye_in_hand_pc = deque(maxlen=n_obs_steps)
        self.agent_pos = deque(maxlen=n_obs_steps)

    def reset(self, agentview_pc: np.ndarray, eye_in_hand_pc: np.ndarray, agent: np.ndarray):
        self.agentview_pc.clear()
        self.eye_in_hand_pc.clear()
        self.agent_pos.clear()
        for _ in range(self.T):
            self.agentview_pc.append(agentview_pc.copy())
            self.eye_in_hand_pc.append(eye_in_hand_pc.copy())
            self.agent_pos.append(agent.copy())

    def push(self, agentview_pc: np.ndarray, eye_in_hand_pc: np.ndarray, agent: np.ndarray):
        self.agentview_pc.append(agentview_pc.copy())
        self.eye_in_hand_pc.append(eye_in_hand_pc.copy())
        self.agent_pos.append(agent.copy())

    def get(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (T_o, N, 6), (T_o, N, 6), and (T_o, 21)."""
        return (
            np.stack(self.agentview_pc, axis=0),
            np.stack(self.eye_in_hand_pc, axis=0),
            np.stack(self.agent_pos, axis=0),
        )


# ============================================================
# Temporal Action Ensemble
# ============================================================
class TemporalActionEnsemble:
    """Average overlapping action predictions to reduce noise."""

    def __init__(self, K: int, action_dim: int, exp_weight: float = 0.0):
        self.K = K
        self.action_dim = action_dim
        self.exp_weight = exp_weight
        self._predictions: List[Tuple[np.ndarray, int]] = []

    def reset(self):
        self._predictions.clear()

    def add_prediction(self, actions: np.ndarray, global_step: int):
        self._predictions.append((actions.copy(), global_step))
        # Prune predictions that no longer cover the current step
        self._predictions = [
            (a, s) for (a, s) in self._predictions if s + len(a) > global_step
        ]

    def get_action(self, global_step: int) -> np.ndarray:
        weighted_sum = np.zeros(self.action_dim, dtype=np.float64)
        weight_total = 0.0

        for i, (pred_actions, pred_start) in enumerate(self._predictions):
            idx = global_step - pred_start
            if 0 <= idx < len(pred_actions):
                age = len(self._predictions) - 1 - i
                w = np.exp(-self.exp_weight * age) if self.exp_weight > 0 else 1.0
                weighted_sum += w * pred_actions[idx].astype(np.float64)
                weight_total += w

        if weight_total < 1e-12:
            return np.zeros(self.action_dim, dtype=np.float32)
        return (weighted_sum / weight_total).astype(np.float32)


# ============================================================
# EMA Smoother for actions
# ============================================================
class ActionEMASmoother:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self._prev: Optional[np.ndarray] = None

    def reset(self):
        self._prev = None

    def smooth(self, action: np.ndarray) -> np.ndarray:
        if self._prev is None:
            self._prev = action.copy()
            return action.copy()
        smoothed = action.copy()
        # Only smooth the 6-DOF arm part, leave gripper untouched
        smoothed[:6] = self.alpha * action[:6] + (1.0 - self.alpha) * self._prev[:6]
        self._prev = smoothed.copy()
        return smoothed


# ============================================================
# Gripper hysteresis to prevent chattering
# ============================================================
class GripperHysteresis:
    def __init__(
        self,
        open_threshold: float = 0.3,
        close_threshold: float = -0.3,
        min_hold_steps: int = 5,
    ):
        self.open_threshold = open_threshold
        self.close_threshold = close_threshold
        self.min_hold_steps = min_hold_steps
        self._current_state = -1.0  # start closed
        self._steps_in_state = 0
        self._pending_state = None
        self._pending_count = 0

    def reset(self):
        self._current_state = -1.0
        self._steps_in_state = 0
        self._pending_state = None
        self._pending_count = 0

    def update(self, raw_gripper: float) -> float:
        if raw_gripper > self.open_threshold:
            desired = 1.0
        elif raw_gripper < self.close_threshold:
            desired = -1.0
        else:
            desired = self._current_state

        if desired != self._current_state:
            if self._pending_state == desired:
                self._pending_count += 1
            else:
                self._pending_state = desired
                self._pending_count = 1
            if self._pending_count >= self.min_hold_steps:
                self._current_state = desired
                self._steps_in_state = 0
                self._pending_state = None
                self._pending_count = 0
        else:
            self._pending_state = None
            self._pending_count = 0

        self._steps_in_state += 1
        return self._current_state


# ============================================================
# Student-t Multi-Sample Action Aggregator
# ============================================================
class StudentTActionAggregator:
    def __init__(self, n_samples: int = 5, min_df: float = 1.0, max_df: float = 30.0):
        self.n_samples = n_samples
        self.min_df = min_df
        self.max_df = max_df

    def aggregate(self, samples: np.ndarray) -> np.ndarray:
        """Full Student-t MLE fit per action dimension (slower but more robust)."""
        N, K, D = samples.shape
        if N == 1:
            return samples[0]
        if N == 2:
            return samples.mean(axis=0)

        result = np.zeros((K, D), dtype=np.float32)
        for k in range(K):
            for d in range(D):
                vals = samples[:, k, d].astype(np.float64)
                if np.std(vals) < 1e-8:
                    result[k, d] = np.mean(vals)
                    continue
                try:
                    df, loc, scale = student_t.fit(vals)
                    df = np.clip(df, self.min_df, self.max_df)
                    result[k, d] = loc
                except Exception:
                    result[k, d] = np.median(vals)
        return result.astype(np.float32)

    def aggregate_fast(self, samples: np.ndarray) -> np.ndarray:
        """Fast vectorized IRLS approximation of Student-t MLE."""
        N, K, D = samples.shape
        if N <= 2:
            return samples.mean(axis=0).astype(np.float32)

        mu = np.median(samples, axis=0)  # (K, D)
        nu = 3.0

        for _ in range(3):
            residuals = samples - mu[None]
            sigma2 = np.var(samples, axis=0)
            np.maximum(sigma2, 1e-10, out=sigma2)
            scaled_r2 = residuals * residuals
            scaled_r2 /= sigma2[None]
            weights = (nu + 1.0) / (nu + scaled_r2)
            w_sum = weights.sum(axis=0)
            mu = np.einsum('nkd,nkd->kd', weights, samples) / np.maximum(w_sum, 1e-10)

        return mu.astype(np.float32)


def multi_sample_inference(
    policy,
    obs_dict: dict,
    n_samples: int,
    aggregator: StudentTActionAggregator,
    fast: bool = True,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Run policy N times on the same observation, aggregate with Student-t.
    Tries batched inference first, falls back to sequential.
    
    Returns: (aggregated_actions, inference_time_ms, all_samples)
    """
    t0 = time.time()

    # Try batched inference (1 GPU call instead of N)
    try:
        batched_obs = {}
        for key, val in obs_dict.items():
            batched_obs[key] = val.expand(n_samples, *val.shape[1:]).contiguous()

        with torch.inference_mode():
            out = policy.predict_action(batched_obs)

        act = out["action"]
        if isinstance(act, torch.Tensor):
            act = act.detach().cpu().numpy()
        samples = act  # (N, K, D)

    except Exception:
        # Fallback: sequential inference
        all_samples = []
        with torch.inference_mode():
            for _ in range(n_samples):
                out = policy.predict_action(obs_dict)
                act = out["action"]
                if isinstance(act, torch.Tensor):
                    act = act.detach().cpu().numpy()
                all_samples.append(act[0])
        samples = np.stack(all_samples, axis=0)

    infer_ms = (time.time() - t0) * 1000.0

    if fast:
        robust_actions = aggregator.aggregate_fast(samples)
    else:
        robust_actions = aggregator.aggregate(samples)

    return robust_actions, infer_ms, samples


# ============================================================
# Camera reader (single agentview camera)
# ============================================================
def make_camera_reader(
    camera_ids: List[int] = None,
    enforce_fresh: bool = True,
    max_spin_ms: float = 50,
    redis_host: str = "127.0.0.1",
):
    """
    Set up Redis camera subscribers and return a `get_obs()` callable.

    By default uses camera_ids=[0] (single agentview).
    Extend to [0, 1] if you have a second (eye-in-hand) camera.
    """
    if camera_ids is None:
        camera_ids = [0]

    interfaces = {}
    last_sig = {}

    for cid in camera_ids:
        cref = f"rs_{cid}"
        assert_camera_ref_convention(cref)
        cinfo = get_camera_info(cref)
        iface = CameraRedisSubInterface(
            camera_info=cinfo, use_depth=True, redis_host=redis_host
        )
        iface.start()
        interfaces[cid] = iface
        last_sig[cid] = None
        print(f"[Camera] Started rs_{cid}: {cinfo}")

    def _recv_fresh(cid):
        t0 = time.time()
        while True:
            imgs = interfaces[cid].get_img()
            ts = imgs.get("timestamp") or imgs.get("ts") or imgs.get("seq")
            sig = ts if ts is not None else (
                np.uint64(np.sum(imgs["depth"])),
                np.uint64(np.sum(imgs["color"])),
            )
            if sig != last_sig[cid] or not enforce_fresh:
                last_sig[cid] = sig
                return imgs["color"].copy(), imgs["depth"].copy(), sig
            if (time.time() - t0) * 1000.0 > max_spin_ms:
                return imgs["color"].copy(), imgs["depth"].copy(), sig
            time.sleep(0.001)

    def _resize(color_raw, depth_raw):
        color = cv2.resize(
            color_raw, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
        )
        if color.shape[:2] != (240, 320):
            color = cv2.resize(color, (320, 240))
        depth = cv2.resize(
            depth_raw, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST
        )
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0
        else:
            depth = depth.astype(np.float32)
        return color.astype(np.uint8), depth.astype(np.float32)

    def get_obs():
        results = {}
        for cid in camera_ids:
            c_raw, d_raw, sig = _recv_fresh(cid)
            c, d = _resize(c_raw, d_raw)
            results[cid] = {"rgb": c, "depth": d, "sig": sig}
        return results

    return get_obs


# ============================================================
# Rollout — Receding Horizon with Temporal Ensemble
# ============================================================
def rollout_dp3(
    policy: DP3,
    get_obs,
    robot: FrankaInterface,
    controller_type: str,
    controller_cfg,
    n_obs_steps: int,
    n_action_steps: int,
    max_steps: int,
    num_points: int = 8192,
    use_pc_color: bool = True,
    camera_ids: List[int] = None,
    control_hz: int = 20,
    plot: bool = False,
    first_frame_o3d: bool = True,
    ema_alpha: float = 1.0,
    temporal_ensemble_weight: float = 0.0,
    n_samples: int = 3,
    use_student_t: bool = False,
    student_t_fast: bool = True,
    # Camera intrinsics (after 0.5× resize → 320×240)
    fx: float = 320.0,
    fy: float = 320.0,
    cx: float = 160.0,
    cy: float = 120.0,
) -> Tuple[bool, List[np.ndarray]]:
    """
    Run a single DP3 rollout episode on the real robot.

    Returns (success_flag, list_of_rgb_frames).
    """
    if camera_ids is None:
        camera_ids = [0, 1]  # Default to dual cameras (agentview + eye_in_hand)

    device = next(policy.parameters()).device
    control_period = 1.0 / max(1, control_hz)
    imgs: List[np.ndarray] = []

    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()

    ema_smoother = ActionEMASmoother(alpha=ema_alpha)
    gripper_hyst = GripperHysteresis(
        open_threshold=0.3, close_threshold=-0.3, min_hold_steps=3
    )
    ensemble = None  # lazily initialised once we know K

    st_aggregator = StudentTActionAggregator(n_samples=n_samples) if use_student_t else None

    if use_student_t and n_samples > 1:
        print(f"[STUDENT-T] Enabled: n_samples={n_samples}, fast={student_t_fast}")
    else:
        print(f"[STUDENT-T] Disabled (single-sample inference)")

    # ---- Bootstrap observation ----
    print("[ROLLOUT] Bootstrapping initial observation …")
    obs0 = get_obs()
    agent_pos0 = agent_pos_21_from_robot(robot)

    # Build separate point clouds for agentview and eye_in_hand
    agentview_pc0, eye_in_hand_pc0 = _build_dual_pointclouds(
        obs0, camera_ids, num_points, use_pc_color, fx, fy, cx, cy
    )

    print(f"[BOOT] agentview_pc shape={agentview_pc0.shape}, eye_in_hand_pc shape={eye_in_hand_pc0.shape}, agent_pos shape={agent_pos0.shape}")
    print(f"[BOOT] agent_pos[:7] (joints) = {agent_pos0[:7]}")

    if first_frame_o3d:
        visualize_pointcloud(agentview_pc0, "AgentView (frame 0)")
        visualize_pointcloud(eye_in_hand_pc0, "Eye-in-Hand (frame 0)")

    framestack = FrameStack(n_obs_steps)
    framestack.reset(agentview_pc0, eye_in_hand_pc0, agent_pos0)

    if plot:
        for name in ["AgentView RGB", "AgentView Depth", "Eye-in-Hand RGB", "Eye-in-Hand Depth"]:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    prev_sigs = {cid: obs0[cid]["sig"] for cid in camera_ids}
    np.set_printoptions(precision=4, suppress=True)
    success = False
    global_step = 0

    print(
        f"[ROLLOUT] Starting: n_obs_steps={n_obs_steps}, "
        f"n_action_steps={n_action_steps}, max_steps={max_steps}, "
        f"control_hz={control_hz}, ema_alpha={ema_alpha}, "
        f"temporal_weight={temporal_ensemble_weight}"
    )

    while global_step < max_steps:

        # ==================== 1. Fresh observation ====================
        obs = get_obs()

        # Warn on stale frames (first few steps only)
        for cid in camera_ids:
            if obs[cid]["sig"] == prev_sigs.get(cid) and global_step < 5:
                print(f"[WARN] step#{global_step}: camera {cid} frame repeated")
            prev_sigs[cid] = obs[cid]["sig"]

        agentview_pc, eye_in_hand_pc = _build_dual_pointclouds(
            obs, camera_ids, num_points, use_pc_color, fx, fy, cx, cy
        )
        agent_pos = agent_pos_21_from_robot(robot)

        framestack.push(agentview_pc, eye_in_hand_pc, agent_pos)
        agentview_stack, eye_in_hand_stack, agent_stack = framestack.get()  # (To, N, 6), (To, N, 6), (To, 21)

        obs_dict = {
            "agentview_point_cloud": torch.from_numpy(agentview_stack[None]).to(device, torch.float32),
            "eye_in_hand_point_cloud": torch.from_numpy(eye_in_hand_stack[None]).to(device, torch.float32),
            "agent_pos": torch.from_numpy(agent_stack[None]).to(device, torch.float32),
        }

        if global_step < 3:
            print(
                f"[OBS #{global_step}] agentview_pc={obs_dict['agentview_point_cloud'].shape}, "
                f"eye_in_hand_pc={obs_dict['eye_in_hand_point_cloud'].shape}, "
                f"agent={obs_dict['agent_pos'].shape}"
            )

        # ==================== 2. Policy inference ====================
        if use_student_t and n_samples > 1 and st_aggregator is not None:
            raw_chunk, infer_ms, all_samples = multi_sample_inference(
                policy, obs_dict, n_samples, st_aggregator, fast=student_t_fast
            )
            K, D = raw_chunk.shape

            if global_step < 3:
                sample_std = all_samples.std(axis=0).mean()
                print(f"[STUDENT-T #{global_step}] {n_samples} samples, "
                      f"mean_std={sample_std:.4f}, took {infer_ms:.1f}ms")
        else:
            t_infer = time.time()
            with torch.inference_mode():
                result = policy.predict_action(obs_dict)
            infer_ms = (time.time() - t_infer) * 1000.0

            action_chunk = result["action"]  # (B, n_action_steps, 7)
            if isinstance(action_chunk, torch.Tensor):
                action_chunk = action_chunk.detach().cpu().numpy()

            B, K, D = action_chunk.shape
            assert B == 1 and D == 7, f"Unexpected action shape {action_chunk.shape}"
            raw_chunk = action_chunk[0]  # (K, 7)

        # ==================== 3. Temporal ensemble ====================
        if ensemble is None:
            ensemble = TemporalActionEnsemble(
                K, D, exp_weight=temporal_ensemble_weight
            )
            print(
                f"[ENSEMBLE] Initialised: K={K}, action_dim={D}, "
                f"weight={temporal_ensemble_weight}"
            )

        ensemble.add_prediction(raw_chunk, global_step)

        if global_step < 3:
            print(f"[INFER #{global_step}] took {infer_ms:.1f} ms, got {K} actions")
            print(f"[ACT #{global_step}] raw[0] = {raw_chunk[0]}")

        # ==================== 4. Execute n_exec actions ====================
        n_exec = min(K, n_action_steps)
        for k in range(n_exec):
            if global_step >= max_steps:
                break

            action_t0 = time.time()

            cmd = ensemble.get_action(global_step)
            cmd = ema_smoother.smooth(cmd)
            cmd[-1] = gripper_hyst.update(cmd[-1])

            if global_step < 5:
                print(f"[CMD #{global_step}] k={k}/{n_exec}: {cmd}")

            robot.control(
                controller_type=controller_type,
                action=cmd.astype(np.float32),
                controller_cfg=controller_cfg,
            )
            global_step += 1

            elapsed = time.time() - action_t0
            sleep_time = control_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # ==================== 5. Visualisation ====================
        if plot:
            # Show agentview camera (camera_ids[0])
            cam0 = obs[camera_ids[0]]
            cv2.imshow("AgentView RGB", cam0["rgb"][..., ::-1])
            d0 = cam0["depth"]
            if d0.max() > 0:
                d0_vis = (d0 / d0.max() * 255).astype(np.uint8)
                cv2.imshow(
                    "AgentView Depth",
                    cv2.applyColorMap(d0_vis, cv2.COLORMAP_JET),
                )

            # Show eye-in-hand camera (camera_ids[1]) if available
            if len(camera_ids) > 1 and camera_ids[1] in obs:
                cam1 = obs[camera_ids[1]]
                cv2.imshow("Eye-in-Hand RGB", cam1["rgb"][..., ::-1])
                d1 = cam1["depth"]
                if d1.max() > 0:
                    d1_vis = (d1 / d1.max() * 255).astype(np.uint8)
                    cv2.imshow(
                        "Eye-in-Hand Depth",
                        cv2.applyColorMap(d1_vis, cv2.COLORMAP_JET),
                    )

            key = cv2.waitKey(1)
            if key == ord("p"):
                visualize_pointcloud(agentview_pc, f"AgentView #{global_step}")
                visualize_pointcloud(eye_in_hand_pc, f"Eye-in-Hand #{global_step}")
            elif key == 27:  # ESC
                print("[VIZ] ESC pressed — ending rollout")
                break

        # Record side-by-side frame from both cameras for video
        frame_cam0 = obs[camera_ids[0]]["rgb"].copy()
        if len(camera_ids) > 1 and camera_ids[1] in obs:
            frame_cam1 = obs[camera_ids[1]]["rgb"].copy()
            # Resize eye-in-hand to match agentview height if needed
            if frame_cam0.shape[0] != frame_cam1.shape[0]:
                frame_cam1 = cv2.resize(frame_cam1, (frame_cam1.shape[1], frame_cam0.shape[0]))
            combined_frame = np.concatenate([frame_cam0, frame_cam1], axis=1)
            imgs.append(combined_frame)
        else:
            imgs.append(frame_cam0)

        if global_step % 20 == 0:
            print(
                f"[ROLLOUT] step={global_step}/{max_steps}, "
                f"infer={infer_ms:.0f} ms, n_exec={n_exec}, "
                f"ensemble_preds={len(ensemble._predictions)}"
            )

    print(f"[ROLLOUT] Finished: {global_step} steps executed")
    return success, imgs


# ============================================================
# Helper: build separate agentview and eye_in_hand point clouds
# ============================================================
def _build_dual_pointclouds(
    obs: dict,
    camera_ids: List[int],
    num_points: int,
    use_pc_color: bool,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build separate (num_points, 6) point clouds for agentview and eye_in_hand cameras.
    
    Args:
        obs: Dictionary mapping camera_id to {'rgb', 'depth', 'sig'}
        camera_ids: List of camera IDs [0, 1] for [agentview, eye_in_hand]
        num_points: Target number of points per cloud
        use_pc_color: Whether to include RGB colors (True) or just XYZ (False)
        fx, fy, cx, cy: Camera intrinsics
    
    Returns:
        (agentview_pc, eye_in_hand_pc) each of shape (num_points, 6)
    """
    # Default cameras: 0=agentview, 1=eye_in_hand
    agentview_id = 0
    eye_in_hand_id = 1
    
    # Build agentview point cloud
    # NOTE: normalize_rgb=False because the training zarr stores RGB as [0, 255].
    # The model's normalizer will map [0, 255] -> [-1, 1] at inference time.
    if agentview_id in obs and agentview_id in camera_ids:
        cam = obs[agentview_id]
        agentview_pc = depth_to_pointcloud(
            cam["rgb"],
            cam["depth"],
            num_points=num_points,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            normalize_rgb=False,
        )
    else:
        # If agentview camera not available, return zeros
        agentview_pc = np.zeros((num_points, 6), dtype=np.float32)
        print(f"[WARN] AgentView camera (id={agentview_id}) not found, using zeros")

    # Build eye_in_hand point cloud
    if eye_in_hand_id in obs and eye_in_hand_id in camera_ids:
        cam = obs[eye_in_hand_id]
        eye_in_hand_pc = depth_to_pointcloud(
            cam["rgb"],
            cam["depth"],
            num_points=num_points,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            normalize_rgb=False,
        )
    else:
        # If eye_in_hand camera not available, return zeros
        eye_in_hand_pc = np.zeros((num_points, 6), dtype=np.float32)
        print(f"[WARN] Eye-in-Hand camera (id={eye_in_hand_id}) not found, using zeros")
    
    return agentview_pc, eye_in_hand_pc


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="DP3 rollout on a real Franka robot (deoxys)"
    )

    # ---------- checkpoint ----------
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to a DP3 checkpoint (.ckpt), e.g. epoch=0340-train_loss=0.145.ckpt or latest.ckpt",
    )
    parser.add_argument(
        "--zarr",
        type=str,
        default=None,
        help="(Optional) Path to training zarr for sanity-checking action statistics. "
             "The normalizer is already stored inside the checkpoint.",
    )

    # ---------- execution ----------
    parser.add_argument("--n_obs_steps", type=int, default=None,
                        help="Override n_obs_steps from config (default: use checkpoint cfg)")
    parser.add_argument("--n_action_steps", type=int, default=1,
                        help="Actions to execute per inference (receding horizon). "
                             "1 = most reactive. K = old chunk-execute behaviour.")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--control_hz", type=int, default=20)

    # ---------- point cloud ----------
    parser.add_argument("--num_points", type=int, default=8192,
                        help="Points per cloud (must match training shape_meta).")
    parser.add_argument("--camera_ids", type=int, nargs="+", default=[0, 1],
                        help="Redis camera IDs, e.g. [0, 1] for agentview + eye_in_hand.")

    # Camera intrinsics (after 0.5× resize → 320×240)
    parser.add_argument("--fx", type=float, default=320.0)
    parser.add_argument("--fy", type=float, default=320.0)
    parser.add_argument("--cx", type=float, default=160.0)
    parser.add_argument("--cy", type=float, default=120.0)

    # ---------- smoothing ----------
    parser.add_argument("--ema_alpha", type=float, default=1.0,
                        help="EMA smoothing factor. 1.0 = none, 0.3 = heavy.")
    parser.add_argument("--temporal_weight", type=float, default=0.0,
                        help="Temporal ensemble exponential weight. 0.0 = uniform avg.")
    parser.add_argument("--n_samples", type=int, default=3,
                        help="Forward passes for Student-t. 1=off, 3=fast+robust.")
    parser.add_argument("--use_student_t", action="store_true", default=True)
    parser.add_argument("--no_student_t", action="store_true", default=False)
    parser.add_argument("--student_t_fast", action="store_true", default=True)

    # ---------- display ----------
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no_plot", action="store_true", default=False)
    parser.add_argument("--first_frame_o3d", action="store_true", default=False,
                        help="Show Open3D visualisation of the very first frame.")
    parser.add_argument("--enforce_fresh", action="store_true", default=True)

    # ---------- seeding ----------
    parser.add_argument("--seed", type=int, default=42)

    # ---------- robot ----------
    parser.add_argument(
        "--reset_joint_positions",
        type=float,
        nargs=7,
        default=[
            -0.1473222, -0.47613712, 0.03962905, -1.95871737,
            0.02210383, 1.41984736, 0.78097532,
        ],
    )

    # Parse robot args first (before our args)
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    parser.add_argument("--controller-cfg", type=str, default="osc-pose-controller.yml")
    
    args = parser.parse_args()
    if args.no_plot:
        args.plot = False
    if args.no_student_t:
        args.use_student_t = False
        args.n_samples = 1

    set_global_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MAIN] Device: {device}")

    # ---- Robot setup ----
    robot = FrankaInterface(os.path.join(config_root, args.interface_cfg))
    controller_cfg = YamlConfig(
        os.path.join(config_root, args.controller_cfg)
    ).as_easydict()
    controller_type = args.controller_type
    print(f"[MAIN] Controller: {controller_type}")

    def set_gripper(open_gripper: bool = True):
        d = -1.0 if open_gripper else 1.0
        act = np.array([0, 0, 0, 0, 0, 0, d], dtype=np.float32)
        robot.control(
            controller_type=controller_type,
            action=act,
            controller_cfg=controller_cfg,
        )

    # ---- Cameras ----
    get_obs = make_camera_reader(
        camera_ids=args.camera_ids,
        enforce_fresh=args.enforce_fresh,
    )

    # ---- Load DP3 checkpoint ----
    print(f"[CKPT] Loading: {args.ckpt}")
    payload = torch.load(
        open(args.ckpt, "rb"), pickle_module=dill, map_location="cpu"
    )
    cfg = payload["cfg"]

    # Make config mutable for any overrides
    OmegaConf.set_struct(cfg, False)

    workspace = TrainDP3Workspace(cfg, output_dir=None)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # Use EMA model if available (better inference quality)
    use_ema = getattr(cfg.training, "use_ema", False)
    if use_ema and workspace.ema_model is not None:
        policy = workspace.ema_model
        print("[CKPT] Using EMA model for inference")
    else:
        policy = workspace.model
        print("[CKPT] Using base model for inference")

    policy.to(device)
    policy.eval()

    # Force all sub-modules into eval mode (Dropout / BatchNorm)
    for m in policy.modules():
        if isinstance(
            m,
            (
                torch.nn.Dropout,
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d,
            ),
        ):
            m.eval()

    print(f"[CKPT] Loaded successfully. use_ema={use_ema}")

    # Print key config values
    horizon = getattr(cfg, "horizon", "?")
    cfg_n_obs = getattr(cfg, "n_obs_steps", 2)
    cfg_n_act = getattr(cfg, "n_action_steps", 4)
    use_pc_color = getattr(cfg.policy, "use_pc_color", False)
    print(f"[CFG] horizon={horizon}, n_obs_steps={cfg_n_obs}, "
          f"n_action_steps={cfg_n_act}, use_pc_color={use_pc_color}")

    # Resolve n_obs_steps (prefer checkpoint config, allow CLI override)
    n_obs_steps = args.n_obs_steps if args.n_obs_steps is not None else cfg_n_obs
    n_action_steps = args.n_action_steps

    # ---- Optional: sanity-check normalizer vs zarr stats ----
    if args.zarr is not None:
        import zarr
        print(f"[SANITY] Loading zarr: {args.zarr}")
        z = zarr.open(args.zarr, mode="r")
        
        # Check available keys
        print(f"[SANITY] Available keys in zarr: {list(z['data'].keys())}")
        
        # Check actions
        acts = np.asarray(z["data"]["action"][:], dtype=np.float32)
        print(f"[SANITY] Actions shape: {acts.shape}")
        print(
            f"[SANITY] Action stats — "
            f"pos: [{acts[:, :3].min():.3f}, {acts[:, :3].max():.3f}], "
            f"rot: [{acts[:, 3:6].min():.3f}, {acts[:, 3:6].max():.3f}], "
            f"grip: [{acts[:, 6].min():.3f}, {acts[:, 6].max():.3f}]"
        )
        
        # Check agent_pos (state)
        if "state" in z["data"]:
            states = np.asarray(z["data"]["state"][:], dtype=np.float32)
            print(f"[SANITY] Agent pos (state) shape: {states.shape} (expected: [N, 21])")
        
        # Check point clouds
        if "agentview_point_cloud" in z["data"]:
            agentview_pc = z["data"]["agentview_point_cloud"]
            print(f"[SANITY] AgentView point cloud shape: {agentview_pc.shape} (expected: [N, {args.num_points}, 6])")
        if "eye_in_hand_point_cloud" in z["data"]:
            eye_in_hand_pc = z["data"]["eye_in_hand_point_cloud"]
            print(f"[SANITY] Eye-in-Hand point cloud shape: {eye_in_hand_pc.shape} (expected: [N, {args.num_points}, 6])")
        
        # Quick round-trip test through the normalizer
        sample = torch.from_numpy(acts[:16]).to(device)
        normed = policy.normalizer["action"].normalize(sample)
        unnormed = policy.normalizer["action"].unnormalize(normed)
        rt_err = (unnormed.cpu().numpy() - acts[:16]).mean()
        print(f"[SANITY] Normalizer round-trip error: {abs(rt_err):.3e} (should be ~0)")

    # ---- Reset robot ----
    print("[MAIN] Resetting robot to start configuration …")
    try:
        reset_joints_to(robot, args.reset_joint_positions)
        set_gripper(True)
    except Exception as e:
        print(f"[RESET] Warning: {e}")
    time.sleep(2)

    print(
        f"[CFG] n_obs_steps={n_obs_steps}, n_action_steps={n_action_steps}, "
        f"max_steps={args.max_steps}, control_hz={args.control_hz}, "
        f"ema_alpha={args.ema_alpha}, temporal_weight={args.temporal_weight}, "
        f"n_samples={args.n_samples}, student_t={args.use_student_t}"
    )

    # ---- Trials ----
    trial_results = []
    for trial in range(args.n_trials):
        print(f"\n{'=' * 60}")
        print(f"  Trial {trial + 1}/{args.n_trials}")
        print(f"{'=' * 60}")

        # Re-seed for reproducibility across trials
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.cuda.empty_cache()

        if hasattr(policy, "reset"):
            policy.reset()

        success, frames = rollout_dp3(
            policy=policy,
            get_obs=get_obs,
            robot=robot,
            controller_type=controller_type,
            controller_cfg=controller_cfg,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_steps=args.max_steps,
            num_points=args.num_points,
            use_pc_color=use_pc_color,
            camera_ids=args.camera_ids,
            control_hz=args.control_hz,
            plot=args.plot,
            first_frame_o3d=args.first_frame_o3d and (trial == 0),
            ema_alpha=args.ema_alpha,
            temporal_ensemble_weight=args.temporal_weight,
            n_samples=args.n_samples,
            use_student_t=args.use_student_t,
            student_t_fast=args.student_t_fast,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
        )

        trial_results.append(success)
        print(f"Trial {trial + 1} result: {success}")

        # Save video
        if frames:
            video_path = f"dp3_trial_{trial + 1}.mp4"
            try:
                imageio.mimwrite(video_path, frames, fps=args.control_hz, quality=8)
                print(f"[VIDEO] Saved: {video_path}")
            except Exception as e:
                print(f"[VIDEO] Warning: {e}")

        # Reset between trials
        time.sleep(2)
        try:
            reset_joints_to(robot, args.reset_joint_positions)
            set_gripper(True)
        except Exception as e:
            print(f"[RESET] Warning: {e}")

    print(f"\n[DONE] Trial results: {trial_results}")
    robot.close()


if __name__ == "__main__":
    main()
