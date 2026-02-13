# Training Guide for 3D Diffusion Policy

Complete guide for training DP3 models with single or dual-camera setups.

## New Features in This Fork

- **24GB VRAM optimization**: Batch size 48 for dual-camera (16384 points)
- **Detailed loss tracking**: Component-wise action losses (position, rotation, gripper)
- **Auto-resume**: Set `resume: True` to continue from latest.ckpt
- **Verification script**: Validates setup before training
- **Simplified checkpoints**: Removed threading issues, more reliable saving

---

## Prerequisites

1. Install dependencies: See [INSTALL.md](INSTALL.md)
2. Activate environment: `conda activate dp3`
3. Prepare demonstration data (see Data Preparation section)

---

## Environment Setup

Set environment variables before training:

```bash
export CC=gcc-10 CXX=g++-10
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/carl_lab/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/cuda/lib64
export MUJOCO_GL=egl
export QT_GRAPHICSSYSTEM=native
```

Initialize Weights & Biases:
```bash
wandb login
```

---

## Data Preparation

### Dual-Camera Setup (Franka Robot)

**Zarr Structure:**
```
stacking_cups_dual_view.zarr/
├── data/
│   ├── agentview_point_cloud    # (N, 8192, 6) - xyz+rgb
│   ├── eye_in_hand_point_cloud  # (N, 8192, 6) - xyz+rgb
│   ├── state                     # (N, 21) - robot state
│   └── action                    # (N, 7) - robot actions
└── meta/episode_ends
```

**Convert HDF5 to Zarr:**
```bash
python scripts/convert_franka_hdf5_to_zarr.py
```

**State Format (21-D):**
- Joint positions (7) + End-effector position (3) + End-effector rotation (9) + Gripper (2)

**Action Format (7-D):**
- Translation (3) + Rotation (3) + Gripper (1)

---

## Training

### Standard Tasks (Adroit, DexArt, MetaWorld)

```bash
bash scripts/train_policy.sh <algorithm> <task> <tag> <seed> <gpu>
```

**Examples:**
```bash
bash scripts/train_policy.sh dp3 adroit_hammer exp1 0 0
bash scripts/train_policy.sh simple_dp3 dexart_laptop exp1 0 0
bash scripts/train_policy.sh dp3 metaworld_basketball exp1 0 0
```

**Parameters:**
- `dp3`: Full version (better performance)
- `simple_dp3`: Fast version (~25 FPS inference)

### Dual-Camera Tasks

**Verify setup first:**
```bash
python scripts/verify_dual_camera_setup.py
```

**Start training:**
```bash
cd 3D-Diffusion-Policy
python train.py
```

**Resume training (if interrupted):**
```bash
# Training automatically resumes from latest.ckpt if resume: True in config
python train.py
```

**Or with custom parameters:**
```bash
bash scripts/train_stacking_cups.sh [gpu_id] [seed]
```

---

## Configuration

### Key Settings (train_stacking_cups.yaml)

```yaml
# Architecture
encoder_output_dim: 256      # For dual-camera (16384 points)
use_pc_color: true           # Use RGB (6 channels)
pointnet_type: "pointnet"

# Point Cloud Encoder
pointcloud_encoder_cfg:
  in_channels: 6             # xyz + rgb
  out_channels: 256
  use_layernorm: true
  final_norm: layernorm

# Training
batch_size: 48               # Dual-camera optimized
num_workers: 4
lr: 1.0e-4
num_epochs: 3000

# Policy
horizon: 4                   # Predict 4 actions
n_obs_steps: 2              # Use 2 frames
n_action_steps: 4           # Execute 4 actions per inference
```

### Memory-Optimized Settings

**24GB GPU:**
```yaml
batch_size: 32
```

**16GB GPU:**
```yaml
batch_size: 16
```

**12GB GPU:**
```yaml
batch_size: 8
```

---

## Architecture Details

### Dual-Camera Pipeline

```
AgentView (8192, 6) + Eye-in-Hand (8192, 6)
    ↓
DP3Encoder concatenates → (16384, 6)
    ↓
PointNet MLP: 6 → 64 → 128 → 256 → 512
    ↓
Max pooling: (16384, 512) → (512)
    ↓
Final projection: 512 → 256
    ↓
Diffusion Model
```

### Critical Fix Applied

**Issue:** DP3Encoder passed dictionary to PointNet (expects tensor)

**Fixed in:** `diffusion_policy_3d/model/vision/pointnet_extractor.py`

```python
# Before (incorrect)
encoder_obs = {'point_cloud': combined_pc}
features = self.pointcloud_encoder(encoder_obs)

# After (correct)
features = self.pointcloud_encoder(combined_pc)
```

---

## Monitoring

### Detailed Loss Tracking

Training now tracks component-wise action losses for detailed analysis:

**Available Metrics:**
- `action_mse`: Overall action MSE
- `action_pos_mse`: Position error (x, y, z)
- `action_rot_mse`: Rotation error (roll, pitch, yaw)
- `action_gripper_mse`: Gripper control error

**Console Output Every 50 Steps:**
```bash
[E  0 B  100] Loss=0.06966 | action_mse=0.12375 action_pos_mse=0.06710 action_rot_mse=0.00078 action_gripper_mse=0.66263
```

**In Weights & Biases:**
All metrics are automatically logged and can be plotted separately to identify which action components need improvement.

**Analyze Losses After Training:**
```bash
python scripts/analyze_action_losses.py --wandb_run <run_id>
# Or from CSV: --log_file logs.csv
```

This generates plots showing:
- Position, rotation, and gripper losses over time
- Normalized comparison for relative performance
- Statistics summary (initial, final, reduction %)

### Training Outputs

**Training outputs:**
- Logs: WandB dashboard
- Checkpoints: `data/outputs/<timestamp>/checkpoints/`
- Best models: Top-5 by `train_loss`
- Latest checkpoint: `latest.ckpt` (for resume)

**Expected convergence:**
- ~3000 epochs
- Loss < 0.2
- Training time: 5-10 hours (single GPU)

**Resume after interruption:**
```bash
# Just run train.py again - automatically resumes from latest.ckpt
cd 3D-Diffusion-Policy && python train.py
```

**Check training progress:**
```bash
# GPU memory usage
watch -n 1 nvidia-smi

# Training logs
tail -f data/outputs/<timestamp>/logs.json.txt
```

---

## Deployment

### Rollout on Robot

**Basic:**
```bash
python scripts/rollout_dp3_franka.py \
    --ckpt path/to/checkpoint.ckpt \
    --camera_ids 0 1 \
    --num_points 8192 \
    --n_action_steps 1 \
    --max_steps 500
```

**With smoothing:**
```bash
python scripts/rollout_dp3_franka.py \
    --ckpt path/to/checkpoint.ckpt \
    --camera_ids 0 1 \
    --ema_alpha 0.7 \
    --temporal_weight 0.1 \
    --n_samples 3 \
    --use_student_t
```

**Key parameters:**
- `--camera_ids 0 1`: Dual cameras (0=agentview, 1=eye_in_hand)
- `--num_points 8192`: Must match training
- `--n_action_steps 1`: Most reactive (recommended)
- `--control_hz 20`: Control frequency

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM during training | Reduce `batch_size: 16` or `8` |
| Loss is NaN | Reduce `lr: 5.0e-5` |
| Constant actions | Train longer (>1000 epochs) |
| Stale camera frames | Check Redis server, verify camera IDs |
| Jittery actions | Add `--ema_alpha 0.7 --temporal_weight 0.1` |

---

## Performance Tips

**Performance speed:**
- Use `num_workers: 4` and `pin_memory: True`
- Disable `persistent_workers` to save memory

**Accuracy improvements:**
- Increase `n_obs_steps: 3` for temporal reasoning
- Increase `horizon: 8` for long-term planning
- Train for >3000 epochs
- Monitor component losses to identify weak points:
  - High gripper loss? Check data normalization
  - High rotation loss? May need more training or different representation

---

## Files Modified in This Fork

1. `diffusion_policy_3d/policy/dp3.py` - Added detailed component-wise loss tracking
2. `diffusion_policy_3d/policy/simple_dp3.py` - Added detailed component-wise loss tracking
3. `diffusion_policy_3d/model/vision/pointnet_extractor.py` - Fixed forward pass (tensor not dict)
4. `diffusion_policy_3d/config/train_stacking_cups.yaml` - Optimized hyperparameters
5. `train.py` - Fixed checkpoint saving (removed threading issues)
6. `scripts/verify_dual_camera_setup.py` - New verification tool
7. `scripts/analyze_action_losses.py` - New loss analysis tool

**All other components verified correct:**
- Dataset loading (FrankaDataset)
- Policy architecture (DP3)
- Rollout code (rollout_dp3_franka.py)

---

## Quick Start Checklist

- [ ] Environment activated and configured
- [ ] Data in correct zarr format
- [ ] Verification script passes: `python scripts/verify_dual_camera_setup.py`
- [ ] WandB logged in
- [ ] Start training: `cd 3D-Diffusion-Policy && python train.py`
- [ ] Monitor on WandB dashboard
- [ ] Wait for convergence (~3000 epochs)
- [ ] Test on robot with rollout script

---

## References

- Paper: [3D Diffusion Policy](https://arxiv.org/abs/2403.03954)
- GitHub: [YanjieZe/3D-Diffusion-Policy](https://github.com/YanjieZe/3D-Diffusion-Policy)