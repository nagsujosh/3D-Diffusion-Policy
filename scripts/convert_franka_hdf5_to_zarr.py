#!/usr/bin/env python3
"""
Create Zarr Dataset from HDF5 Robot Demonstration Data with Two Point Clouds

This script extracts data from HDF5 robot demonstrations and creates a zarr dataset
with the same format as the Gaussian splatting version, but using two separate point clouds
(agentview and eye_in_hand) from RGB-D data.
"""

import h5py
import zarr
import numpy as np
import json
import os
from typing import Tuple, List, Dict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm

def voxel_grid_sampling(points, num_samples):
    """
    FAST voxel grid downsampling - fully vectorized, no Python loops.
    
    Deterministic and structure-preserving. Selects first point in each voxel
    (sorted by original index) for consistency.
    
    Args:
        points: [N, 3] array of XYZ coordinates
        num_samples: Target number of samples
    
    Returns:
        indices: Selected point indices
    """
    N = len(points)
    if N <= num_samples:
        return np.arange(N)
    
    # Calculate voxel size to achieve target number of samples
    points_min = points.min(axis=0)
    points_max = points.max(axis=0)
    bbox_size = points_max - points_min + 1e-6  # Avoid division by zero
    
    # Calculate voxel size to approximately achieve target samples
    voxel_volume = bbox_size.prod() / num_samples
    voxel_size = np.cbrt(voxel_volume)
    
    # Ensure minimum voxel size
    if voxel_size < 1e-6:
        voxel_size = bbox_size.max() / np.cbrt(num_samples)
    
    # Compute voxel indices for each point
    voxel_indices = np.floor((points - points_min) / voxel_size).astype(np.int64)
    
    # Create unique voxel keys using large primes (deterministic hash)
    voxel_keys = (voxel_indices[:, 0] * 73856093 + 
                  voxel_indices[:, 1] * 19349663 + 
                  voxel_indices[:, 2] * 83492791)
    
    # VECTORIZED: Get first occurrence of each unique voxel key
    # np.unique with return_index gives the FIRST occurrence (lowest original index)
    _, first_occurrence_indices = np.unique(voxel_keys, return_index=True)
    
    # Sort to maintain deterministic order
    selected_indices = np.sort(first_occurrence_indices)
    
    # If we have enough, subsample; if not enough, add more points
    if len(selected_indices) >= num_samples:
        # Take first num_samples (deterministic)
        selected_indices = selected_indices[:num_samples]
    else:
        # Need more points - add uniformly from remaining
        remaining_mask = np.ones(N, dtype=bool)
        remaining_mask[selected_indices] = False
        remaining_indices = np.where(remaining_mask)[0]
        
        num_needed = num_samples - len(selected_indices)
        if len(remaining_indices) > 0:
            step = max(1, len(remaining_indices) // num_needed)
            additional = remaining_indices[::step][:num_needed]
            selected_indices = np.concatenate([selected_indices, additional])
    
    return selected_indices[:num_samples].astype(np.int32)


class HDF5ToZarrConverter:
    def __init__(self, hdf5_path: str, max_points: int = 8192):
        """
        Initialize the converter.
        
        Args:
            hdf5_path: Path to the HDF5 file containing robot demonstration data
            max_points: Maximum number of points per point cloud
        """
        self.hdf5_path = hdf5_path
        self.max_points = max_points
        
        # Camera parameters (from flow_value_function_utils.py)
        self.img_width = 320
        self.img_height = 240
        
        # Default camera intrinsics
        self.fx = 320.0
        self.fy = 320.0  
        self.cx = 160.0
        self.cy = 120.0
        
        # Depth scale factor (convert uint16 depth to meters)
        self.depth_scale = 0.001
        
    def depth_to_pointcloud(self, rgb_image: np.ndarray, depth_image: np.ndarray) -> np.ndarray:
        """
        Convert RGB-D image to point cloud with exactly max_points.
        
        Args:
            rgb_image: RGB image array of shape (H, W, 3)
            depth_image: Depth image array of shape (H, W)
            
        Returns:
            Point cloud array of shape (max_points, 6) where each row is [x, y, z, r, g, b]
        """
        height, width = depth_image.shape
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Flatten arrays
        u_flat = u.flatten()
        v_flat = v.flatten()
        depth_flat = depth_image.flatten() * self.depth_scale
        
        # Filter out invalid depth values
        valid_mask = depth_flat > 0.01  # Filter out depths less than 1cm
        
        u_valid = u_flat[valid_mask]
        v_valid = v_flat[valid_mask]
        depth_valid = depth_flat[valid_mask]
        
        # Convert to 3D coordinates
        x = (u_valid - self.cx) * depth_valid / self.fx
        y = (v_valid - self.cy) * depth_valid / self.fy
        z = depth_valid
        
        # Get corresponding RGB values (keep original RGB order)
        rgb_flat = rgb_image.reshape(-1, 3)
        rgb_valid = rgb_flat[valid_mask]
        # Note: RGB channels are kept in their original order
        
        # Combine 3D coordinates with RGB values
        valid_pointcloud = np.column_stack([x, y, z, rgb_valid])
        
        # Create output array with exactly max_points
        pointcloud = np.zeros((self.max_points, 6), dtype=np.float32)
        
        if len(valid_pointcloud) > 0:
            if len(valid_pointcloud) >= self.max_points:
                # Use voxel grid sampling for better distribution (fast & deterministic)
                indices = voxel_grid_sampling(valid_pointcloud[:, :3], self.max_points)
                sampled_points = valid_pointcloud[indices]
                # Ensure we get exactly max_points
                pointcloud[:len(sampled_points)] = sampled_points
            else:
                # Use all valid points and pad with zeros to reach max_points
                pointcloud[:len(valid_pointcloud)] = valid_pointcloud
        
        # Final check: ensure exactly max_points are returned
        assert pointcloud.shape[0] == self.max_points, f"Expected {self.max_points} points, got {pointcloud.shape[0]}"
        
        return pointcloud.astype(np.float32)
    
    def extract_agent_pos(self, demo_data: dict, timestep: int) -> np.ndarray:
        """
        Extract comprehensive robot state vector (same format as original).
        
        Args:
            demo_data: Dictionary containing demonstration data
            timestep: Current timestep index
            
        Returns:
            agent_pos: [21] dimensional state vector containing:
                     - joint_states: [7] joint positions  
                     - ee_position: [3] end-effector XYZ
                     - ee_rotation: [9] end-effector rotation matrix (flattened)
                     - gripper_states: [2] gripper positions
        """
        obs = demo_data['obs']
        state_components = []        
        # 1. Joint positions (7 DOF arm)
        if 'joint_states' in obs:
            joint_pos = obs['joint_states'][timestep]  # [7]
            state_components.append(joint_pos)
        else:
            joint_pos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
            state_components.append(joint_pos)
        
        # 2. End-effector pose (position + rotation)
        if 'ee_states' in obs:
            ee_pose_flat = obs['ee_states'][timestep]  # [16] flattened 4x4 transformation matrix
            # Convert from flattened 16-element pose to transformation matrix
            ee_pose_matrix = np.array(ee_pose_flat).reshape(4, 4).T
            
            # Extract position (3 elements)
            ee_position = ee_pose_matrix[:3, 3]  # [3]
            state_components.append(ee_position)
            
            # Extract rotation matrix and flatten (9 elements)
            ee_rotation = ee_pose_matrix[:3, :3].flatten()  # [9]
            state_components.append(ee_rotation)
        else:
            # Default end-effector pose
            ee_position = np.zeros(3)
            ee_rotation = np.eye(3).flatten()
            state_components.extend([ee_position, ee_rotation])
        
        # 3. Gripper state (2 elements for parallel gripper)
        if 'gripper_states' in obs:
            gripper_state = obs['gripper_states'][timestep]
            if len(gripper_state) == 1:
                gripper_pos = [gripper_state[0], gripper_state[0]]  # Symmetric gripper
            else:
                gripper_pos = gripper_state[:2]
            state_components.append(np.array(gripper_pos))
        else:
            gripper_pos = [0.04, 0.04]  # Default gripper opening
            state_components.append(np.array(gripper_pos))
        
        # Combine all components into single state vector
        agent_pos = np.concatenate(state_components).astype(np.float32)
        
        return agent_pos  # Shape: [21]
    
    def list_available_demos(self) -> List[str]:
        """List all available demonstrations in the HDF5 file."""
        demo_names = []
        with h5py.File(self.hdf5_path, 'r') as f:
            data_group = f['data']
            for key in data_group.keys():
                demo_names.append(key)
        return sorted(demo_names)
    
    def process_single_demo(self, demo_name: str, show_progress: bool = False) -> Dict:
        """
        Process a single demonstration to extract point clouds and states.
        
        Args:
            demo_name: Name of the demonstration (e.g., "demo_0")
            show_progress: Whether to show per-timestep progress bar
            
        Returns:
            Dictionary containing processed demo data
        """
        with h5py.File(self.hdf5_path, 'r') as f:
            demo_group = f[f'data/{demo_name}']
            obs_group = demo_group['obs']
            actions = demo_group['actions'][:]
            
            # Get sequence length
            seq_length = len(actions)
            
            # Initialize arrays for this demo
            agentview_pointclouds = np.zeros((seq_length, self.max_points, 6), dtype=np.float32)
            eye_in_hand_pointclouds = np.zeros((seq_length, self.max_points, 6), dtype=np.float32)
            agent_positions = np.zeros((seq_length, 21), dtype=np.float32)
            
            # Prepare demo data dict for agent_pos extraction
            demo_data = {'obs': {key: obs_group[key][:] for key in obs_group.keys()}}
            
            # Process timesteps with optional progress bar
            timestep_iter = range(seq_length)
            if show_progress:
                timestep_iter = tqdm(timestep_iter, desc=f"  {demo_name}", leave=False)
            
            for t in timestep_iter:
                # Extract RGB and depth for both cameras
                agentview_rgb = obs_group['agentview_rgb'][t]
                agentview_depth = obs_group['agentview_rgb_depth'][t]
                eye_in_hand_rgb = obs_group['eye_in_hand_rgb'][t]
                eye_in_hand_depth = obs_group['eye_in_hand_rgb_depth'][t]
                
                # Generate point clouds
                agentview_pc = self.depth_to_pointcloud(agentview_rgb, agentview_depth)
                eye_in_hand_pc = self.depth_to_pointcloud(eye_in_hand_rgb, eye_in_hand_depth)
                
                # Extract agent position
                agent_pos = self.extract_agent_pos(demo_data, t)
                
                # Store data
                agentview_pointclouds[t] = agentview_pc
                eye_in_hand_pointclouds[t] = eye_in_hand_pc
                agent_positions[t] = agent_pos
        
        return {
            'demo_name': demo_name,
            'agentview_pointcloud': agentview_pointclouds,
            'eye_in_hand_pointcloud': eye_in_hand_pointclouds,
            'agent_pos': agent_positions,
            'actions': actions,
            'episode_length': seq_length
        }
    
    def print_data_format(self, demos_data: List[Dict]):
        """Print the data format that will be saved to zarr file."""
        if not demos_data:
            print("No demo data to print format for.")
            return
            
        print("=" * 80)
        print("DATA FORMAT SUMMARY FOR ZARR FILE")
        print("=" * 80)
        
        # Calculate total dimensions
        total_timesteps = sum(demo['episode_length'] for demo in demos_data)
        num_episodes = len(demos_data)
        
        print(f"\nOVERVIEW:")
        print(f"   Total Episodes: {num_episodes}")
        print(f"   Total Timesteps: {total_timesteps:,}")
        print(f"   Max Points per Point Cloud: {self.max_points:,}")
        
        # Sample data from first demo
        sample_demo = demos_data[0]
        agentview_pc = sample_demo['agentview_pointcloud']
        eye_in_hand_pc = sample_demo['eye_in_hand_pointcloud']
        agent_pos = sample_demo['agent_pos']
        actions = sample_demo['actions']
        
        print(f"\nARRAY SHAPES:")
        print(f"   AgentView Point Cloud: {agentview_pc.shape} -> Total: {(total_timesteps, self.max_points, 6)}")
        print(f"   Eye-in-Hand Point Cloud: {eye_in_hand_pc.shape} -> Total: {(total_timesteps, self.max_points, 6)}")
        print(f"   Agent Positions (State): {agent_pos.shape} -> Total: {(total_timesteps, 21)}")
        print(f"   Actions: {actions.shape} -> Total: {(total_timesteps, actions.shape[1])}")
        
        print(f"\nZARR FILE STRUCTURE:")
        print(f"   /data/")
        print(f"   ├── agentview_point_cloud     : shape {(total_timesteps, self.max_points, 6)}, dtype=float32")
        print(f"   ├── eye_in_hand_point_cloud   : shape {(total_timesteps, self.max_points, 6)}, dtype=float32")  
        print(f"   ├── state                     : shape {(total_timesteps, 21)}, dtype=float32")
        print(f"   └── action                    : shape {(total_timesteps, actions.shape[1])}, dtype=float64")
        print(f"   /meta/")
        print(f"   ├── episode_ends              : shape ({num_episodes},), dtype=int")
        print(f"   └── attrs: num_episodes, total_timesteps, etc.")
        
        print(f"\nPOINT CLOUD FORMAT (6 channels per point):")
        print(f"   [x, y, z, r, g, b] where:")
        print(f"   - x, y, z: 3D coordinates in meters (camera frame)")
        print(f"   - r, g, b: RGB color values [0-255]")
        print(f"   - Points are padded with zeros if fewer than {self.max_points:,} valid points")
        
        print(f"\nAGENT STATE FORMAT (21 dimensions):")
        print(f"   [joint_states(7) + ee_position(3) + ee_rotation(9) + gripper_states(2)]")
        print(f"   - joint_states: 7 DOF arm joint positions")
        print(f"   - ee_position: End-effector XYZ position")  
        print(f"   - ee_rotation: End-effector rotation matrix (flattened)")
        print(f"   - gripper_states: Gripper positions (symmetric parallel gripper)")
        
        print(f"\nEPISODE INFORMATION:")
        episode_lengths = [demo['episode_length'] for demo in demos_data]
        print(f"   Episode Lengths: {episode_lengths}")
        print(f"   Min Length: {min(episode_lengths)}")
        print(f"   Max Length: {max(episode_lengths)}")
        print(f"   Avg Length: {sum(episode_lengths) / len(episode_lengths):.1f}")
        
        # Show sample data values
        print(f"\nSAMPLE DATA VALUES (from first timestep):")
        sample_agentview = agentview_pc[0]
        sample_eye_in_hand = eye_in_hand_pc[0]
        sample_agent_pos = agent_pos[0]
        sample_action = actions[0]
        
        # Count non-zero points
        agentview_valid = np.sum(np.any(sample_agentview[:, :3] != 0, axis=1))
        eye_in_hand_valid = np.sum(np.any(sample_eye_in_hand[:, :3] != 0, axis=1))
        
        print(f"   AgentView Valid Points: {agentview_valid:,} / {self.max_points:,}")
        print(f"   Eye-in-Hand Valid Points: {eye_in_hand_valid:,} / {self.max_points:,}")
        print(f"   Agent Position Sample: {sample_agent_pos[:5]} ... (showing first 5/21)")
        print(f"   Action Sample: {sample_action}")
        
        print("=" * 80)
    
    def save_to_zarr(self, demos_data: List[Dict], output_path: str):
        """Save converted demos to Zarr format."""
        print(f"Saving to Zarr format: {output_path}")
        
        # Create zarr file
        zarr_root = zarr.open(output_path, mode='w')
        
        # Concatenate all demos
        all_agentview_pointclouds = []
        all_eye_in_hand_pointclouds = []
        all_agent_pos = []
        all_actions = []
        episode_ends = []
        
        current_length = 0
        for demo_data in demos_data:
            all_agentview_pointclouds.append(demo_data['agentview_pointcloud'])
            all_eye_in_hand_pointclouds.append(demo_data['eye_in_hand_pointcloud'])
            all_agent_pos.append(demo_data['agent_pos'])
            all_actions.append(demo_data['actions'])
            
            current_length += len(demo_data['agentview_pointcloud'])
            episode_ends.append(current_length)
        
        # Concatenate arrays
        agentview_pointclouds = np.concatenate(all_agentview_pointclouds, axis=0)
        eye_in_hand_pointclouds = np.concatenate(all_eye_in_hand_pointclouds, axis=0)
        agent_pos = np.concatenate(all_agent_pos, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        
        print(f"Final dataset summary:")
        print(f"   Total timesteps: {len(agentview_pointclouds):,}")
        print(f"   Episodes: {len(demos_data)}")
        print(f"   Episode lengths: {[d['episode_length'] for d in demos_data]}")
        
        # Create data group
        data_group = zarr_root.create_group('data')
        data_group.create_dataset('agentview_point_cloud', data=agentview_pointclouds, chunks=True, compressor=zarr.Blosc())
        data_group.create_dataset('eye_in_hand_point_cloud', data=eye_in_hand_pointclouds, chunks=True, compressor=zarr.Blosc())
        data_group.create_dataset('state', data=agent_pos, chunks=True, compressor=zarr.Blosc())
        data_group.create_dataset('action', data=actions, chunks=True, compressor=zarr.Blosc())
        
        # Create meta group
        meta_group = zarr_root.create_group('meta')
        meta_group.create_dataset('episode_ends', data=np.array(episode_ends))
        
        # Add metadata
        meta_group.attrs['num_episodes'] = len(demos_data)
        meta_group.attrs['total_timesteps'] = len(agentview_pointclouds)
        meta_group.attrs['agentview_point_cloud_shape'] = list(agentview_pointclouds.shape)
        meta_group.attrs['eye_in_hand_point_cloud_shape'] = list(eye_in_hand_pointclouds.shape)
        meta_group.attrs['state_shape'] = list(agent_pos.shape)
        meta_group.attrs['action_shape'] = list(actions.shape)
        meta_group.attrs['max_points_per_cloud'] = self.max_points
        
        # Add configuration as metadata
        config_dict = {
            'source_hdf5': self.hdf5_path,
            'max_points': self.max_points,
            'img_width': self.img_width,
            'img_height': self.img_height,
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy,
            'depth_scale': self.depth_scale
        }
        meta_group.attrs['config'] = json.dumps(config_dict)
        
        print(f"Saved {len(demos_data)} episodes to {output_path}")
        print(f"   AgentView Point Clouds: {agentview_pointclouds.shape}")
        print(f"   Eye-in-Hand Point Clouds: {eye_in_hand_pointclouds.shape}")
        print(f"   Agent Positions: {agent_pos.shape}")
        print(f"   Actions: {actions.shape}")
        
        return output_path
    
    def convert_to_zarr(self, output_path: str, demo_limit: int = None, num_workers: int = None) -> str:
        """
        Convert HDF5 demonstrations to zarr format with parallel processing.
        
        Args:
            output_path: Path where zarr file will be saved
            demo_limit: Optional limit on number of demos to process
            num_workers: Number of parallel workers (default: CPU count)
            
        Returns:
            Path to the created zarr file
        """
        if num_workers is None:
            num_workers = min(cpu_count(), 8)  # Cap at 8 to avoid memory issues
        
        print(f"Starting HDF5 to Zarr conversion...")
        print(f"   Input: {self.hdf5_path}")
        print(f"   Output: {output_path}")
        print(f"   Workers: {num_workers}")
        
        # Get available demos
        available_demos = self.list_available_demos()
        if demo_limit:
            available_demos = available_demos[:demo_limit]
        
        print(f"   Processing {len(available_demos)} demonstrations")
        
        # Process demos with progress bar
        demos_data = []
        
        # Sequential processing with progress bar (safer for HDF5)
        for demo_name in tqdm(available_demos, desc="Processing demos"):
            demo_data = self.process_single_demo(demo_name, show_progress=False)
            demos_data.append(demo_data)
        
        # Print data format before saving
        self.print_data_format(demos_data)
        
        # Save to zarr format
        zarr_path = self.save_to_zarr(demos_data, output_path)
        
        return zarr_path

def main():
    """Main function to demonstrate zarr conversion."""
    # Configuration
    hdf5_path = "add_your_own_file_path.hdf5"
    output_path = "something.zarr"
    max_points = 8192
    demo_limit = None  # Process ALL demos
    
    try:
        # Initialize converter
        converter = HDF5ToZarrConverter(hdf5_path, max_points=max_points)
        
        # Convert to zarr format (now actually saves the file)
        zarr_path = converter.convert_to_zarr(output_path, demo_limit=demo_limit)
        
        print(f"\nZarr file successfully created!")
        print(f"   File path: {zarr_path}")
        
        # Check file size
        if os.path.exists(zarr_path):
            # For zarr directories, calculate total size
            total_size = 0
            for root, dirs, files in os.walk(zarr_path):
                for file in files:
                    total_size += os.path.getsize(os.path.join(root, file))
            print(f"   Total size: {total_size / (1024**3):.2f} GB")
        
    except FileNotFoundError:
        print(f"ERROR: HDF5 file not found at {hdf5_path}")
    except Exception as e:
        print(f"ERROR: Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
