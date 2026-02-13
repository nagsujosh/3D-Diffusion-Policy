import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules




class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
       
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()

    


class DP3Encoder(nn.Module):
    def __init__(self, observation_space, img_crop_shape=None, out_channel=256, 
             pointcloud_encoder_cfg=None, use_pc_color=False, pointnet_type="pointnet"):
        super().__init__()
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        
        # Detect dual pointcloud setup
        has_agentview = "agentview_point_cloud" in observation_space
        has_eye_in_hand = "eye_in_hand_point_cloud" in observation_space
        has_single = "point_cloud" in observation_space
        
        if has_agentview and has_eye_in_hand:
            # Dual camera setup
            self.use_dual_pointclouds = True
            self.point_cloud_key = None  # Not used in dual mode
            
            agentview_shape = observation_space["agentview_point_cloud"]
            eye_in_hand_shape = observation_space["eye_in_hand_point_cloud"]
            
            # Verify both have same feature dimension (should be 6: xyz+rgb or 3: xyz)
            assert agentview_shape[-1] == eye_in_hand_shape[-1], \
                f"Point cloud feature dims must match: {agentview_shape} vs {eye_in_hand_shape}"
            
            # Combined point cloud will have N1+N2 points
            num_points_agentview = agentview_shape[0]
            num_points_eye = eye_in_hand_shape[0]
            num_points_total = num_points_agentview + num_points_eye
            feature_dim = agentview_shape[-1]
            
            self.point_cloud_shape = [num_points_total, feature_dim]
            
            print(f"[DP3Encoder] Dual-camera mode: agentview({num_points_agentview}) + "
                  f"eye_in_hand({num_points_eye}) = {num_points_total} points × {feature_dim} features")
            
        elif has_single:
            # Single camera setup (backward compatibility)
            self.use_dual_pointclouds = False
            self.point_cloud_key = "point_cloud"
            self.point_cloud_shape = observation_space[self.point_cloud_key]
            
            print(f"[DP3Encoder] Single-camera mode: {self.point_cloud_shape[0]} points × "
                  f"{self.point_cloud_shape[-1]} features")
            
        else:
            raise KeyError(
                f"Invalid observation space configuration. Expected either:\n"
                f"  - Dual cameras: 'agentview_point_cloud' + 'eye_in_hand_point_cloud'\n"
                f"  - Single camera: 'point_cloud'\n"
                f"Got keys: {list(observation_space.keys())}"
            )
    
        # Build point cloud encoder with the combined/single shape
        self.pointcloud_encoder = self._build_pointcloud_encoder(
            point_cloud_shape=self.point_cloud_shape,
            out_channel=out_channel,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )
        
        # Set output channel attribute (required by dp3.py)
        self.out_channel = out_channel
        self.n_output_channels = out_channel

    def _build_pointcloud_encoder(self, point_cloud_shape, out_channel, pointcloud_encoder_cfg, use_pc_color, pointnet_type):
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                encoder = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                encoder = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")
        
        return encoder

    def forward(self, obs_dict):
        """
        Process observation dictionary containing point cloud(s) and optional agent position.
        
        Args:
            obs_dict: Dictionary containing:
                Dual-camera mode:
                    - 'agentview_point_cloud': (B, N1, C) or (B, T, N1, C)
                    - 'eye_in_hand_point_cloud': (B, N2, C) or (B, T, N2, C)
                Single-camera mode:
                    - 'point_cloud': (B, N, C) or (B, T, N, C)
                Both modes:
                    - 'agent_pos': (B, D) or (B, T, D) [optional]
        
        Returns:
            features: (B, out_channel) or (B, T, out_channel)
        """
        if self.use_dual_pointclouds:
            # Extract both pointclouds
            agentview_pc = obs_dict['agentview_point_cloud']
            eye_in_hand_pc = obs_dict['eye_in_hand_point_cloud']
            
            # Handle both (B, N, C) and (B, T, N, C) cases
            has_time_dim = agentview_pc.ndim == 4
            
            if has_time_dim:
                B, T, N1, C = agentview_pc.shape
                _, _, N2, _ = eye_in_hand_pc.shape
                
                # Flatten temporal dimension: (B, T, N, C) -> (B*T, N, C)
                agentview_flat = agentview_pc.reshape(B * T, N1, C)
                eye_in_hand_flat = eye_in_hand_pc.reshape(B * T, N2, C)
                
                # Concatenate along point dimension
                combined_pc = torch.cat([agentview_flat, eye_in_hand_flat], dim=1)  # (B*T, N1+N2, C)
                
                # Encode - pass tensor directly to PointNet
                features = self.pointcloud_encoder(combined_pc)  # (B*T, out_channel)
                
                # Reshape back to temporal: (B*T, out_channel) -> (B, T, out_channel)
                features = features.reshape(B, T, -1)
                
            else:  # No time dimension: (B, N, C)
                B, N1, C = agentview_pc.shape
                _, N2, _ = eye_in_hand_pc.shape
                
                # Concatenate along point dimension
                combined_pc = torch.cat([agentview_pc, eye_in_hand_pc], dim=1)  # (B, N1+N2, C)
                
                # Encode - pass tensor directly to PointNet
                features = self.pointcloud_encoder(combined_pc)  # (B, out_channel)
    
        else:
            # Single pointcloud mode - pass through directly
            features = self.pointcloud_encoder(obs_dict)
        
        return features

    def output_shape(self):
        return self.n_output_channels