import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

@MODELS.register_module()
class AttentionFusion(BaseModule):
    """Attention-based fusion module for multi-modal 3D detection.
    
    This module uses attention mechanisms to better integrate image and point cloud features.
    It computes attention weights based on feature similarity and applies them to create
    a more effective fusion of the two modalities.
    
    Args:
        img_channels (int): Number of channels in image features.
        pts_channels (int): Number of channels in point cloud features.
        mid_channels (int): Number of channels in intermediate layers.
        out_channels (int): Number of output channels.
        img_levels (list): List of image feature levels to use.
        align_corners (bool): Whether to align corners when interpolating.
        activate_out (bool): Whether to apply activation to output.
        fuse_out (bool): Whether to fuse output with original features.
    """
    def __init__(self,
                 img_channels,
                 pts_channels,
                 mid_channels=128,
                 out_channels=128,
                 img_levels=[0, 1, 2, 3, 4],
                 align_corners=False,
                 activate_out=True,
                 fuse_out=False):
        super().__init__()
        self.img_channels = img_channels
        self.pts_channels = pts_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.img_levels = img_levels
        self.align_corners = align_corners
        self.activate_out = activate_out
        self.fuse_out = fuse_out
        
        # Image feature projection
        self.img_proj = nn.Sequential(
            nn.Conv2d(img_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False)
        )
        
        # Point cloud feature projection
        self.pts_proj = nn.Sequential(
            nn.Conv3d(pts_channels, mid_channels, 1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=False)
        )
        
        # Attention modules for each image level
        self.attention_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(mid_channels, 1, 1),
                nn.Sigmoid()
            ) for _ in img_levels
        ])
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Conv3d(mid_channels, out_channels, 1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False) if activate_out else nn.Identity()
        )
        
        # Optional fusion with original features
        if fuse_out:
            self.fusion = nn.Sequential(
                nn.Conv3d(out_channels + pts_channels, out_channels, 1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=False)
            )
    
    def forward(self, pts_feats, img_feats, input_metas=None):
        """Forward function.
        
        Args:
            pts_feats (torch.Tensor): Point cloud features of shape (B, C, H, W, D).
            img_feats (list[torch.Tensor]): Image features from different levels.
            input_metas (list[dict], optional): Input meta information.
            
        Returns:
            torch.Tensor: Fused features of shape (B, C_out, H, W, D).
        """
        batch_size, _, height, width, depth = pts_feats.shape
        
        # Project point cloud features
        pts_proj = self.pts_proj(pts_feats)
        
        # Initialize output tensor
        fused_feats = torch.zeros_like(pts_proj)
        
        # Process each image level
        for i, level_idx in enumerate(self.img_levels):
            if level_idx >= len(img_feats):
                continue
                
            img_feat = img_feats[level_idx]
            
            # Project image features
            img_proj = self.img_proj(img_feat)
            
            # Reshape image features to match point cloud features
            # Assuming image features are of shape (B, C, H, W)
            img_proj = img_proj.unsqueeze(-1).expand(-1, -1, -1, -1, depth)
            
            # Compute attention weights
            attention = self.attention_modules[i](pts_proj)
            
            # Apply attention to image features
            attended_img = img_proj * attention
            
            # Add to fused features
            fused_feats = fused_feats + attended_img
        
        # Project to output channels
        out_feats = self.out_proj(fused_feats)
        
        # Optionally fuse with original point cloud features
        if self.fuse_out:
            out_feats = self.fusion(torch.cat([out_feats, pts_feats], dim=1))
        
        return out_feats 
