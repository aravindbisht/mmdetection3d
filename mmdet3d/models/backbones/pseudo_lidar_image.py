import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

class CrossModalAttention(nn.Module):
    """Attention module for fusing image and point features."""
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        """
        Args:
            x: Pseudo-LiDAR features (B, C, H, W)
            y: Image features (B, C, H, W)
        """
        B, C, H, W = x.size()
        
        # Compute attention
        q = self.query(x).view(B, C, -1)  # (B, C, H*W)
        k = self.key(y).view(B, C, -1)    # (B, C, H*W)
        v = self.value(y).view(B, C, -1)  # (B, C, H*W)
        
        energy = torch.bmm(q.transpose(1, 2), k)  # (B, H*W, H*W)
        attention = F.softmax(energy, dim=-1)
        
        out = torch.bmm(v, attention.transpose(1, 2))  # (B, C, H*W)
        out = out.view(B, C, H, W)
        
        return self.gamma * out + x

@MODELS.register_module()
class PseudoLidarImageBackbone(BaseModule):
    """Enhanced with three fusion techniques and attention."""
    
    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 depth=18,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 projection_type='spherical',
                 fusion_type='deep',
                 use_attention=True,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        self.projection_type = projection_type
        self.fusion_type = fusion_type
        self.use_attention = use_attention
        
        # Projection
        if projection_type == 'spherical':
            self.projection = nn.Sequential(
                nn.Conv2d(1, 3, kernel_size=3, padding=1),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True))
        else:
            self.projection = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        
        # Fusion components
        self.fusion_convs = nn.ModuleList()
        if fusion_type == 'deep':
            for i in range(num_stages):
                ch = base_channels * (2 ** i)
                self.fusion_convs.append(ConvModule(
                    ch * 2, ch, kernel_size=3, padding=1, norm_cfg=norm_cfg))
                if use_attention:
                    self.fusion_convs.append(CrossModalAttention(ch))
        
        # Backbone layers
        self.layers = nn.ModuleList()
        for i in range(num_stages):
            layer_channels = base_channels * (2 ** i)
            
            layer = nn.Sequential(
                ConvModule(
                    in_channels if i == 0 else base_channels * (2 ** (i-1)),
                    layer_channels,
                    kernel_size=3,
                    stride=strides[i],
                    padding=dilations[i],
                    dilation=dilations[i],
                    norm_cfg=norm_cfg,
                    bias=False),
                ConvModule(
                    layer_channels,
                    layer_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    bias=False)
            )
            self.layers.append(layer)
        
        self.out_indices = out_indices
    
    def spherical_projection(self, points):
        """Convert 3D points to spherical coordinates."""
        r = torch.norm(points, dim=1)
        theta = torch.atan2(points[:,1], points[:,0])
        phi = torch.atan2(torch.norm(points[:,:2], dim=1), points[:,2])
        return torch.stack([r, theta, phi], dim=1)
    
    def forward(self, x, img_feats=None):
        """Forward with optional image features for fusion."""
        # Projection
        if self.projection_type == 'spherical':
            x = self.spherical_projection(x)
            depth_map = x[:,0]
        else:
            depth_map = x.mean(dim=-1)
            
        pseudo_image = self.projection(depth_map.unsqueeze(1))
        
        # Early fusion
        if self.fusion_type == 'early' and img_feats is not None:
            pseudo_image = torch.cat([pseudo_image, img_feats[0]], dim=1)
            if self.use_attention:
                pseudo_image = CrossModalAttention(pseudo_image.size(1))(pseudo_image, img_feats[0])
        
        # Backbone processing
        outs = []
        x = pseudo_image
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Deep fusion
            if self.fusion_type == 'deep' and img_feats is not None and i < len(img_feats):
                x = torch.cat([x, img_feats[i]], dim=1)
                x = self.fusion_convs[2*i](x)
                if self.use_attention:
                    x = self.fusion_convs[2*i+1](x, img_feats[i])
                
            if i in self.out_indices:
                outs.append(x)
        
        # Late fusion
        if self.fusion_type == 'late' and img_feats is not None:
            new_outs = []
            for j, (out, feat) in enumerate(zip(outs, img_feats)):
                fused = torch.cat([out, feat], dim=1)
                if self.use_attention:
                    fused = CrossModalAttention(fused.size(1))(fused, feat)
                new_outs.append(fused)
            outs = new_outs
                
        return tuple(outs)
