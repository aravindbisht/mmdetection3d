import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

@MODELS.register_module()
class LightweightAttentionFusion(BaseModule):
    """Lightweight attention fusion module for image and point cloud features.
    
    This module efficiently fuses image and point cloud features using a lightweight
    attention mechanism. It reduces memory usage and computation while maintaining
    feature alignment between modalities.
    
    Args:
        img_channels (int): Number of input image channels.
        pts_channels (int): Number of input point cloud channels.
        mid_channels (int): Number of middle channels for feature processing.
        out_channels (int): Number of output channels.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout ratio.
        use_sparse_attention (bool): Whether to use sparse attention.
    """
    
    def __init__(self,
                 img_channels,
                 pts_channels,
                 mid_channels,
                 out_channels,
                 num_heads=4,
                 dropout=0.1,
                 use_sparse_attention=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        self.img_channels = img_channels
        self.pts_channels = pts_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_sparse_attention = use_sparse_attention
        
        # Lightweight feature projection
        self.img_proj = ConvModule(
            img_channels,
            mid_channels,
            1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            act_cfg=dict(type='ReLU', inplace=True))
            
        self.pts_proj = ConvModule(
            pts_channels,
            mid_channels,
            1,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d'),
            act_cfg=dict(type='ReLU', inplace=True))
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            mid_channels,
            num_heads,
            dropout,
            use_sparse_attention)
        
        # Output projection
        self.output_proj = ConvModule(
            mid_channels,
            out_channels,
            1,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d'),
            act_cfg=dict(type='ReLU', inplace=True))
            
        # Feature alignment
        self.align_conv = ConvModule(
            out_channels,
            out_channels,
            3,
            padding=1,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d'),
            act_cfg=dict(type='ReLU', inplace=True))
    
    def forward(self, img_feats, pts_feats, img_metas=None):
        """Forward function.
        
        Args:
            img_feats (list[Tensor]): List of image features from FPN.
                Each tensor has shape (B, C, H, W).
            pts_feats (Tensor): Point cloud features.
                Shape (B, C, H, W, D).
            img_metas (list[dict], optional): Image meta info. Defaults to None.
            
        Returns:
            Tensor: Fused features with shape (B, C, H, W, D).
        """
        batch_size = pts_feats.size(0)
        
        # Project features to same dimension
        img_feats = [self.img_proj(feat) for feat in img_feats]
        pts_feats = self.pts_proj(pts_feats)
        
        # Reshape image features for attention
        img_feats = [feat.view(batch_size, self.mid_channels, -1).permute(0, 2, 1) 
                    for feat in img_feats]
        
        # Reshape point cloud features
        pts_shape = pts_feats.shape
        pts_feats = pts_feats.view(batch_size, self.mid_channels, -1).permute(0, 2, 1)
        
        # Apply attention
        if self.use_sparse_attention:
            # Sparse attention for efficiency
            fused_feats = self.attention(
                pts_feats,
                img_feats[0],  # Use highest resolution feature
                pts_feats)
        else:
            # Dense attention
            fused_feats = self.attention(
                pts_feats,
                torch.cat(img_feats, dim=1),  # Concatenate all levels
                pts_feats)
        
        # Reshape back to 3D
        fused_feats = fused_feats.permute(0, 2, 1).view(
            batch_size, self.mid_channels, *pts_shape[2:])
        
        # Project to output dimension
        fused_feats = self.output_proj(fused_feats)
        
        # Align features
        fused_feats = self.align_conv(fused_feats)
        
        return fused_feats

class MultiHeadAttention(BaseModule):
    """Lightweight multi-head attention module.
    
    Args:
        channels (int): Number of input channels.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout ratio.
        use_sparse_attention (bool): Whether to use sparse attention.
    """
    
    def __init__(self,
                 channels,
                 num_heads,
                 dropout=0.1,
                 use_sparse_attention=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        self.channels = channels
        self.num_heads = num_heads
        self.use_sparse_attention = use_sparse_attention
        
        # Ensure channels is divisible by num_heads
        assert channels % num_heads == 0
        self.head_dim = channels // num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Sparse attention parameters
        if use_sparse_attention:
            self.sparse_ratio = 0.5  # Keep 50% of attention weights
    
    def forward(self, q, k, v):
        """Forward function.
        
        Args:
            q (Tensor): Query tensor with shape (B, N, C).
            k (Tensor): Key tensor with shape (B, M, C).
            v (Tensor): Value tensor with shape (B, M, C).
            
        Returns:
            Tensor: Output tensor with shape (B, N, C).
        """
        batch_size = q.size(0)
        
        # Linear projections
        q = self.q_proj(q).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_proj(k).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(v).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # (B, H, N, D)
        k = k.transpose(1, 2)  # (B, H, M, D)
        v = v.transpose(1, 2)  # (B, H, M, D)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if self.use_sparse_attention:
            # Apply sparse attention
            scores = self._sparse_attention(scores)
        
        # Apply softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Compute output
        out = torch.matmul(attn, v)  # (B, H, N, D)
        out = out.transpose(1, 2).contiguous()  # (B, N, H, D)
        out = out.view(batch_size, -1, self.channels)  # (B, N, C)
        
        # Final projection
        out = self.out_proj(out)
        
        return out
    
    def _sparse_attention(self, scores):
        """Apply sparse attention to reduce computation.
        
        Args:
            scores (Tensor): Attention scores with shape (B, H, N, M).
            
        Returns:
            Tensor: Sparse attention scores.
        """
        # Keep top-k attention weights
        k = int(scores.size(-1) * self.sparse_ratio)
        topk_values, _ = torch.topk(scores, k, dim=-1)
        min_values = topk_values[..., -1].unsqueeze(-1)
        
        # Create sparse mask
        sparse_mask = scores >= min_values
        scores = scores.masked_fill(~sparse_mask, float('-inf'))
        
        return scores 
