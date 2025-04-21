import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from ..layers.optimized_sparse_conv import OptimizedSparseConvBlock

@MODELS.register_module()
class OptimizedVoxelNeXtNeck(BaseModule):
    """Optimized VoxelNeXt neck with improved performance.
    
    This neck is an improved version of the VoxelNeXtNeck that uses
    optimized sparse convolutions and better memory management.
    
    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (list[int]): Number of output channels per scale.
        upsample_strides (list[float]): Upsample strides for each scale.
        sparse_shape (list[int]): Shape of the sparse tensor.
        use_sparse_conv (bool): Whether to use sparse convolutions.
    """
    def __init__(self,
                 in_channels=[128, 256, 512],
                 out_channels=[256, 256, 256],
                 upsample_strides=[0.5, 1, 2],
                 sparse_shape=[41, 1600, 1408],
                 use_sparse_conv=True):
        super(OptimizedVoxelNeXtNeck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_strides = upsample_strides
        self.sparse_shape = sparse_shape
        self.use_sparse_conv = use_sparse_conv

        # Build deconvolution layers
        self.deblocks = nn.ModuleList()
        for i, (in_channel, out_channel, stride) in enumerate(
                zip(in_channels, out_channels, upsample_strides)):
            if use_sparse_conv:
                deblock = OptimizedSparseConvBlock(
                    in_channel,
                    out_channel,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    norm_cfg=dict(type='BN3d'),
                    act_cfg=dict(type='ReLU', inplace=False))
            else:
                deblock = ConvModule(
                    in_channel,
                    out_channel,
                    3,
                    stride=stride,
                    padding=1,
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=dict(type='BN3d'),
                    act_cfg=dict(type='ReLU', inplace=False))
            self.deblocks.append(deblock)

    def forward(self, x):
        """Forward function.
        
        Args:
            x (list[torch.Tensor]): List of feature maps from backbone.
            
        Returns:
            list[torch.Tensor]: Multi-scale feature maps.
        """
        outs = []
        for i, feat in enumerate(x):
            # Create a new tensor to avoid in-place operations
            feat = feat.clone()
            
            # Apply deconvolution
            out = self.deblocks[i](feat)
            outs.append(out)
        
        return outs 
