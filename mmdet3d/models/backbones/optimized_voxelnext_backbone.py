import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from ..layers.optimized_sparse_conv import OptimizedSparseConvBlock
from mmdet3d.registry import MODELS

class OptimizedVoxelNeXtBlock(BaseModule):
    """Optimized VoxelNeXt block with improved sparse convolutions.
    
    This block is an improved version of the VoxelNeXtBlock that uses
    optimized sparse convolutions for better performance.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride of the first convolution.
        with_cp (bool): Whether to use checkpointing to save memory.
        use_sparse_conv (bool): Whether to use sparse convolutions.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 with_cp=False,
                 use_sparse_conv=True):
        super().__init__()
        self.with_cp = with_cp
        self.use_sparse_conv = use_sparse_conv
        self.stride = stride
        
        if use_sparse_conv:
            self.conv1 = OptimizedSparseConvBlock(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                norm_cfg=dict(type='BN3d'),
                act_cfg=dict(type='ReLU', inplace=False))
            self.conv2 = OptimizedSparseConvBlock(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=dict(type='BN3d'),
                act_cfg=dict(type='ReLU', inplace=False))
            
            if stride != 1 or in_channels != out_channels:
                self.downsample = OptimizedSparseConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    norm_cfg=dict(type='BN3d'),
                    act_cfg=None)
            else:
                self.downsample = None
        else:
            # Fallback to standard ConvModule
            self.conv1 = ConvModule(
                in_channels,
                out_channels,
                3,
                stride=stride,
                padding=1,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=dict(type='BN3d'),
                act_cfg=dict(type='ReLU', inplace=False))
            self.conv2 = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=dict(type='BN3d'),
                act_cfg=dict(type='ReLU', inplace=False))
            
            if stride != 1 or in_channels != out_channels:
                self.downsample = ConvModule(
                    in_channels,
                    out_channels,
                    1,
                    stride=stride,
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=dict(type='BN3d'),
                    act_cfg=None)
            else:
                self.downsample = None

    def forward(self, x):
        """Forward function.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W, D).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out, D_out).
        """
        identity = x.clone()  # Create a copy to avoid in-place operations
        
        if self.with_cp and x.requires_grad:
            out = torch.utils.checkpoint.checkpoint(self.conv1, x)
            out = torch.utils.checkpoint.checkpoint(self.conv2, out)
        else:
            out = self.conv1(x)
            out = self.conv2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        elif self.stride != 1:
            # Handle stride mismatch in identity path
            identity = F.interpolate(
                identity,
                size=out.shape[2:],
                mode='trilinear',
                align_corners=False)
        
        # Ensure shapes match before addition
        if identity.shape != out.shape:
            # Resize identity to match output shape
            identity = F.interpolate(
                identity,
                size=out.shape[2:],
                mode='trilinear',
                align_corners=False)
        
        # Add residual connection
        out = out + identity
        
        return out

@MODELS.register_module()
class OptimizedVoxelNeXtBackbone(BaseModule):
    """Optimized VoxelNeXt backbone with improved sparse convolutions.
    
    This backbone is an improved version of the VoxelNeXtBackbone that uses
    optimized sparse convolutions for better performance.
    
    Args:
        in_channels (int): Number of input channels.
        layer_nums (list): Number of layers in each stage.
        layer_strides (list): Stride of the first layer in each stage.
        out_channels (list): Number of output channels in each stage.
        sparse_shape (list): Shape of the sparse tensor.
        with_cp (bool): Whether to use checkpointing to save memory.
        use_sparse_conv (bool): Whether to use sparse convolutions.
    """
    def __init__(self,
                 in_channels=4,
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 out_channels=[128, 256, 512],
                 sparse_shape=[41, 1600, 1408],
                 with_cp=False,
                 use_sparse_conv=True):
        super(OptimizedVoxelNeXtBackbone, self).__init__()
        self.in_channels = in_channels
        self.layer_nums = layer_nums
        self.layer_strides = layer_strides
        self.out_channels = out_channels
        self.sparse_shape = sparse_shape
        self.with_cp = with_cp
        self.use_sparse_conv = use_sparse_conv
        
        # Build backbone layers
        self.blocks = nn.ModuleList()
        for i, (layer_num, layer_stride, out_channel) in enumerate(
                zip(layer_nums, layer_strides, out_channels)):
            layers = []
            for j in range(layer_num):
                if j == 0:
                    layers.append(
                        OptimizedVoxelNeXtBlock(
                            in_channels if i == 0 else out_channels[i - 1],
                            out_channel,
                            stride=layer_stride,
                            with_cp=with_cp,
                            use_sparse_conv=use_sparse_conv))
                else:
                    layers.append(
                        OptimizedVoxelNeXtBlock(
                            out_channel,
                            out_channel,
                            stride=1,
                            with_cp=with_cp,
                            use_sparse_conv=use_sparse_conv))
            self.blocks.append(nn.Sequential(*layers))
    
    def forward(self, x):
        """Forward function.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W, D).
            
        Returns:
            list[torch.Tensor]: Multi-scale feature maps.
        """
        # Initial feature extraction
        x = x.clone()  # Create a copy to avoid in-place operations
        
        # Forward through backbone layers
        outs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            outs.append(x)
        
        return outs 
