import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from ..layers.sparse_conv import SparseConvBlock
from mmdet3d.registry import MODELS

class VoxelNeXtBlock(BaseModule):
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
            self.conv1 = SparseConvBlock(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                norm_cfg=dict(type='BN3d'),
                act_cfg=dict(type='ReLU', inplace=False))
            self.bn1 = nn.BatchNorm3d(out_channels)
            self.conv2 = SparseConvBlock(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=dict(type='BN3d'),
                act_cfg=dict(type='ReLU', inplace=False))
            self.bn2 = nn.BatchNorm3d(out_channels)
            
            if stride != 1 or in_channels != out_channels:
                self.downsample = SparseConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    norm_cfg=dict(type='BN3d'),
                    act_cfg=None)
                self.bn_downsample = nn.BatchNorm3d(out_channels)
            else:
                self.downsample = None
                self.bn_downsample = None
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
        identity = x.clone()  # Create a copy to avoid in-place operations
        
        if self.with_cp and x.requires_grad:
            out = torch.utils.checkpoint.checkpoint(self.conv1, x)
            out = self.bn1(out)
            out = F.relu(out, inplace=False)
            out = torch.utils.checkpoint.checkpoint(self.conv2, out)
            out = self.bn2(out)
            out = F.relu(out, inplace=False)
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = F.relu(out, inplace=False)
            out = self.conv2(out)
            out = self.bn2(out)
            out = F.relu(out, inplace=False)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            if self.bn_downsample is not None:
                identity = self.bn_downsample(identity)
        elif self.stride != 1:
            # Handle stride mismatch in identity path
            identity = F.interpolate(
                identity,
                size=out.shape[2:],
                mode='trilinear',
                align_corners=False)
        
        # Ensure shapes match before addition
        if identity.shape != out.shape:
            identity = F.interpolate(
                identity,
                size=out.shape[2:],
                mode='trilinear',
                align_corners=False)
        
        out = out + identity  # Use addition instead of in-place operation
        return out

@MODELS.register_module()
class VoxelNeXtBackbone(BaseModule):
    def __init__(self,
                 in_channels,
                 layer_nums,
                 layer_strides,
                 out_channels,
                 sparse_shape,
                 with_cp=False,
                 use_sparse_conv=True):
        super().__init__()
        self.sparse_shape = sparse_shape
        self.with_cp = with_cp
        self.use_sparse_conv = use_sparse_conv
        
        # Initial conv layer
        if use_sparse_conv:
            self.conv1 = SparseConvBlock(
                in_channels,
                out_channels[0],
                kernel_size=3,
                stride=layer_strides[0],
                padding=1,
                norm_cfg=dict(type='BN3d'),
                act_cfg=dict(type='ReLU', inplace=False))
            self.bn1 = nn.BatchNorm3d(out_channels[0])
        else:
            self.conv1 = ConvModule(
                in_channels,
                out_channels[0],
                3,
                stride=layer_strides[0],
                padding=1,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=dict(type='BN3d'),
                act_cfg=dict(type='ReLU', inplace=False))
        
        # Build backbone layers
        self.layers = nn.ModuleList()
        for i in range(len(layer_nums)):
            layer = nn.Sequential()
            for j in range(layer_nums[i]):
                layer.add_module(
                    f'block_{j}',
                    VoxelNeXtBlock(
                        out_channels[i],
                        out_channels[i],
                        stride=layer_strides[i] if j == 0 else 1,
                        with_cp=with_cp,
                        use_sparse_conv=use_sparse_conv))
            self.layers.append(layer)
            
        # Additional conv layers for feature refinement
        if use_sparse_conv:
            self.conv2 = SparseConvBlock(
                out_channels[-1],
                out_channels[-1],
                kernel_size=3,
                padding=1,
                norm_cfg=dict(type='BN3d'),
                act_cfg=dict(type='ReLU', inplace=False))
            self.bn2 = nn.BatchNorm3d(out_channels[-1])
        else:
            self.conv2 = ConvModule(
                out_channels[-1],
                out_channels[-1],
                3,
                padding=1,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=dict(type='BN3d'),
                act_cfg=dict(type='ReLU', inplace=False))

    def forward(self, x):
        # Initial feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=False)
        
        # Backbone feature extraction
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x.clone())  # Create a copy to avoid in-place operations
        
        # Feature refinement
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=False)
        features.append(x.clone())  # Create a copy to avoid in-place operations
        
        return features 
