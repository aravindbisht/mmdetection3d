import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from ..layers.sparse_conv import SparseConvBlock

from mmdet3d.registry import MODELS

@MODELS.register_module()
class VoxelNeXtNeck(BaseModule):
    def __init__(self,
                 in_channels,
                 upsample_strides,
                 out_channels,
                 use_sparse_conv=True):
        super().__init__()
        self.in_channels = in_channels
        self.upsample_strides = upsample_strides
        self.out_channels = out_channels
        self.use_sparse_conv = use_sparse_conv
        
        # Build upsampling layers
        self.deblocks = nn.ModuleList()
        for i, (in_channel, out_channel, stride) in enumerate(
                zip(in_channels, out_channels, upsample_strides)):
            if use_sparse_conv:
                self.deblocks.append(
                    SparseConvBlock(
                        in_channel,
                        out_channel,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        norm_cfg=dict(type='BN3d'),
                        act_cfg=dict(type='ReLU')))
            else:
                self.deblocks.append(
                    ConvModule(
                        in_channel,
                        out_channel,
                        3,
                        stride=stride,
                        padding=1,
                        conv_cfg=dict(type='Conv3d'),
                        norm_cfg=dict(type='BN3d'),
                        act_cfg=dict(type='ReLU')))

    def forward(self, x):
        """Forward function.
        Args:
            x (list[Tensor]): List of 4D tensors of shape (N, C, H, W, D).
        Returns:
            list[Tensor]: Multi-scale feature maps.
        """
        outs = []
        for i, deblock in enumerate(self.deblocks):
            out = deblock(x[i])
            outs.append(out)
        
        return outs 
