import torch
import torch.nn as nn
import spconv.pytorch as spconv
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

@MODELS.register_module()
class VoxelNeXt(BaseModule):
    def __init__(self, in_channels=4, base_channels=64, norm_cfg=dict(type='LN', eps=1e-6), init_cfg=None):
        super(VoxelNeXt, self).__init__(init_cfg)

        self.base_channels = base_channels

        # Sparse convolution layers
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )

        # Sparse residual blocks (VoxelNeXt)
        self.conv1 = self._make_layer(base_channels, base_channels * 2, stride=2)
        self.conv2 = self._make_layer(base_channels * 2, base_channels * 4, stride=2)
        self.conv3 = self._make_layer(base_channels * 4, base_channels * 8, stride=2)

    def _make_layer(self, in_channels, out_channels, stride):
        return spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_input(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
