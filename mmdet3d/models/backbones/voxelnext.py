import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType
from mmcv.ops import SparseConvTensor, SubMConv3d
from mmdet3d.models.layers import SparseBasicBlock, make_sparse_convmodule

from mmcv.cnn import build_norm_layer

@MODELS.register_module()
class VoxelNeXt(BaseModule):
    """VoxelNeXt backbone using mmdet3d's sparse ops."""
    
    def __init__(self,
                 in_channels: int = 128,
                 base_channels: int = 64,
                 out_indices: tuple = (0, 1, 2),
                 norm_cfg: ConfigType = dict(type='BN', eps=1e-3, momentum=0.01),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg=init_cfg)
        self.out_indices = out_indices
        
        # Input projection
        self.input_conv = SubMConv3d(
            in_channels,
            base_channels,
            kernel_size=3,
            padding=1,
            bias=False)
        self.input_norm = build_norm_layer(norm_cfg, base_channels)[1]
        self.input_act = nn.ReLU(inplace=True)
        
        # Feature pyramid stages using mmdet3d's sparse blocks
        self.stages = nn.ModuleList()
        for i in range(3):
            channels = base_channels * (2 ** i)
            stage = SparseBasicBlock(
                channels,
                channels * 2,
                norm_cfg=norm_cfg,
                stride=2,
                downsample=None)
            self.stages.append(stage)
    
    def forward(self, x):
        """Forward pass."""
        if isinstance(x, SparseConvTensor):
            if x.features.numel() == 0:
                return x
        elif x.numel() == 0:
            return x
        
        x = self.input_act(self.input_norm(self.input_conv(x)))
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)