import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

@MODELS.register_module()
class PseudoLidarBackbone(BaseModule):
    """Pseudo-LiDAR backbone for processing point cloud data.
    
    This backbone converts depth-based point clouds into pseudo-LiDAR
    representations and processes them through a series of 3D convolutions.
    """
    
    def __init__(self,
                 in_channels=4,
                 base_channels=64,
                 depth=18,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.out_indices = out_indices
        
        # Build the backbone layers
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
    
    def forward(self, x):
        """Forward function."""
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)