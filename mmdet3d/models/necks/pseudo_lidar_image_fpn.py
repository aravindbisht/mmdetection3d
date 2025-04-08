import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

@MODELS.register_module()
class PseudoLidarImageFPN(BaseModule):
    """FPN for Pseudo-LiDAR image features."""
    
    def __init__(self,
                 in_channels,
                 out_channels=256,
                 num_outs=4,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        if isinstance(in_channels, int):
            in_channels = [in_channels] * num_outs
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        
        # Lateral convolutions
        self.lateral_convs = nn.ModuleList()
        for i in range(start_level, self.num_ins):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                bias=False)
            self.lateral_convs.append(l_conv)
        
        # Fusion convolutions
        self.fpn_convs = nn.ModuleList()
        for i in range(self.num_outs):
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                bias=False)
            self.fpn_convs.append(fpn_conv)
    
    def forward(self, inputs):
        """Forward function."""
        # Build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Build outputs
        outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(self.num_outs)
        ]
        
        return tuple(outs)
