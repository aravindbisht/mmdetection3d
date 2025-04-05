import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

@MODELS.register_module()
class PseudoLidarFPN(BaseModule):
    """3D Feature Pyramid Network for Pseudo-LiDAR features.
    
    Key enhancements:
    - Proper 3D feature handling
    - Configurable multi-scale fusion
    - Memory-efficient operations
    """
    
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
        self.start_level = start_level
        self.add_extra_convs = add_extra_convs
        
        # Lateral convolutions
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.num_ins):
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
        """Forward with 3D feature fusion."""
        assert len(inputs) == len(self.in_channels)
        
        # Build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if laterals[i].shape[2:] != laterals[i-1].shape[2:]:
                laterals[i - 1] += F.interpolate(
                    laterals[i], 
                    size=laterals[i-1].shape[2:], 
                    mode='trilinear',  # Changed from 'nearest' for smoother 3D fusion
                    align_corners=False)
        
        # Build outputs
        outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels)
        ]
        
        # Add extra levels if needed
        if self.num_outs > len(outs):
            for i in range(self.num_outs - used_backbone_levels):
                outs.append(F.max_pool3d(
                    outs[-1], kernel_size=3, stride=2, padding=1))
                
        return tuple(outs)
