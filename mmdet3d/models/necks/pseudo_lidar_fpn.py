import torch.nn as nn
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType

@MODELS.register_module()
class PseudoLidarFPN(BaseModule):
    """FPN for Pseudo-LiDAR feature processing.
    
    Args:
        in_channels (list): Number of input channels per scale
        out_channels (int): Number of output channels
        num_outs (int): Number of output feature maps
        start_level (int): Index of the start input backbone level
        end_level (int): Index of the end input backbone level
        add_extra_convs (bool): Whether to add extra conv layers
        norm_cfg (dict): Config for normalization layers
    """
    
    def __init__(self,
                 in_channels: list,
                 out_channels: int,
                 num_outs: int,
                 start_level: int = 0,
                 end_level: int = -1,
                 add_extra_convs: bool = False,
                 norm_cfg: ConfigType = dict(type='BN', requires_grad=True),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg=init_cfg)
        
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        
        if end_level == -1:
            self.end_level = len(in_channels) - 1
        
        # Build lateral and output conv layers
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for i in range(self.start_level, self.end_level + 1):
            l_conv = nn.Conv2d(
                in_channels[i],
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False)
            fpn_conv = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False)
            
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
    
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        
        # Build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += nn.functional.interpolate(
                laterals[i], scale_factor=2, mode='nearest')
        
        # Build outputs
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        
        # Add extra levels if needed
        if self.num_outs > len(outs):
            for i in range(self.num_outs - used_backbone_levels):
                outs.append(nn.functional.max_pool2d(
                    outs[-1], kernel_size=1, stride=2, padding=0))
        
        return tuple(outs)
