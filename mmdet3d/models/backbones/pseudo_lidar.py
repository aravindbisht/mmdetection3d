import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType

@MODELS.register_module()
class PseudoLidarBackbone(BaseModule):
    """Pseudo-LiDAR backbone for processing 2D images with distance channels."""
    
    def __init__(self,
                 in_channels: int = 4,
                 base_channels: int = 64,
                 out_indices: tuple = (0, 1, 2, 3),
                 depth: int = 18,
                 strides: tuple = (1, 2, 2, 2),
                 dilations: tuple = (1, 1, 1, 1),
                 norm_cfg: ConfigType = dict(type='BN', requires_grad=True),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_indices = out_indices
        self.depth = depth
        self.strides = strides
        self.dilations = dilations
        
        # Initialize base layers
        self.conv1 = nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Initialize residual layers
        self.layers = nn.ModuleList()
        in_planes = base_channels
        planes = [base_channels * (2**i) for i in range(4)]
        
        for i, (stride, dilation) in enumerate(zip(strides, dilations)):
            layer = self._make_layer(
                planes[i], 
                stride=stride,
                dilation=dilation,
                norm_cfg=norm_cfg)
            self.layers.append(layer)
    
    def _make_layer(self, planes: int, 
                   stride: int = 1,
                   dilation: int = 1,
                   norm_cfg: ConfigType = None) -> nn.Module:
        """Build a residual layer block."""
        downsample = None
        if stride != 1 or self.base_channels != planes:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.base_channels,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes))
        
        layers = []
        layers.append(
            nn.Conv2d(
                self.base_channels,
                planes,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                bias=False))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward function."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        
        return tuple(outs)