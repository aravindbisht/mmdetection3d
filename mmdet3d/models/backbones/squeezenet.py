from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmcv.cnn import build_conv_layer, build_norm_layer
import torch.nn as nn
from typing import Sequence, Optional, Tuple
from torch import Tensor

@MODELS.register_module()
class SqueezeNet(BaseModule):
    """Backbone network using SqueezeNet architecture.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: Sequence[int] = [64, 128, 256],
                 norm_cfg: dict = dict(type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg: dict = dict(type='Conv2d', bias=False),
                 init_cfg: Optional[dict] = None,
                 pretrained: Optional[str] = None) -> None:
        super(SqueezeNet, self).__init__(init_cfg=init_cfg)

        # Define the SqueezeNet fire modules
        self.features = nn.Sequential(
            build_conv_layer(conv_cfg, in_channels, 96, kernel_size=7, stride=2),
            build_norm_layer(norm_cfg, 96)[1],
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            self._make_fire_module(96, 16, 64, 64),
            self._make_fire_module(128, 16, 64, 64),
            self._make_fire_module(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            self._make_fire_module(256, 32, 128, 128),
            self._make_fire_module(256, 48, 192, 192),
            self._make_fire_module(384, 48, 192, 192),
            self._make_fire_module(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            self._make_fire_module(512, 64, 256, 256),
        )

        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        else:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def _make_fire_module(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        return nn.Sequential(
            build_conv_layer(self.conv_cfg, in_channels, squeeze_channels, kernel_size=1),
            build_norm_layer(self.norm_cfg, squeeze_channels)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        for layer in self.features:
            x = layer(x)
            outs.append(x)
        return tuple(outs)
