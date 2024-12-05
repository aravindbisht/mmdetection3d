import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmengine.model import BaseModule
from torch import nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class SQUEEZEFPN(BaseModule):
    """FPN using SqueezeNet architecture.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        init_cfg (dict or :obj:`ConfigDict` or list[dict or :obj:`ConfigDict`],
            optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 out_channels=[256, 256, 256, 256],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 upsample_cfg=dict(type='deconv', bias=False),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 init_cfg=None):
        super(SQUEEZEFPN, self).__init__(init_cfg=init_cfg)
        assert len(out_channels) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.deblocks = nn.ModuleList()
        for i, out_channel in enumerate(out_channels):
            upsample_layer = build_upsample_layer(
                upsample_cfg,
                in_channels=in_channels[i],
                out_channels=out_channel,
                kernel_size=2,
                stride=2)
            deblock = nn.Sequential(
                upsample_layer,
                build_norm_layer(norm_cfg, out_channel)[1],
                nn.ReLU(inplace=True)
            )
            self.deblocks.append(deblock)

    def forward(self, x):
        """Forward function.

        Args:
            x (List[torch.Tensor]): Multi-level features with 4D Tensor in
                (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        return [out]
