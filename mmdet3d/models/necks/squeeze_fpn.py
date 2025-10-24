import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmengine.model import BaseModule
from torch import nn

from mmdet3d.registry import MODELS


class LastLevelMaxPool(nn.Module):
    def __init__(self):
        super(LastLevelMaxPool, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        return self.pool(x)


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
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Build lateral convs
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels[0], kernel_size=1)
            for in_ch in in_channels
        ])
        
        # Build FPN convs
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, padding=1)
            for _ in range(len(in_channels))
        ])
        
        self.last_level_pool = LastLevelMaxPool()

    def forward(self, x):
        """Forward function.

        Args:
            x (List[torch.Tensor]): Multi-level features with 4D Tensor in
                (N, C, H, W) shape.

        Returns:
            tuple[torch.Tensor]: Multi-level feature maps.
        """
        assert len(x) == len(self.in_channels), \
            f'Length of inputs ({len(x)}) must match in_channels ({len(self.in_channels)})'
        
        # Apply lateral convs
        laterals = [lateral_conv(feat) for lateral_conv, feat in zip(self.lateral_convs, x)]
        
        # Build top-down path
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += nn.functional.interpolate(
                laterals[i],
                size=prev_shape,
                mode='nearest'
            )
        
        # Apply FPN convs
        outs = [fpn_conv(feat) for fpn_conv, feat in zip(self.fpn_convs, laterals)]
        
        # Add max pool for the last level if training
        if self.training:
            outs.append(self.last_level_pool(outs[0]))
        
        return tuple(outs)


@MODELS.register_module()
class SQUEEZEFPNV2(BaseModule):
    """Enhanced FPN for SqueezeNet architecture with dynamic channel handling.

    This version maintains compatibility with pretrained backbones while supporting
    unified output channels for all levels.
    """
    def __init__(self,
                 in_channels=[64, 128, 256, 512],  # From backbone
                 out_channels=[512, 512, 512, 512],  # Unified output channels for all levels
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Validate input/output channels
        if len(out_channels) != len(in_channels):
            raise ValueError(f'out_channels ({len(out_channels)}) must have same length as in_channels ({len(in_channels)})')
        
        # Build lateral convs to project backbone features to target channels
        self.lateral_convs = nn.ModuleList()
        for in_ch, out_ch in zip(in_channels, out_channels):
            # Always use a 1x1 conv to project to target channels
            # This ensures consistent feature dimensions across all levels
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                    build_norm_layer(norm_cfg, out_ch)[1],
                    nn.ReLU(inplace=True)
                )
            )
        
        # Build FPN convs for feature refinement
        self.fpn_convs = nn.ModuleList()
        for out_ch in out_channels:
            # Use a simple 3x3 conv for feature refinement
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    build_norm_layer(norm_cfg, out_ch)[1],
                    nn.ReLU(inplace=True)
                )
            )
        
        # For the last level pooling
        self.last_level_pool = LastLevelMaxPool()
        self.pool_conv = nn.Sequential(
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=1, bias=False),
            build_norm_layer(norm_cfg, out_channels[0])[1],
            nn.ReLU(inplace=True)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        import torch.nn.functional as F
        
        assert len(x) == len(self.in_channels), \
            f'Input length ({len(x)}) must match in_channels ({len(self.in_channels)})'
        
        # 1. Apply lateral convs to project all inputs to target channels
        laterals = []
        for i, (feat, lateral_conv) in enumerate(zip(x, self.lateral_convs)):
            # Ensure the input has the expected number of channels
            if feat.size(1) != self.in_channels[i]:
                # If channel dimension doesn't match, use the first conv to project
                if hasattr(lateral_conv, '0') and isinstance(lateral_conv[0], nn.Conv2d):
                    proj_conv = nn.Conv2d(feat.size(1), self.out_channels[i], kernel_size=1).to(feat.device)
                    feat = proj_conv(feat)
                else:
                    # If no conv is available, use zero-padding or truncation
                    if feat.size(1) < self.out_channels[i]:
                        # Zero-pad if input has fewer channels
                        pad = torch.zeros_like(feat[:, :self.out_channels[i] - feat.size(1)])
                        feat = torch.cat([feat, pad], dim=1)
                    else:
                        # Truncate if input has more channels
                        feat = feat[:, :self.out_channels[i]]
                laterals.append(lateral_conv(feat) if not isinstance(lateral_conv, nn.Identity) else feat)
            else:
                laterals.append(lateral_conv(feat) if not isinstance(lateral_conv, nn.Identity) else feat)
        
        # 2. Build top-down path
        used_levels = len(laterals)
        for i in range(used_levels - 1, 0, -1):
            # Upsample current level features to match previous level's spatial size
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=prev_shape,
                mode='nearest'
            )
        
        # 3. Apply FPN convs for feature refinement
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_levels)]
        
        # 4. Add max pool for the last level if training
        if self.training:
            pool_feat = self.pool_conv(self.last_level_pool(outs[0]))
            outs.append(pool_feat)
        
        return tuple(outs)
