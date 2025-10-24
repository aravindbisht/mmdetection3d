"""Enhanced SqueezeNet backbone with pretrained model support.

This version adds:
- Pretrained model loading (ImageNet weights)
- frozen_stages parameter to freeze early layers during training
- norm_eval parameter to keep BatchNorm layers in eval mode
- Full backward compatibility with original SQUEEZE backbone
"""

import warnings
from typing import Optional, Sequence

import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.model import BaseModule
from mmengine.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet3d.registry import MODELS


@MODELS.register_module()
class SQUEEZEv2(BaseModule):
    """Enhanced SqueezeNet backbone with pretrained support.

    This backbone supports pretrained models (like ImageNet-pretrained SqueezeNet)
    and provides fine-grained control over which layers to freeze during training.

    Args:
        in_channels (int): Number of input channels. Default: 3.
        out_channels (Sequence[int]): Output channels for multi-scale feature maps.
            Default: [64, 128, 256, 512].
        frozen_stages (int): Stages to be frozen (stop gradient and set eval mode).
            -1 means not freezing any parameters. Default: -1.
            Stage 0: stem (first conv + BN + ReLU + maxpool)
            Stage 1: Fire modules 1-2 (after first maxpool)
            Stage 2: Fire modules 3-4 (after second maxpool)
            Stage 3: Fire modules 5-7 (after third maxpool)
            Stage 4: Fire module 8 (final fire module)
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: True.
        norm_cfg (dict): Config dict of normalization layers.
            Default: dict(type='BN', eps=1e-3, momentum=0.01).
        conv_cfg (dict): Config dict of convolutional layers.
            Default: dict(type='Conv2d', bias=False).
        init_cfg (Optional[dict]): Initialization config dict.
            Examples:
            - For pretrained ImageNet model:
              dict(type='Pretrained', checkpoint='path/to/squeezenet1_1.pth')
            - For random initialization (default):
              dict(type='Kaiming', layer='Conv2d')
        pretrained (Optional[str]): Deprecated. Use init_cfg instead.
            Path to pretrained weights.

    Example:
        >>> # With pretrained ImageNet weights and frozen early stages
        >>> model = dict(
        ...     type='SQUEEZEv2',
        ...     in_channels=3,
        ...     out_channels=[64, 128, 256, 512],
        ...     frozen_stages=1,
        ...     norm_eval=True,
        ...     init_cfg=dict(
        ...         type='Pretrained',
        ...         checkpoint='path/to/squeezenet_imagenet.pth'))
        >>>
        >>> # Without pretrained weights (backward compatible)
        >>> model = dict(
        ...     type='SQUEEZEv2',
        ...     in_channels=3,
        ...     out_channels=[64, 128, 256, 512])
    """

    # Define stage boundaries for freezing
    # Stage 0: [0] - stem
    # Stage 1: [1, 4] - fire modules 1-2 and first maxpool
    # Stage 2: [5, 7] - fire module 3 and second maxpool
    # Stage 3: [8, 12] - fire modules 4-7 and third maxpool
    # Stage 4: [13] - fire module 8
    STAGE_RANGES = {
        0: (0, 0),      # stem only
        1: (1, 4),      # up to and including first set of fire modules
        2: (5, 7),      # second set
        3: (8, 12),     # third set
        4: (13, 13)     # final fire module
    }

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: Sequence[int] = [64, 128, 256, 512],
                 frozen_stages: int = -1,
                 norm_eval: bool = True,
                 norm_cfg: dict = dict(type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg: dict = dict(type='Conv2d', bias=False),
                 init_cfg: Optional[dict] = None,
                 pretrained: Optional[str] = None) -> None:
        super(SQUEEZEv2, self).__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # Handle deprecated pretrained parameter
        if isinstance(pretrained, str):
            warnings.warn(
                'DeprecationWarning: pretrained is deprecated, '
                'please use "init_cfg" instead. Example: '
                'init_cfg=dict(type="Pretrained", checkpoint="path/to/weights.pth")',
                DeprecationWarning)
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = dict(type='Kaiming', layer='Conv2d')

        # Build the SqueezeNet architecture
        # Stage 0: Stem
        self.features = nn.ModuleList([
            # [0] Initial conv block (stem)
            nn.Sequential(
                build_conv_layer(conv_cfg, in_channels, 64, kernel_size=3, stride=2),
                build_norm_layer(norm_cfg, 64)[1],
                nn.ReLU(inplace=True)
            ),
            # [1] First maxpool
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            # Stage 1: Fire modules 1-2
            # [2] Fire module 1
            self._make_fire_module(64, 16, 64, 64),
            # [3] Fire module 2
            self._make_fire_module(128, 16, 64, 64),
            # [4] Fire module 3
            self._make_fire_module(128, 32, 128, 128),
            # Stage 2: Second maxpool + fire modules
            # [5] Second maxpool
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            # [6] Fire module 4
            self._make_fire_module(256, 32, 128, 128),
            # [7] Fire module 5
            self._make_fire_module(256, 48, 192, 192),
            # Stage 3: Fire modules 6-7
            # [8] Fire module 6
            self._make_fire_module(384, 48, 192, 192),
            # [9] Fire module 7
            self._make_fire_module(384, 64, 256, 256),
            # [10] Third maxpool
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            # Stage 4: Final fire module
            # [11] Fire module 8
            self._make_fire_module(512, 64, 256, 256),
        ])

        # Indices where we extract features for multi-scale output
        # To match original SQUEEZE behavior: [64, 128, 256, 512] channels
        # idx=1: after maxpool1 (64ch), idx=3: after fire2 (128ch),
        # idx=6: after fire4 (256ch), idx=11: after fire8 (512ch)
        self.out_indices = [1, 3, 6, 11]  # Matches original SQUEEZE: [64, 128, 256, 512]

    def _load_torchvision_checkpoint(self, checkpoint_path):
        """Load torchvision SqueezeNet weights and adapt to our structure.

        Torchvision SqueezeNet doesn't have BatchNorm layers, so we only load
        the convolutional weights and initialize BN layers randomly.
        """
        # Load the checkpoint
        checkpoint = load_checkpoint(self, checkpoint_path, map_location='cpu', strict=False)

        # Print info about what was loaded
        print(f"Loaded pretrained SqueezeNet weights from torchvision.")
        print(f"Note: BatchNorm layers are randomly initialized (not in torchvision model).")

        return checkpoint

    def _make_fire_module(self, in_channels, squeeze_channels,
                         expand1x1_channels, expand3x3_channels):
        """Create a Fire module.

        A Fire module consists of:
        1. Squeeze layer: 1x1 conv to reduce channels
        2. Expand layer: parallel 1x1 and 3x3 convs, concatenated

        Args:
            in_channels (int): Input channels.
            squeeze_channels (int): Channels after squeeze layer.
            expand1x1_channels (int): Channels for 1x1 expand.
            expand3x3_channels (int): Channels for 3x3 expand.

        Returns:
            nn.Sequential: The fire module.
        """
        layers = nn.Sequential()

        # Squeeze layer
        squeeze = nn.Sequential(
            build_conv_layer(self.conv_cfg, in_channels, squeeze_channels,
                           kernel_size=1),
            build_norm_layer(self.norm_cfg, squeeze_channels)[1],
            nn.ReLU(inplace=True)
        )
        layers.add_module('squeeze', squeeze)

        # Expand 1x1 layer
        expand1x1 = nn.Sequential(
            build_conv_layer(self.conv_cfg, squeeze_channels, expand1x1_channels,
                           kernel_size=1),
            build_norm_layer(self.norm_cfg, expand1x1_channels)[1],
            nn.ReLU(inplace=True)
        )
        layers.add_module('expand1x1', expand1x1)

        # Expand 3x3 layer
        expand3x3 = nn.Sequential(
            build_conv_layer(self.conv_cfg, squeeze_channels, expand3x3_channels,
                           kernel_size=3, padding=1),
            build_norm_layer(self.norm_cfg, expand3x3_channels)[1],
            nn.ReLU(inplace=True)
        )
        layers.add_module('expand3x3', expand3x3)

        return layers

    def _freeze_stages(self):
        """Freeze parameters of specified stages.

        This method sets the specified stages to eval mode and freezes
        their parameters (sets requires_grad=False).
        """
        if self.frozen_stages >= 0:
            # Freeze stem (stage 0)
            stem = self.features[0]
            stem.eval()
            for param in stem.parameters():
                param.requires_grad = False

        # Freeze subsequent stages based on frozen_stages
        # Stage 1: indices 1-4, Stage 2: indices 5-7, etc.
        for stage_idx in range(1, self.frozen_stages + 1):
            if stage_idx in self.STAGE_RANGES:
                start_idx, end_idx = self.STAGE_RANGES[stage_idx]
                for idx in range(start_idx, end_idx + 1):
                    if idx < len(self.features):
                        m = self.features[idx]
                        m.eval()
                        for param in m.parameters():
                            param.requires_grad = False

    def train(self, mode: bool = True):
        """Set training mode and handle frozen stages.

        Args:
            mode (bool): Whether to set training mode (True) or eval mode (False).
        """
        super(SQUEEZEv2, self).train(mode)
        self._freeze_stages()

        if mode and self.norm_eval:
            # Keep all BatchNorm layers in eval mode even during training
            # This freezes the running statistics (mean and variance)
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            list[torch.Tensor]: Multi-scale feature maps.
        """
        # Process through initial conv/stem
        x = self.features[0](x)

        outs = []
        for idx, layer in enumerate(self.features[1:], 1):
            if isinstance(layer, nn.Sequential) and 'squeeze' in layer._modules:
                # This is a Fire module, handle concatenation
                squeeze_output = layer.squeeze(x)
                x1 = layer.expand1x1(squeeze_output)
                x3 = layer.expand3x3(squeeze_output)
                x = torch.cat([x1, x3], 1)
            else:
                # Regular layer (maxpool, etc.)
                x = layer(x)

            # Collect outputs at specified indices
            if idx in self.out_indices:
                outs.append(x)

        return outs


@MODELS.register_module()
class SQUEEZE_Pretrained(SQUEEZEv2):
    """Alias for SQUEEZEv2 with clearer naming for pretrained usage.

    This is simply an alias to SQUEEZEv2, provided for clarity when
    using pretrained models.
    """
    pass