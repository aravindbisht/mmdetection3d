"""
FireRPFNet2D: Efficient image backbone using Fire modules + CBAM + Residual.

This is the image counterpart to FireRPFNet (BEV backbone), enabling a unified
architecture design for multi-modal 3D object detection.

"""

import torch
from torch import nn
from mmcv.cnn import build_norm_layer
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from typing import Sequence, Optional


class FireBlock(nn.Module):
    """SqueezeNet-style fire module with residual shortcut.

    Args:
        in_ch (int): Input channels.
        out_ch (int): Output channels.
        norm_cfg (dict): Normalization config.
        stride (int): Stride for downsampling (default: 1).
    """

    def __init__(self, in_ch, out_ch, norm_cfg, stride=1):
        super().__init__()
        squeeze_ch = max(16, in_ch // 4)

        # Squeeze layer
        self.squeeze = nn.Conv2d(in_ch, squeeze_ch, kernel_size=1,
                                stride=stride, bias=False)
        self.squeeze_bn = build_norm_layer(norm_cfg, squeeze_ch)[1]

        # Expand paths (parallel)
        self.expand1x1 = nn.Conv2d(squeeze_ch, out_ch // 2, 1, bias=False)
        self.expand3x3 = nn.Conv2d(squeeze_ch, out_ch // 2, 3, padding=1, bias=False)
        self.expand_bn = build_norm_layer(norm_cfg, out_ch)[1]

        self.act = nn.ReLU(inplace=True)

        # Residual connection with potential downsampling
        self.downsample = None
        if in_ch != out_ch or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                build_norm_layer(norm_cfg, out_ch)[1],
            )

    def forward(self, x):
        identity = x

        # Squeeze phase
        x = self.act(self.squeeze_bn(self.squeeze(x)))

        # Expand phase (parallel paths)
        out1 = self.expand1x1(x)
        out3 = self.expand3x3(x)
        out = torch.cat([out1, out3], dim=1)
        out = self.expand_bn(out)

        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(identity)

        return self.act(out + identity)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM).

    Sequential channel and spatial attention for adaptive feature refinement.
    Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018

    Args:
        ch (int): Number of channels.
        reduction (int): Reduction ratio for channel attention (default: 16).
    """

    def __init__(self, ch, reduction=16):
        super().__init__()

        # Channel attention
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid(),
        )

        # Spatial attention
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # Channel attention: "What to focus on"
        att_c = self.mlp(x).view(b, c, 1, 1)
        x = x * att_c

        # Spatial attention: "Where to focus on"
        att_s = self.spatial(
            torch.cat([x.mean(1, keepdim=True),
                      x.max(1, keepdim=True)[0]], dim=1)
        )
        x = x * att_s

        return x


@MODELS.register_module()
class FireRPFNet2D(BaseModule):
    """FireRPFNet for 2D image features (image backbone).

    Efficient CNN backbone combining Fire modules (SqueezeNet-inspired),
    CBAM attention, and residual learning. Designed as a unified building
    block for multi-modal 3D object detection.

    This is the image counterpart to FireRPFNet (used for BEV features).

    Args:
        in_channels (int): Input channels (typically 3 for RGB).
        layer_channels (Sequence[int]): Output channels for each stage.
        layer_blocks (Sequence[int]): Number of blocks per stage (default: [2,2,2,2]).
        with_cbam (bool): Whether to use CBAM attention (default: True).
        norm_cfg (dict): Normalization config.
        init_cfg (dict): Initialization config.

    Example:
        >>> # For images (replace SQUEEZE backbone)
        >>> backbone = FireRPFNet2D(
        ...     in_channels=3,
        ...     layer_channels=[64, 128, 256, 512],
        ...     with_cbam=True
        ... )
    """

    def __init__(self,
                 in_channels: int = 3,
                 layer_channels: Sequence[int] = (64, 128, 256, 512),
                 layer_blocks: Sequence[int] = (2, 2, 2, 2),
                 with_cbam: bool = True,
                 norm_cfg: dict = dict(type='BN', eps=1e-3, momentum=0.01),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.layer_channels = layer_channels
        self.with_cbam = with_cbam

        # Initial stem (7x7 conv + pool for downsampling)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, layer_channels[0], kernel_size=7,
                     stride=2, padding=3, bias=False),
            build_norm_layer(norm_cfg, layer_channels[0])[1],
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Build stages
        self.stages = nn.ModuleList()
        ch = layer_channels[0]

        for stage_idx, (out_ch, num_blocks) in enumerate(zip(layer_channels, layer_blocks)):
            stage = []

            for block_idx in range(num_blocks):
                # First block in each stage may downsample
                stride = 2 if (block_idx == 0 and stage_idx > 0) else 1
                in_ch = ch if block_idx > 0 else ch

                # FireBlock
                fire_block = FireBlock(in_ch, out_ch, norm_cfg, stride)
                stage.append(fire_block)

                # CBAM after each FireBlock
                if with_cbam:
                    stage.append(CBAM(out_ch))

                ch = out_ch

            self.stages.append(nn.Sequential(*stage))

        # Store output indices for FPN
        self.out_indices = tuple(range(len(layer_channels)))

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input image tensor (B, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale feature maps.
        """
        # Stem
        x = self.stem(x)

        # Stages (collect multi-scale outputs)
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)

        return tuple(outs)


@MODELS.register_module()
class FireRPFNet2DSimple(BaseModule):
    """Simplified FireRPFNet2D with single block per stage (faster).

    Use this for a lighter image backbone when you want maximum efficiency.
    """

    def __init__(self,
                 in_channels: int = 3,
                 layer_channels: Sequence[int] = (64, 128, 256, 512),
                 with_cbam: bool = True,
                 norm_cfg: dict = dict(type='BN', eps=1e-3, momentum=0.01),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)

        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, layer_channels[0], kernel_size=7,
                     stride=2, padding=3, bias=False),
            build_norm_layer(norm_cfg, layer_channels[0])[1],
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Build stages (single block per stage)
        self.stages = nn.ModuleList()
        ch = layer_channels[0]

        for stage_idx, out_ch in enumerate(layer_channels):
            stage = []

            # One FireBlock per stage
            stride = 2 if stage_idx > 0 else 1
            stage.append(FireBlock(ch, out_ch, norm_cfg, stride))

            # CBAM
            if with_cbam:
                stage.append(CBAM(out_ch))

            self.stages.append(nn.Sequential(*stage))
            ch = out_ch

    def forward(self, x):
        x = self.stem(x)

        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)

        return tuple(outs)


@MODELS.register_module()
class FireRPFNetUnified(BaseModule):
    """Unified FireRPFNet for both BEV and Image processing.

    This is a flexible architecture that can be configured for:
    1. BEV processing (pts_backbone): No stem, single block per stage, no downsampling
    2. Image processing (img_backbone): With stem, multiple blocks per stage, with downsampling

    Args:
        in_channels (int): Input channels (3 for RGB, 256 for BEV).
        layer_channels (Sequence[int]): Output channels for each stage.
        layer_blocks (Sequence[int] | int): Number of blocks per stage.
            If int, same number for all stages. Default: 1.
        with_cbam (bool): Whether to use CBAM attention. Default: True.
        with_stem (bool): Whether to use stem (for image processing). Default: False.
        with_downsampling (bool): Whether to downsample between stages. Default: False.
        multi_scale_output (bool): Whether to output multi-scale features (for FPN).
            If False, returns only final output (for BEV). Default: False.
        stem_kernel_size (int): Kernel size for stem conv. Default: 7.
        stem_stride (int): Stride for stem conv. Default: 2.
        stem_pooling (bool): Whether to add MaxPool after stem. Default: True.
        norm_cfg (dict): Normalization config.
        init_cfg (dict): Initialization config.

    Example:
        >>> # For BEV processing (backward compatible with original FireRPFNet)
        >>> pts_backbone = FireRPFNetUnified(
        ...     in_channels=256,
        ...     layer_channels=[128, 256, 256, 256],
        ...     layer_blocks=1,  # Single block per stage
        ...     with_cbam=True,
        ...     with_stem=False,  # No stem for BEV
        ...     with_downsampling=False,  # Preserve resolution
        ...     multi_scale_output=False  # Single output
        ... )

        >>> # For image processing (backward compatible with FireRPFNet2D)
        >>> img_backbone = FireRPFNetUnified(
        ...     in_channels=3,
        ...     layer_channels=[64, 128, 256, 512],
        ...     layer_blocks=[2, 2, 2, 2],  # Multiple blocks per stage
        ...     with_cbam=True,
        ...     with_stem=True,  # Stem for raw images
        ...     with_downsampling=True,  # Progressive downsampling
        ...     multi_scale_output=True  # Multi-scale for FPN
        ... )

        >>> # Lightweight image backbone
        >>> img_backbone_lite = FireRPFNetUnified(
        ...     in_channels=3,
        ...     layer_channels=[64, 128, 256, 512],
        ...     layer_blocks=1,  # Single block per stage (faster)
        ...     with_cbam=True,
        ...     with_stem=True,
        ...     with_downsampling=True,
        ...     multi_scale_output=True
        ... )
    """

    def __init__(self,
                 in_channels: int,
                 layer_channels: Sequence[int],
                 layer_blocks: Sequence[int] | int = 1,
                 with_cbam: bool = True,
                 with_stem: bool = False,
                 with_downsampling: bool = False,
                 multi_scale_output: bool = False,
                 stem_kernel_size: int = 7,
                 stem_stride: int = 2,
                 stem_pooling: bool = True,
                 norm_cfg: dict = dict(type='BN', eps=1e-3, momentum=0.01),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.layer_channels = layer_channels
        self.with_cbam = with_cbam
        self.with_stem = with_stem
        self.with_downsampling = with_downsampling
        self.multi_scale_output = multi_scale_output

        # Handle layer_blocks: can be int or sequence
        if isinstance(layer_blocks, int):
            self.layer_blocks = [layer_blocks] * len(layer_channels)
        else:
            self.layer_blocks = list(layer_blocks)
            assert len(self.layer_blocks) == len(layer_channels), \
                f"layer_blocks ({len(self.layer_blocks)}) must match layer_channels ({len(layer_channels)})"

        # Build stem if needed (for image processing)
        if with_stem:
            stem_layers = [
                nn.Conv2d(in_channels, layer_channels[0],
                         kernel_size=stem_kernel_size,
                         stride=stem_stride,
                         padding=stem_kernel_size // 2,
                         bias=False),
                build_norm_layer(norm_cfg, layer_channels[0])[1],
                nn.ReLU(inplace=True)
            ]
            if stem_pooling:
                stem_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.stem = nn.Sequential(*stem_layers)
            ch = layer_channels[0]
        else:
            self.stem = None
            ch = in_channels

        # Build stages
        self.stages = nn.ModuleList()

        for stage_idx, (out_ch, num_blocks) in enumerate(zip(layer_channels, self.layer_blocks)):
            stage = []

            for block_idx in range(num_blocks):
                # Determine stride for this block
                if with_downsampling and block_idx == 0 and stage_idx > 0:
                    # First block of stages 1+ downsamples
                    stride = 2
                else:
                    stride = 1

                # Input channels for this block
                in_ch = ch if block_idx > 0 else ch

                # Add FireBlock
                fire_block = FireBlock(in_ch, out_ch, norm_cfg, stride)
                stage.append(fire_block)

                # Add CBAM if enabled
                if with_cbam:
                    stage.append(CBAM(out_ch))

                ch = out_ch

            self.stages.append(nn.Sequential(*stage))

        # Store output indices for multi-scale output
        if multi_scale_output:
            self.out_indices = tuple(range(len(layer_channels)))
        else:
            self.out_indices = (len(layer_channels) - 1,)  # Only last stage

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor (B, C, H, W).

        Returns:
            tuple[torch.Tensor]: Feature maps.
                - If multi_scale_output=True: Multi-level features for FPN
                - If multi_scale_output=False: Single output (final features)
        """
        # Apply stem if present
        if self.stem is not None:
            x = self.stem(x)

        # Process through stages
        outs = []
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)
            if stage_idx in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def __repr__(self):
        """String representation for debugging."""
        mode = "Image" if self.with_stem else "BEV"
        output_type = "Multi-scale" if self.multi_scale_output else "Single-scale"
        return (f"{self.__class__.__name__}(\n"
                f"  Mode: {mode},\n"
                f"  Channels: {self.layer_channels},\n"
                f"  Blocks per stage: {self.layer_blocks},\n"
                f"  CBAM: {self.with_cbam},\n"
                f"  Downsampling: {self.with_downsampling},\n"
                f"  Output: {output_type}\n"
                f")")

