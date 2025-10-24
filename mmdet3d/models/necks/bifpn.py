# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmengine.model.weight_init import constant_init
from mmengine.registry import MODELS


class BiFPNLayer(BaseModule):
    """Bidirectional Feature Pyramid Network (BiFPN) layer.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level.
        end_level (int): Index of the end input backbone level.
        stack (int): Number of BiFPN blocks.
        add_extra_convs (bool): Whether to add extra conv layers.
        extra_convs_on_inputs (bool): Whether to apply extra conv on input.
        relu_before_extra_convs (bool): Whether to apply relu before extra convs.
        no_norm_on_lateral (bool): Whether to apply norm on lateral connections.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
        act_cfg (str): Config dict for activation layer.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 stack=1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):
        super(BiFPNLayer, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.stack = stack
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.weights = nn.ParameterList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)

        # Add top-down and bottom-up paths
        for _ in range(self.stack):
            # Top-down path
            self.weights.append(nn.Parameter(torch.ones(2, len(self.lateral_convs) - 1)))
            # Bottom-up path
            self.weights.append(nn.Parameter(torch.ones(3, len(self.lateral_convs) - 1)))

        # Add extra conv layers (e.g., for RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # Build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Build top-down and bottom-up paths
        used_backbone_levels = len(laterals)
        for i in range(self.stack):
            # Top-down path
            top_down_weights = F.relu(self.weights[2 * i])
            for j in range(used_backbone_levels - 1, 0, -1):
                # Get target size from the previous level feature map
                target_size = laterals[j - 1].shape[2:]
                
                # Process each branch with proper size alignment
                branch1 = laterals[j - 1]
                
                # For interpolate path
                branch2 = F.interpolate(
                    laterals[j], size=target_size, mode='nearest')
                
                # Apply weights and sum
                weights = top_down_weights[:, j-1:j].softmax(dim=0)
                laterals[j - 1] = (
                    weights[0] * branch1 +
                    weights[1] * branch2)

            # Bottom-up path
            bottom_up_weights = F.relu(self.weights[2 * i + 1])
            for j in range(used_backbone_levels - 1):
                # Get target size from the next level feature map
                target_size = laterals[j + 1].shape[2:]
                
                # Process each branch with proper size alignment
                branch1 = laterals[j + 1]
                
                # For max pool path
                branch2 = F.max_pool2d(laterals[j], kernel_size=3, stride=2, padding=1)
                if branch2.shape[2:] != target_size:
                    branch2 = F.interpolate(branch2, size=target_size, mode='nearest')
                
                # For interpolate path
                branch3 = F.interpolate(
                    laterals[j], size=target_size, mode='nearest')
                
                # Apply weights and sum
                weights = bottom_up_weights[:, j:j+1].softmax(dim=0)
                laterals[j + 1] = (
                    weights[0] * branch1 +
                    weights[1] * branch2 +
                    weights[2] * branch3)

        # Build outputs
        outs = [laterals[i] for i in range(used_backbone_levels)]

        # Part 2: add extra levels
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[0](orig))
                else:
                    outs.append(self.fpn_convs[0](outs[-1]))
                for i in range(1, self.num_outs - used_backbone_levels):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


@MODELS.register_module()
class BiFPN(BaseModule):
    """Bidirectional Feature Pyramid Network (BiFPN).

    This is an implementation of - `EfficientDet: Scalable and Efficient Object Detection
    <https://arxiv.org/abs/1911.09070>`_
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 stack=1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):
        super(BiFPN, self).__init__()
        self.bifpn = BiFPNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            start_level=start_level,
            end_level=end_level,
            stack=stack,
            add_extra_convs=add_extra_convs,
            extra_convs_on_inputs=extra_convs_on_inputs,
            relu_before_extra_convs=relu_before_extra_convs,
            no_norm_on_lateral=no_norm_on_lateral,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, inputs):
        """Forward function."""
        return self.bifpn(inputs)
