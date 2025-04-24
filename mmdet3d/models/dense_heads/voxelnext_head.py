from typing import List

import torch
from mmdet3d.registry import MODELS
from mmdet3d.models.dense_heads.centerpoint_head import CenterHead


@MODELS.register_module()
class VoxelNeXtHead(CenterHead):
    """Light wrapper around CenterHead with VoxelNeXt defaults.

    Keeps CenterHead's tested loss / decode logic but matches the lightweight
    128-channel BEV feature coming from our VoxelNeXt backbone.
    """

    def __init__(self,
                 in_channels: int = 128,
                 feat_channels: int = 128,
                 num_classes: int = 3,
                 pc_range: List[float] = [0, -40, -3, 70.4, 40, 1],
                 voxel_size: List[float] = [0.05, 0.05, 0.1],
                 out_stride: int = 8,
                 with_velocity: bool = False,
                 **kwargs):

        tasks = (dict(num_class=num_classes,
                      class_names=['Pedestrian', 'Cyclist', 'Car']),)

        common_heads = dict(
            center=(2, feat_channels),
            reg=(2, feat_channels),
            height=(1, feat_channels),
            dim=(3, feat_channels),
            rot=(2, feat_channels))
        code_size = 7
        if with_velocity:
            common_heads['vel'] = (2, feat_channels)
            code_size = 9

        super().__init__(
            in_channels=in_channels,
            tasks=tasks,
            share_conv_channel=feat_channels,
            common_heads=common_heads,
            separate_head=dict(type='SeparateHead', init_bias=-2.19, final_kernel=3),
            bbox_coder=dict(
                type='CenterPointBBoxCoder',
                pc_range=pc_range,
                post_center_range=pc_range,
                max_num=50,
                score_threshold=0.1,
                out_size_factor=out_stride,
                voxel_size=voxel_size,
                code_size=code_size),
            **kwargs)

        # default Gaussian focal / L1 losses defined in parent
