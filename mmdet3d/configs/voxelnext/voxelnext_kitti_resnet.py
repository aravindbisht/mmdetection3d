# Copyright (c) OpenMMLab. All rights reserved.
custom_imports = dict(
    imports=[
        'mmdet.models.backbones.identity',
        'mmdet3d.models.detectors.voxelnext',
        'mmdet3d.models.dense_heads.voxelnext_head',
        'mmdet3d.datasets.kitti'
    ],
    allow_failed_imports=False)

# Basic parameters
voxel_size = [0.05, 0.05, 0.1]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
# Model configuration
model = dict(
    type='mmdet3d.VoxelNeXt',
    data_preprocessor=dict(
        type='mmdet3d.Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000),
            point_cloud_range=point_cloud_range)),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    pts_voxel_encoder=dict(
        type='mmdet3d.HardVFE', 
        in_channels=4,  
        feat_channels=[64],
        with_distance=False),
    pts_middle_encoder=dict(
        type='mmdet3d.SparseEncoderVOXELNEXT',
        in_channels=64,
        sparse_shape=[41, 1600, 1408],
        output_channels=128,
        order=('conv', 'norm', 'act')),
    pts_backbone=dict(type='nn.Identity'),
    pts_bbox_head=dict(
        type='mmdet3d.VoxelNeXtHead',
        in_channels=128,
        tasks=[dict(num_class=1, class_names=['Pedestrian']), 
               dict(num_class=1, class_names=['Cyclist']),
               dict(num_class=1, class_names=['Car'])],
        bbox_coder=dict(
            type='mmdet3d.VoxelNeXtBBoxCoder',
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            out_size_factor=8,
            code_size=9)),
    train_cfg=dict(
        pts=dict(
            grid_size=[1600, 1408, 40],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=100,
            min_radius=2)),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=point_cloud_range,
            max_per_img=100,
            score_threshold=0.1,
            nms_thr=0.2)))


# dataset settings
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='RandomResize', scale=[(640, 192), (2560, 768)], keep_ratio=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.2, 0.2, 0.2]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1280, 384),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            # Temporary solution, fix this after refactor the augtest
            dict(type='Resize', scale=0, keep_ratio=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        ]),
    dict(type='Pack3DDetInputs', keys=['points', 'img'])
]
modality = dict(use_lidar=True, use_camera=True)
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            modality=modality,
            ann_file='kitti_infos_train.pkl',
            data_prefix=dict(
                pts='training/velodyne_reduced', img='training/image_2'),
            pipeline=train_pipeline,
            filter_empty_gt=False,
            metainfo=metainfo,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            backend_args=backend_args)))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        modality=modality,
        ann_file='kitti_infos_val.pkl',
        data_prefix=dict(
            pts='training/velodyne_reduced', img='training/image_2'),
        pipeline=test_pipeline,
        metainfo=metainfo,
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kitti_infos_val.pkl',
        modality=modality,
        data_prefix=dict(
            pts='training/velodyne_reduced', img='training/image_2'),
        pipeline=test_pipeline,
        metainfo=metainfo,
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))

# Training configuration
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=5,
    val_interval=1)


# Validation configuration
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Complete optimizer config
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

# Learning rate schedule
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        T_max=80,
        begin=0,
        end=80,
        eta_min_ratio=0.1)
]

# # Evaluation hooks
# default_hooks = dict(
#     timer=dict(type='IterTimerHook'),
#     logger=dict(type='LoggerHook', interval=50),
#     param_scheduler=dict(type='ParamSchedulerHook'),
#     checkpoint=dict(type='CheckpointHook', interval=2),
#     sampler_seed=dict(type='DistSamplerSeedHook'))

val_evaluator = dict(
    type='KittiMetric', ann_file='data/kitti/kitti_infos_val.pkl')
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(
#     type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')



# _base_ = [
#     '../_base_/datasets/kitti-3d-3class.py',
#     # '../_base_/models/voxelneXt.py',
#     #'../_base_/schedules/cyclic-20e.py',
#      '../_base_/default_runtime.py'
# ]
# from torch import nn
# backbone = nn.Identity()
# # Basic parameters
# voxel_size = [0.05, 0.05, 0.1]
# point_cloud_range = [0, -40, -3, 70.4, 40, 1]
# # Model configuration
# model = dict(
#     type='mmdet3d.VoxelNeXt',
#     data_preprocessor=dict(
#         type='mmdet3d.Det3DDataPreprocessor',
#         voxel=True,
#         voxel_layer=dict(
#             max_num_points=5,
#             voxel_size=voxel_size,
#             max_voxels=(16000, 40000),
#             point_cloud_range=point_cloud_range)),
#     img_backbone=dict(
#         type='mmdet.ResNet',
#         depth=50,
#         num_stages=4,
#         out_indices=(0, 1, 2, 3),
#         frozen_stages=1,
#         norm_cfg=dict(type='BN', requires_grad=True),
#         norm_eval=True,
#         style='pytorch',
#         init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
#     img_neck=dict(
#         type='mmdet.FPN',
#         in_channels=[256, 512, 1024, 2048],
#         out_channels=256,
#         num_outs=5),
#     pts_voxel_encoder=dict(
#         type='mmdet3d.HardVFE', 
#         in_channels=4,  
#         feat_channels=[64],
#         with_distance=False),
#     pts_middle_encoder=dict(
#         type='mmdet3d.SparseEncoderVOXELNEXT',
#         in_channels=64,
#         sparse_shape=[41, 1600, 1408],
#         output_channels=128,
#         base_channels=16,
#         order=('conv', 'norm', 'act')),
#     pts_backbone=backbone,
#     pts_bbox_head=dict(
#         type='mmdet3d.VoxelNeXtHead',
#         in_channels=128,
#         tasks=[dict(num_class=1, class_names=['Pedestrian']), 
#                dict(num_class=1, class_names=['Cyclist']),
#                dict(num_class=1, class_names=['Car'])],
#         bbox_coder=dict(
#             type='mmdet3d.VoxelNeXtBBoxCoder',
#             voxel_size=voxel_size[:2],
#             pc_range=point_cloud_range[:2],
#             out_size_factor=8,
#             code_size=9)),
#     train_cfg=dict(
#         pts=dict(
#             grid_size=[1600, 1408, 40],
#             voxel_size=voxel_size,
#             point_cloud_range=point_cloud_range,
#             out_size_factor=4,
#             dense_reg=1,
#             gaussian_overlap=0.1,
#             max_objs=100,
#             min_radius=2)),
#     test_cfg=dict(
#         pts=dict(
#             post_center_limit_range=point_cloud_range,
#             max_per_img=100,
#             score_threshold=0.1,
#             nms_thr=0.2)))


# # dataset settings
# dataset_type = 'KittiDataset'
# data_root = 'data/kitti/'
# class_names = ['Pedestrian', 'Cyclist', 'Car']
# metainfo = dict(classes=class_names)
# input_modality = dict(use_lidar=True, use_camera=True)
# backend_args = None
# train_pipeline = [
#     dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=4,
#         use_dim=4,
#         backend_args=backend_args),
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
#     dict(
#         type='RandomResize', scale=[(640, 192), (2560, 768)], keep_ratio=True),
#     dict(
#         type='GlobalRotScaleTrans',
#         rot_range=[-0.78539816, 0.78539816],
#         scale_ratio_range=[0.95, 1.05],
#         translation_std=[0.2, 0.2, 0.2]),
#     dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
#     dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
#     dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
#     dict(type='PointShuffle'),
#     dict(
#         type='Pack3DDetInputs',
#         keys=[
#             'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
#             'gt_labels'
#         ])
# ]
# test_pipeline = [
#     dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=4,
#         use_dim=4,
#         backend_args=backend_args),
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(
#         type='MultiScaleFlipAug3D',
#         img_scale=(1280, 384),
#         pts_scale_ratio=1,
#         flip=False,
#         transforms=[
#             # Temporary solution, fix this after refactor the augtest
#             dict(type='Resize', scale=0, keep_ratio=True),
#             dict(
#                 type='GlobalRotScaleTrans',
#                 rot_range=[0, 0],
#                 scale_ratio_range=[1., 1.],
#                 translation_std=[0, 0, 0]),
#             dict(type='RandomFlip3D'),
#             dict(
#                 type='PointsRangeFilter', point_cloud_range=point_cloud_range),
#         ]),
#     dict(type='Pack3DDetInputs', keys=['points', 'img'])
# ]
# modality = dict(use_lidar=True, use_camera=True)
# train_dataloader = dict(
#     batch_size=1,
#     num_workers=1,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     dataset=dict(
#         type='RepeatDataset',
#         times=2,
#         dataset=dict(
#             type=dataset_type,
#             data_root=data_root,
#             modality=modality,
#             ann_file='kitti_infos_train.pkl',
#             data_prefix=dict(
#                 pts='training/velodyne_reduced', img='training/image_2'),
#             pipeline=train_pipeline,
#             filter_empty_gt=False,
#             metainfo=metainfo,
#             # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
#             # and box_type_3d='Depth' in sunrgbd and scannet dataset.
#             box_type_3d='LiDAR',
#             backend_args=backend_args)))

# val_dataloader = dict(
#     batch_size=1,
#     num_workers=1,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         modality=modality,
#         ann_file='kitti_infos_train.pkl',
#         data_prefix=dict(
#             pts='training/velodyne_reduced', img='training/image_2'),
#         pipeline=test_pipeline,
#         metainfo=metainfo,
#         test_mode=True,
#         box_type_3d='LiDAR',
#         backend_args=backend_args))
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=1,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='kitti_infos_train.pkl',
#         modality=modality,
#         data_prefix=dict(
#             pts='training/velodyne_reduced', img='training/image_2'),
#         pipeline=test_pipeline,
#         metainfo=metainfo,
#         test_mode=True,
#         box_type_3d='LiDAR',
#         backend_args=backend_args))

# # Training configuration
# train_cfg = dict(
#     type='EpochBasedTrainLoop',
#     max_epochs=5,
#     val_interval=1)


# # Validation configuration
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')

# # Complete optimizer config
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
#     clip_grad=dict(max_norm=35, norm_type=2))

# # Learning rate schedule
# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=0.001,
#         begin=0,
#         end=500),
#     dict(
#         type='CosineAnnealingLR',
#         T_max=80,
#         begin=0,
#         end=80,
#         eta_min_ratio=0.1)
# ]

# val_evaluator = dict(
#     type='KittiMetric', ann_file='data/kitti/kitti_infos_train.pkl')
# test_evaluator = val_evaluator

# vis_backends = [dict(type='LocalVisBackend')]
