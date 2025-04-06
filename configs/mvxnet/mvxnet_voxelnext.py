_base_ = ['../_base_/schedules/cosine.py', '../_base_/default_runtime.py']

voxel_size = [0.1, 0.1, 0.2]  # Balanced voxel size
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]  # Standard KITTI range

import torch
pcd_range_tensor = torch.tensor(point_cloud_range, dtype=torch.float32)

backend_args = None

model = dict(
    type='MVXFasterRCNN',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='hard',
        voxel_layer=dict(
            max_num_points=10,  # Increased minimum points
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(24000, 48000),  # Increased capacity
            deterministic=False  # Allow some randomness
        ),
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),

    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),

    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),

    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=4,
        feat_channels=[32, 64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range
    ),

    pts_middle_encoder=dict(
        type='SparseEncoderSparseOutput',
        in_channels=64,
        sparse_shape=[21, 256, 256],
        output_channels=64,
    ),

    pts_backbone=dict(
        type='VoxelNeXt',
        in_channels=64,
        base_channels=32,
        out_indices=(0, 1, 2)
    ),

    pts_neck=dict(
        type='SparseFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128]
    ),

    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=sum([128, 128, 128]),
        feat_channels=128,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[0, -39.68, -1.78, 69.12, 39.68, -1.78]],
            sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False
        ),
        assigner_per_size=True,  # Need separate assigners per anchor size
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=1.0/9.0, loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),

    train_cfg=dict(
        pts=dict(
            assigner=[
                dict(
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1
                ) for _ in range(3)  # One per anchor size
            ],
            allowed_border=0,
            pos_weight=-1,
            debug=False
        )
    ),
    test_cfg=dict(
        pts=dict(
            nms_pre=100,
            min_bbox_size=0,
            score_thr=0.1,
            nms=dict(type='nms', iou_thr=0.5),
            max_num=50)))

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
    #dict(
    #    type='RandomResize', scale=[(320, 96), (1280, 384)], keep_ratio=True),
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
        ann_file='kitti_infos_train.pkl',
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
        ann_file='kitti_infos_train.pkl',
        modality=modality,
        data_prefix=dict(
            pts='training/velodyne_reduced', img='training/image_2'),
        pipeline=test_pipeline,
        metainfo=metainfo,
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))

optim_wrapper = dict(
    optimizer=dict(weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)
val_evaluator = dict(
    type='KittiMetric', ann_file='data/kitti/kitti_infos_train.pkl')
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)
