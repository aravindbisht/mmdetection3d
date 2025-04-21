_base_ = [
    '../_base_/schedules/schedule-2x.py',
    '../_base_/default_runtime.py'
]

# model settings
voxel_size = [0.05, 0.05, 0.1]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

model = dict(
    type='DynamicMVXFasterRCNN',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='dynamic',
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(-1, -1)),
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
        style='caffe'),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_cfg=dict(type='BN', requires_grad=False),
        num_outs=5),
    pts_voxel_encoder=dict(
        type='DynamicVFE',
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=64,
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act')),
    pts_backbone=dict(
        type='OptimizedVoxelNeXtBackbone',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[128, 128, 128],
        sparse_shape=[41, 1600, 1408],
        with_cp=False,
        use_sparse_conv=True),
    pts_neck=dict(
        type='OptimizedVoxelNeXtNeck',
        in_channels=[128, 128, 128],
        out_channels=[384, 384, 384],
        upsample_strides=[1, 2, 4],
        sparse_shape=[41, 1600, 1408],
        use_sparse_conv=True),
    pts_bbox_head=dict(
        type='OptimizedVoxelNeXtHead',
        in_channels=384,
        feat_channels=384,
        use_sparse_conv=True,
        num_classes=3,
        fusion_layer=dict(
            type='AttentionFusion',
            img_channels=256,
            pts_channels=384,
            mid_channels=128,
            out_channels=384,
            img_levels=[0, 1, 2, 3, 4],
            align_corners=False,
            activate_out=True,
            fuse_out=True),
        bbox_coder=dict(
            type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss',
            beta=1/9.0,
            loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.2),
        loss_iou=dict(
            type='RotatedIoU3DLoss',
            loss_weight=1.0),
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=100,
            nms_thr=0.01,
            score_thr=0.1,
            min_bbox_size=0,
            max_num=50,
            nms_type='sparse')),
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            allowed_border=0,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=100,
            nms_thr=0.01,
            score_thr=0.1,
            min_bbox_size=0,
            max_num=50,
            nms_type='sparse')))

# dataset settings
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
    classes=class_names,
    sample_groups=dict(Car=12, Pedestrian=6, Cyclist=6))

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
        type='RandomResize', 
        scale=[(640, 192), (2560, 768)], 
        keep_ratio=True),
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
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=point_cloud_range),
        ]),
    dict(type='Pack3DDetInputs', keys=['points', 'img'])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kitti_infos_train.pkl',
        data_prefix=dict(
            img='training/image_2',
            pts='training/velodyne_reduced'),
        pipeline=train_pipeline,
        modality=input_modality,
        metainfo=metainfo,
        test_mode=False))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kitti_infos_train.pkl',
        data_prefix=dict(
            img='training/image_2',
            pts='training/velodyne_reduced'),
        pipeline=test_pipeline,
        modality=input_modality,
        metainfo=metainfo,
        test_mode=True))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kitti_infos_train.pkl',
        data_prefix=dict(
            img='training/image_2',
            pts='training/velodyne_reduced'),
        pipeline=test_pipeline,
        modality=input_modality,
        metainfo=metainfo,
        test_mode=True))

val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_train.pkl',
    metric='bbox',
    backend_args=backend_args)

test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[20, 23],
        gamma=0.1)
]

# training schedule
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=5, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Default setting for scaling LR automatically
auto_scale_lr = dict(base_batch_size=16)
