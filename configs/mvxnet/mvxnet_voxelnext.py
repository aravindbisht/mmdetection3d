_base_ = ['../_base_/schedules/cosine.py', '../_base_/default_runtime.py']

# Model settings optimized for VoxelNeXt
model = dict(
    type='DynamicMVXFasterRCNN',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='dynamic',
        voxel_layer=dict(
            max_num_points=10,  # Increased for better feature capture
            point_cloud_range=[-50, -50, -5, 50, 50, 3],
            voxel_size=[0.1, 0.1, 0.2],
            max_voxels=(160000, 200000)),  # Increased capacity
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    
    # Image backbone (unchanged)
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    
    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=4,
        feat_channels=[64, 128],
        with_distance=False,
        voxel_size=[0.1, 0.1, 0.2],
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=[-50, -50, -5, 50, 50, 3],
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)),
    
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=128,
        sparse_shape=[41, 1024, 1024],
        output_channels=128,
        order=('conv', 'norm', 'act')),
    
    pts_backbone=dict(
        type='VoxelNeXt',
        in_channels=128,  # Matches middle encoder output
        base_channels=64,
        out_indices=(0, 1, 2),
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        with_cp=True),  # Enable checkpointing
    
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256, 512],
        out_channels=256,
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    
    # Detection head (unchanged)
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-50, -50, -1.8, 50, 50, -1.8]],
            sizes=[[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6, 1.73]],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', 
            beta=1.0/9.0, 
            loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', 
            use_sigmoid=False,
            loss_weight=0.2)),
    
    # Training and testing settings
    train_cfg=dict(
        pts=dict(
            assigner=[
                dict(
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1),
                dict(
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1)
            ],
            allowed_border=0,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(
            nms_pre=100,
            min_bbox_size=0,
            score_thr=0.1,
            nms=dict(type='nms', iou_thr=0.5),
            max_num=50)))

# Training schedule
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=24,
    val_interval=1,
    dynamic_intervals=[(20, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

# Learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0/3,
        begin=0,
        end=1000,
        by_epoch=False),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        end=24,
        T_max=24,
        eta_min=1e-5,
        by_epoch=True,
        convert_to_iter_based=True)
]

# Dataset and evaluation
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type='KittiDataset',
            data_root='data/kitti/',
            ann_file='kitti_infos_train.pkl',
            pipeline=train_pipeline)))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='KittiDataset',
        data_root='data/kitti/',
        ann_file='kitti_infos_val.pkl',
        pipeline=test_pipeline,
        test_mode=True))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='KittiMetric',
    ann_file='data/kitti/kitti_infos_val.pkl',
    metric='bbox')

test_evaluator = val_evaluator

# Visualization
vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

# Logging
log_processor = dict(
    type='LogProcessor',
    window_size=50,
    by_epoch=True,
    custom_cfg=[dict(data_src='loss', method='mean', window_size='global')])

# Hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook', interval=5))

# Environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

# Distributed training
find_unused_parameters = True

# Set seed and deterministic
deterministic = True
seed = 0
