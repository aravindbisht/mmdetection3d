# Optimized MVX-Net (SqueezeFPN camera branch) + FireRPFNet for KITTI 3-class
# with training optimizations for faster convergence

_base_ = ['../_base_/schedules/cosine.py', '../_base_/default_runtime.py']

# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------
voxel_size = [0.05, 0.05, 0.1]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
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

    # Image backbone and neck
    img_backbone=dict(
        type='SQUEEZE',
        in_channels=3,
        out_channels=[64, 128, 256, 512],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    img_neck=dict(
        type='SQUEEZEFPN',
        in_channels=[64, 128, 256, 512],
        out_channels=[256, 256, 256, 256],
        norm_cfg=dict(type='BN', requires_grad=False)),

    # LiDAR voxel encoder with fusion
    pts_voxel_encoder=dict(
        type='DynamicVFE',
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        fusion_layer=dict(
            type='PointFusion',
            img_channels=256,
            pts_channels=64,
            mid_channels=128,
            out_channels=128,
            img_levels=[0, 1, 2, 3],
            align_corners=False,
            activate_out=True,
            fuse_out=False)),
    
    # Middle encoder for point clouds
    pts_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=128,  # Increased from 64 to 128 to match fusion output
        output_shape=[400, 400]),
    
    # RPN Head
    rpn_head=dict(
        type='RPNHead',
        in_channels=384,  # 128*3 (FPN outputs)
        feat_channels=384,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[[0, -40.0, -1.78, 70.4, 40.0, -1.78]],
            sizes=[[3.9, 1.6, 1.56]],  # Average size for KITTI
            rotations=[0, 1.57],
            reshape_out=False),
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0)),
    
    # ROI Head
    roi_head=dict(
        type='PartA2RoIHead',
        num_classes=3,
        semantic_head=dict(
            type='PointwiseSemanticHead',
            in_channels=16,
            extra_width=0.2,
            seg_score_thr=0.3,
            loss_seg=dict(
                type='FocalLoss',
                use_sigmoid=True,
                reduction='sum',
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0)),
        bbox_roi_extractor=dict(
            type='Single3DRoIAwareExtractor',
            roi_layer=dict(
                type='RoIAwarePool3d',
                out_size=14,
                max_pts_per_voxel=128,
                mode='max')),
        bbox_head=dict(
            type='PartA2BboxHead',
            num_classes=3,
            seg_in_channels=16,
            part_in_channels=4,
            seg_conv_channels=[64, 64],
            part_conv_channels=[64, 64],
            merge_conv_channels=[128, 128],
            down_conv_channels=[128, 256],
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=2.0),
            loss_bbox_roi=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    
    # Model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=9000,
            nms_post=1024,
            max_num=1024,
            nms_thr=0.7,
            score_thr=0.1,
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.6,
                min_pos_iou=0.6,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=128,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            nms_post=300,
            max_num=300,
            nms_thr=0.7,
            score_thr=0.1,
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.1,
            nms=dict(type='nms', iou_thr=0.5),
            max_per_img=100))
)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

# Optimized training pipeline
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4, backend_args=backend_args),
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True, with_label=True),
    dict(type='RandomResize', scale=[(640, 192), (1024, 320)], keep_ratio=True),  # Slightly reduced resolution
    dict(type='GlobalRotScaleTrans', rot_range=[-0.5, 0.5], scale_ratio_range=[0.95, 1.05], translation_std=[0.1, 0.1, 0.1]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.3),  # Slightly reduced flip probability
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4, backend_args=backend_args),
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1280, 384),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='Resize', scale=0, keep_ratio=True),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        ]),
    dict(type='Pack3DDetInputs', keys=['points', 'img'])
]

# Optimized data loaders
train_dataloader = dict(
    batch_size=8,  # Increased batch size for better GPU utilization
    num_workers=4,  # Increased workers for faster data loading
    persistent_workers=True,  # Keep workers alive between epochs
    sampler=dict(type='DefaultSampler', shuffle=True),
    pin_memory=True,  # Enable pin memory for faster data transfer
    dataset=dict(
        type='RepeatDataset',
        times=1,  # Single pass through dataset
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            modality=input_modality,
            ann_file='kitti_infos_train.pkl',
            data_prefix=dict(pts='training/velodyne_reduced', img='training/image_2'),
            pipeline=train_pipeline,
            filter_empty_gt=True,  # Skip empty samples
            metainfo=metainfo,
            box_type_3d='LiDAR',
            backend_args=backend_args)))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,  # Slightly increased for validation
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        modality=input_modality,
        ann_file='kitti_infos_val.pkl',
        data_prefix=dict(pts='training/velodyne_reduced', img='training/image_2'),
        pipeline=test_pipeline,
        metainfo=metainfo,
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))

test_dataloader = val_dataloader

# -----------------------------------------------------------------------------
# Optimizer / Schedulers / Runtime
# -----------------------------------------------------------------------------
# Optimized optimizer with mixed precision training
optim_wrapper = dict(
    type='AmpOptimWrapper',  # Enable mixed precision training
    optimizer=dict(
        type='AdamW',
        lr=0.002,  # Slightly higher learning rate for larger batch size
        betas=(0.9, 0.999),  # Default betas for AdamW
        weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
    loss_scale='dynamic'  # Dynamic loss scaling for mixed precision
)

# Learning rate schedule
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,  # Warmup from 10% of initial lr
        by_epoch=False,
        begin=0,
        end=1000),  # 1000 warmup iters
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end=20,  # Total epochs
        eta_min=1e-5)  # Minimum learning rate
]

# Evaluation and visualization
val_evaluator = dict(type='KittiMetric', ann_file='data/kitti/kitti_infos_val.pkl')
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

# Training configuration
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=5,  # Original number of epochs
    val_interval=2)  # Validate every 2 epochs to save time

# Checkpoint and early stopping configuration
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='Kitti metric/pred_instances_3d/KITTI/Car_3D_AP40_moderate_strict',
        rule='greater',
        max_keep_ckpts=3))  # Keep only the best 3 checkpoints

# Early stopping hook
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='Kitti metric/pred_instances_3d/KITTI/Car_3D_AP40_moderate_strict',
        patience=5,  # Stop if no improvement for 5 epochs
        rule='greater',
        min_delta=0.001)
]

# Set the random seed for reproducibility
randomness = dict(seed=0, deterministic=False)
