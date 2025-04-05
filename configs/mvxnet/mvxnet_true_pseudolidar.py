_base_ = ['../_base_/schedules/cosine.py', '../_base_/default_runtime.py']

# Model settings
model = dict(
    type='DynamicMVXFasterRCNN',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='dynamic',
        voxel_layer=dict(
            max_num_points=5,
            point_cloud_range=[-50, -50, -5, 50, 50, 3],
            voxel_size=[0.1, 0.1, 0.2],
            max_voxels=(120000, 160000)),
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
    
    # Pseudo-LiDAR Image components
    pts_backbone=dict(
        type='PseudoLidarImageBackbone',
        in_channels=3,
        base_channels=64,
        depth=18,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(0, 1, 2, 3),
        projection_type='spherical',
        fusion_type='deep',
        use_attention=True,
        norm_cfg=dict(type='BN', requires_grad=True)),
    
    pts_neck=dict(
        type='PseudoLidarImageFPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=4),
    
    # 3D Detection Head with Losses
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
    
    # Training/Testing Settings
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

# Dataset and Pipeline
point_cloud_range = [-50, -50, -5, 50, 50, 3]
class_names = ['Pedestrian', 'Cyclist', 'Car']

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=dict(
        data_root='data/kitti/',
        info_path='data/kitti/kitti_dbinfos_train.pkl',
        rate=1.0,
        prepare=dict(
            filter_by_difficulty=[-1],
            filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
        classes=class_names,
        sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10))),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='Pack3DDetInputs', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadImageFromFile'),
    dict(type='Pack3DDetInputs', keys=['points', 'img'])
]

# Data Config
data = dict(
    train=dict(
        pipeline=train_pipeline,
        dataset=dict(
            ann_file='kitti_infos_train.pkl',
            data_prefix=dict(pts='training/velodyne', img='training/image_2'))),
    val=dict(
        pipeline=test_pipeline,
        ann_file='kitti_infos_val.pkl',
        data_prefix=dict(pts='training/velodyne', img='training/image_2')),
    test=dict(
        pipeline=test_pipeline,
        ann_file='kitti_infos_test.pkl',
        data_prefix=dict(pts='testing/velodyne', img='testing/image_2')))

# Evaluation
evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    metric=['bbox', 'depth', 'segm'],  # Multiple metrics
    save_best='Car_3D_moderate',
    rule='greater',
    additional_metrics=dict(
        compute_metrics=['precision', 'recall', 'f1_score', 'iou'],
        difficulty_levels=[0, 1, 2],  # Easy, Moderate, Hard
        class_specific=True,
        performance_stats=dict(
            track_latency=True,
            track_memory=True,
            track_fps=True,
            log_interval=50
        )
    ))

# Custom Metrics Hook
custom_hooks = [
    dict(
        type='KITTIMetricHook',
        eval_metrics=['bbox', 'depth', 'segm'],
        eval_class=['Car', 'Pedestrian', 'Cyclist'],
        eval_level=[0, 1, 2],
        priority='HIGH'
    ),
    dict(
        type='PerformanceBenchmarkHook',
        latency_window_size=100,
        memory_interval=10,
        log_level='INFO'
    ),
    dict(
        type='ModelComplexityHook',
        input_shape=dict(
            points=[120000, 4],  # Max points in point cloud
            img=[3, 384, 1280]   # Typical image dimensions
        ),
        flops_unit='GFLOPs',
        params_unit='M',
        compute_per_layer=True,
        show_table=True,
        show_arch=True
    )
]

# FLOPs/Params will be logged during first forward pass

# Training Schedule
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0/3,
    min_lr_ratio=1e-3)
runner = dict(type='EpochBasedRunner', max_epochs=24)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# Pretrained weights (optional)
load_from = 'https://download.openmmlab.com/mmdetection3d/pretrain_models/mvx_faster_rcnn_detectron2-caffe_20e_coco-pretrain_gt-sample_kitti-3-class_moderate-79.3_20200207-a4a6a3c7.pth'
