auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
class_names = [
    'Pedestrian',
    'Cyclist',
    'Car',
]
custom_hooks = [
    dict(
        min_delta=0.001,
        monitor=
        'Kitti metric/pred_instances_3d/KITTI/Car_3D_AP40_moderate_strict',
        patience=5,
        rule='greater',
        type='EarlyStoppingHook'),
]
data_root = 'data/kitti/'
dataset_type = 'KittiDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        rule='greater',
        save_best=
        'Kitti metric/pred_instances_3d/KITTI/Car_3D_AP40_moderate_strict',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
input_modality = dict(use_camera=True, use_lidar=True)
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
lr = 0.003
metainfo = dict(classes=[
    'Pedestrian',
    'Cyclist',
    'Car',
])
model = dict(
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            102.9801,
            115.9465,
            122.7717,
        ],
        pad_size_divisor=32,
        std=[
            1.0,
            1.0,
            1.0,
        ],
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=-1,
            max_voxels=(
                -1,
                -1,
            ),
            point_cloud_range=[
                0,
                -40,
                -3,
                70.4,
                40,
                1,
            ],
            voxel_size=[
                0.05,
                0.05,
                0.1,
            ]),
        voxel_type='dynamic'),
    img_backbone=dict(
        conv_cfg=dict(bias=False, type='Conv2d'),
        in_channels=3,
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN'),
        out_channels=[
            64,
            128,
            256,
            512,
        ],
        type='SQUEEZE'),
    img_neck=dict(
        in_channels=[
            64,
            128,
            256,
            512,
        ],
        norm_cfg=dict(requires_grad=False, type='BN'),
        out_channels=[
            256,
            256,
            256,
            256,
        ],
        type='SQUEEZEFPN'),
    pts_middle_encoder=dict(
        in_channels=128, output_shape=[
            400,
            400,
        ], type='PointPillarsScatter'),
    pts_voxel_encoder=dict(
        feat_channels=[
            64,
            64,
        ],
        fusion_layer=dict(
            activate_out=True,
            align_corners=False,
            fuse_out=False,
            img_channels=256,
            img_levels=[
                0,
                1,
                2,
                3,
            ],
            mid_channels=128,
            out_channels=128,
            pts_channels=64,
            type='PointFusion'),
        in_channels=4,
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN1d'),
        point_cloud_range=[
            0,
            -40,
            -3,
            70.4,
            40,
            1,
        ],
        type='DynamicVFE',
        voxel_size=[
            0.05,
            0.05,
            0.1,
        ],
        with_cluster_center=True,
        with_distance=False,
        with_voxel_center=True),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            down_conv_channels=[
                128,
                256,
            ],
            loss_bbox=dict(beta=1.0, loss_weight=2.0, type='SmoothL1Loss'),
            loss_bbox_roi=dict(beta=1.0, loss_weight=1.0, type='SmoothL1Loss'),
            loss_cls=dict(
                alpha=0.25,
                gamma=2.0,
                loss_weight=1.0,
                type='FocalLoss',
                use_sigmoid=True),
            merge_conv_channels=[
                128,
                128,
            ],
            num_classes=3,
            part_conv_channels=[
                64,
                64,
            ],
            part_in_channels=4,
            seg_conv_channels=[
                64,
                64,
            ],
            seg_in_channels=16,
            type='PartA2BboxHead'),
        bbox_roi_extractor=dict(
            roi_layer=dict(
                max_pts_per_voxel=128,
                mode='max',
                out_size=14,
                type='RoIAwarePool3d'),
            type='Single3DRoIAwareExtractor'),
        num_classes=3,
        semantic_head=dict(
            extra_width=0.2,
            in_channels=16,
            loss_seg=dict(
                alpha=0.25,
                gamma=2.0,
                loss_weight=1.0,
                reduction='sum',
                type='FocalLoss',
                use_sigmoid=True),
            seg_score_thr=0.3,
            type='PointwiseSemanticHead'),
        type='PartA2RoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ranges=[
                [
                    0,
                    -40.0,
                    -1.78,
                    70.4,
                    40.0,
                    -1.78,
                ],
            ],
            reshape_out=False,
            rotations=[
                0,
                1.57,
            ],
            sizes=[
                [
                    3.9,
                    1.6,
                    1.56,
                ],
            ],
            type='Anchor3DRangeGenerator'),
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        feat_channels=384,
        in_channels=384,
        loss_bbox=dict(
            beta=0.1111111111111111, loss_weight=2.0, type='SmoothL1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=100, nms=dict(iou_thr=0.5, type='nms'), score_thr=0.1),
        rpn=dict(
            max_num=300,
            min_bbox_size=0,
            nms_post=300,
            nms_pre=1000,
            nms_thr=0.7,
            score_thr=0.1)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                iou_calculator=dict(coordinate='lidar', type='BboxOverlaps3D'),
                min_pos_iou=0.6,
                neg_iou_thr=0.6,
                pos_iou_thr=0.6,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=128,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn=dict(
            allowed_border=0,
            assigner=dict(
                ignore_iof_thr=-1,
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                min_pos_iou=0.45,
                neg_iou_thr=0.45,
                pos_iou_thr=0.6,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1),
        rpn_proposal=dict(
            max_num=1024,
            min_bbox_size=0,
            nms_post=1024,
            nms_pre=9000,
            nms_thr=0.7,
            score_thr=0.1)),
    type='DynamicMVXFasterRCNN')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.002, type='AdamW', weight_decay=0.01),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=1000, start_factor=0.1, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=20,
        eta_min=1e-05,
        type='CosineAnnealingLR'),
]
point_cloud_range = [
    0,
    -40,
    -3,
    70.4,
    40,
    1,
]
randomness = dict(deterministic=False, seed=0)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(
            img='training/image_2', pts='training/velodyne_reduced'),
        data_root='data/kitti/',
        metainfo=dict(classes=[
            'Pedestrian',
            'Cyclist',
            'Car',
        ]),
        modality=dict(use_camera=True, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                flip=False,
                img_scale=(
                    1280,
                    384,
                ),
                pts_scale_ratio=1,
                transforms=[
                    dict(keep_ratio=True, scale=0, type='Resize'),
                    dict(
                        rot_range=[
                            0,
                            0,
                        ],
                        scale_ratio_range=[
                            1.0,
                            1.0,
                        ],
                        translation_std=[
                            0,
                            0,
                            0,
                        ],
                        type='GlobalRotScaleTrans'),
                    dict(type='RandomFlip3D'),
                    dict(
                        point_cloud_range=[
                            0,
                            -40,
                            -3,
                            70.4,
                            40,
                            1,
                        ],
                        type='PointsRangeFilter'),
                ],
                type='MultiScaleFlipAug3D'),
            dict(keys=[
                'points',
                'img',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='KittiDataset'),
    num_workers=2,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/kitti/kitti_infos_val.pkl', type='KittiMetric')
test_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        flip=False,
        img_scale=(
            1280,
            384,
        ),
        pts_scale_ratio=1,
        transforms=[
            dict(keep_ratio=True, scale=0, type='Resize'),
            dict(
                rot_range=[
                    0,
                    0,
                ],
                scale_ratio_range=[
                    1.0,
                    1.0,
                ],
                translation_std=[
                    0,
                    0,
                    0,
                ],
                type='GlobalRotScaleTrans'),
            dict(type='RandomFlip3D'),
            dict(
                point_cloud_range=[
                    0,
                    -40,
                    -3,
                    70.4,
                    40,
                    1,
                ],
                type='PointsRangeFilter'),
        ],
        type='MultiScaleFlipAug3D'),
    dict(keys=[
        'points',
        'img',
    ], type='Pack3DDetInputs'),
]
train_cfg = dict(max_epochs=5, type='EpochBasedTrainLoop', val_interval=2)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        dataset=dict(
            ann_file='kitti_infos_train.pkl',
            backend_args=None,
            box_type_3d='LiDAR',
            data_prefix=dict(
                img='training/image_2', pts='training/velodyne_reduced'),
            data_root='data/kitti/',
            filter_empty_gt=True,
            metainfo=dict(classes=[
                'Pedestrian',
                'Cyclist',
                'Car',
            ]),
            modality=dict(use_camera=True, use_lidar=True),
            pipeline=[
                dict(
                    backend_args=None,
                    coord_type='LIDAR',
                    load_dim=4,
                    type='LoadPointsFromFile',
                    use_dim=4),
                dict(backend_args=None, type='LoadImageFromFile'),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox=True,
                    with_bbox_3d=True,
                    with_label=True,
                    with_label_3d=True),
                dict(
                    keep_ratio=True,
                    scale=[
                        (
                            640,
                            192,
                        ),
                        (
                            1024,
                            320,
                        ),
                    ],
                    type='RandomResize'),
                dict(
                    rot_range=[
                        -0.5,
                        0.5,
                    ],
                    scale_ratio_range=[
                        0.95,
                        1.05,
                    ],
                    translation_std=[
                        0.1,
                        0.1,
                        0.1,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(flip_ratio_bev_horizontal=0.3, type='RandomFlip3D'),
                dict(
                    point_cloud_range=[
                        0,
                        -40,
                        -3,
                        70.4,
                        40,
                        1,
                    ],
                    type='PointsRangeFilter'),
                dict(
                    point_cloud_range=[
                        0,
                        -40,
                        -3,
                        70.4,
                        40,
                        1,
                    ],
                    type='ObjectRangeFilter'),
                dict(type='PointShuffle'),
                dict(
                    keys=[
                        'points',
                        'img',
                        'gt_bboxes_3d',
                        'gt_labels_3d',
                        'gt_bboxes',
                        'gt_labels',
                    ],
                    type='Pack3DDetInputs'),
            ],
            type='KittiDataset'),
        times=1,
        type='RepeatDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_bbox_3d=True,
        with_label=True,
        with_label_3d=True),
    dict(
        keep_ratio=True,
        scale=[
            (
                640,
                192,
            ),
            (
                1024,
                320,
            ),
        ],
        type='RandomResize'),
    dict(
        rot_range=[
            -0.5,
            0.5,
        ],
        scale_ratio_range=[
            0.95,
            1.05,
        ],
        translation_std=[
            0.1,
            0.1,
            0.1,
        ],
        type='GlobalRotScaleTrans'),
    dict(flip_ratio_bev_horizontal=0.3, type='RandomFlip3D'),
    dict(
        point_cloud_range=[
            0,
            -40,
            -3,
            70.4,
            40,
            1,
        ],
        type='PointsRangeFilter'),
    dict(
        point_cloud_range=[
            0,
            -40,
            -3,
            70.4,
            40,
            1,
        ],
        type='ObjectRangeFilter'),
    dict(type='PointShuffle'),
    dict(
        keys=[
            'points',
            'img',
            'gt_bboxes_3d',
            'gt_labels_3d',
            'gt_bboxes',
            'gt_labels',
        ],
        type='Pack3DDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='kitti_infos_val.pkl',
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=dict(
            img='training/image_2', pts='training/velodyne_reduced'),
        data_root='data/kitti/',
        metainfo=dict(classes=[
            'Pedestrian',
            'Cyclist',
            'Car',
        ]),
        modality=dict(use_camera=True, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=4,
                type='LoadPointsFromFile',
                use_dim=4),
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                flip=False,
                img_scale=(
                    1280,
                    384,
                ),
                pts_scale_ratio=1,
                transforms=[
                    dict(keep_ratio=True, scale=0, type='Resize'),
                    dict(
                        rot_range=[
                            0,
                            0,
                        ],
                        scale_ratio_range=[
                            1.0,
                            1.0,
                        ],
                        translation_std=[
                            0,
                            0,
                            0,
                        ],
                        type='GlobalRotScaleTrans'),
                    dict(type='RandomFlip3D'),
                    dict(
                        point_cloud_range=[
                            0,
                            -40,
                            -3,
                            70.4,
                            40,
                            1,
                        ],
                        type='PointsRangeFilter'),
                ],
                type='MultiScaleFlipAug3D'),
            dict(keys=[
                'points',
                'img',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='KittiDataset'),
    num_workers=2,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/kitti/kitti_infos_val.pkl', type='KittiMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
voxel_size = [
    0.05,
    0.05,
    0.1,
]
work_dir = 'work_dirs_exp/mvxnet_sqeezefpn_fire_rpfnet_kitti-3d-3class-optimized'
