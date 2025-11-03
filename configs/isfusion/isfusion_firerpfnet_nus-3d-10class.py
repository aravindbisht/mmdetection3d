# IS-Fusion with FireRPFNet backbone for nuScenes 10-class 3D detection
# Based on the original IS-Fusion implementation but with FireRPFNet as the point cloud backbone

_base_ = ['../_base_/schedules/cyclic-20e.py', '../_base_/default_runtime.py']

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-54, -54, -5, 54, 54, 3]
img_scale = (384, 1056)

total_epochs = 20
res_factor = 1
out_size_factor = 8
voxel_shape = int((point_cloud_range[3]-point_cloud_range[0])//voxel_size[0])
bev_size = voxel_shape//out_size_factor
grid_size = [[bev_size, bev_size, 1], [bev_size//2, bev_size//2, 1]]
region_shape = [(6, 6, 1), (6, 6, 1)]
region_drop_info = [
    {0:{'max_tokens':36, 'drop_range':(0, 100000)},},
    {0:{'max_tokens':36, 'drop_range':(0, 100000)},},
]

model = dict(
    type='ISFusionDetector',
    detach=True,
    pc_range=point_cloud_range,
    voxel_size=voxel_size,
    out_size_factor=out_size_factor,

    # Image backbone (keep original)
    img_backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
        convert_weights=False,
    ),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=3),

    # Point cloud processing
    pts_voxel_layer=dict(
        point_cloud_range=point_cloud_range,
        max_num_points=-1, 
        voxel_size=voxel_size, 
        max_voxels=(-1, -1)
    ),
    pts_voxel_encoder=dict(
        type='DynamicVFE',
        in_channels=5,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
    ),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=64,
        sparse_shape=[41, voxel_shape, voxel_shape],
        base_channels=32,
        output_channels=256,
        order=('conv', 'norm', 'act'),
        encoder_channels=((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 256)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock',
    ),

    # FireRPFNet as the point cloud backbone
    pts_backbone=dict(
        type='FireRPFNet',
        in_channels=256,  # Output from SparseEncoder
        layer_channels=[128, 256, 256, 256],
        with_cbam=True,
    ),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256, 256, 256],  # Match FireRPFNet outputs
        out_channels=[256, 256, 256, 256],
        upsample_strides=[1, 2, 4, 4],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True
    ),

    # Multi-modal fusion (keep original)
    fusion_encoder=dict(
        type='ISFusionEncoder',
        num_points_in_pillar=12,
        embed_dims=256,
        bev_size=bev_size,
        num_views=6,
        region_shape=region_shape,
        grid_size=grid_size,
        region_drop_info=region_drop_info,
        instance_num=200,
    ),

    # Detection head (keep original)
    pts_bbox_head=dict(
        type='TransFusionHeadV2',
        num_proposals=200,
        auxiliary=True,
        in_channels=256 * 2,  # 2x due to concat of image and point features
        hidden_channel=128,
        num_classes=len(class_names),
        num_decoder_layers=1,
        num_heads=8,
        nms_kernel_size=3,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation='relu',
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            code_size=10,
        ),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
    ),

    # Training configuration (keep original)
    train_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25)
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[voxel_shape, voxel_shape, 40],
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range
        )
    ),
    
    # Testing configuration (keep original)
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=[voxel_shape, voxel_shape, 40],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
            nms_type=None,
            use_rotate_nms=True,
            nms_thr=0.2,
            max_num=200,
        )
    )
)

# -----------------------------------------------------------------------------
# Dataset Configuration
# -----------------------------------------------------------------------------
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

# Training pipeline
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=file_client_args),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=point_cloud_range),
    dict(
        type='ObjectNameFilter',
        classes=class_names),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

# Testing pipeline
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=file_client_args),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points', 'img'])
]

# Data configuration
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=dict(classes=class_names),
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=dict(classes=class_names),
        test_mode=True,
        box_type_3d='LiDAR'))

test_dataloader = val_dataloader

# Evaluation configuration
val_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    metric=['bbox', 'segm', 'track'],
)
test_evaluator = val_evaluator

# -----------------------------------------------------------------------------
# Training Configuration
# -----------------------------------------------------------------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        end=total_epochs,
        T_max=total_epochs,
        by_epoch=True,
        eta_min_ratio=1e-3)
]

# Training loop configuration
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Checkpoint configuration
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='nuscenes/NDS',
        rule='greater',
        max_keep_ckpts=5
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

# Visualization
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Add EarlyStoppingHook
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='nuscenes/NDS',
        patience=5,
        rule='greater',
        min_delta=0.001,
    )
]
