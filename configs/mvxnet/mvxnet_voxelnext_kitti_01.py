_base_ = ['../_base_/schedules/cosine.py', '../_base_/default_runtime.py']

# model settings
voxel_size = [0.05, 0.05, 0.1]
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

# Calculate sparse shape based on point cloud range and voxel size
sparse_shape = [
    int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
    int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
    int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2])
]

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
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        pad_size_divisor=32,
        pad_value=0,
        bgr_to_rgb=True),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=18,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_outs=3),
    pts_voxel_encoder=dict(
        type='DynamicVFE',
        in_channels=4,
        feat_channels=[32],
        with_distance=True,
        with_cluster_center=True,
        with_voxel_center=True,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)),
    pts_middle_encoder=dict(
        type='SimpleVoxelScatter',
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range),
    pts_backbone=dict(
        type='LightweightVoxelNeXtBackbone',
        in_channels=32,
        layer_nums=[2, 3, 3],
        layer_strides=[2, 2, 2],
        out_channels=[64, 64, 64],
        sparse_shape=sparse_shape,
        with_cp=True,
        use_sparse_conv=False,
        groups=4,
        use_subm_conv=True),
    pts_neck=dict(type='BEVPoolNeck', pool_type='max'),
    pts_fusion_layer=dict(
        type='LightweightAttentionFusion',
        img_channels=64,
        pts_channels=64,
        mid_channels=64,
        out_channels=64,
        num_heads=2,
        dropout=0.1,
        use_sparse_attention=True),
    pts_bbox_head=dict(
        type='VoxelNeXtHead',
        in_channels=64,
        feat_channels=64,
        num_classes=3,
        with_velocity=True),
    train_cfg=dict(
        pts=dict(
            max_objs=300,
            dense_reg=1,
            gaussian_overlap=0.1,
            min_radius=2,
            out_size_factor=8,
            code_weights=[1.0]*10,
            grid_size=[1408, 1600, 40],
            point_cloud_range=[0, -40, -3, 70.4, 40, 1],
            voxel_size=[0.05, 0.05, 0.1])),
    test_cfg=dict(
        pts=dict(
            nms_type='rotate',
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_thr=0.25,
            pre_max_size=1000,
            post_max_size=83,
            post_center_limit_range=[0, -40, -10, 70.4, 40, 10],
            score_threshold=0.05,
            max_pool_nms=False,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            min_bbox_size=0,
            max_per_img=50)))

# dataset settings
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

# Enable automatic mixed precision training to further cut memory usage
fp16 = dict(loss_scale='dynamic')

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4, backend_args=backend_args),
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PadGTBoxVelocity', keys=['gt_bboxes_3d']),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='GlobalRotScaleTrans', rot_range=[-0.78539816, 0.78539816], scale_ratio_range=[0.95, 1.05]),
    dict(type='LightweightPointAugmentation', 
         drop_ratio=0.1,
         jitter_std=0.01,
         rot_range=[-0.78539816, 0.78539816],
         sample_ratio=0.9,
         prob=0.5,
         keep_original_format=True),
    dict(type='SparseImageAugmentation',
         drop_ratio=0.05,
         contrast_range=[0.8, 1.2],
         color_jitter=[0.0, 0.1],
         prob=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
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
    batch_size=1,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=1,
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

val_evaluator = dict(
    type='KittiMetric', ann_file='data/kitti/kitti_infos_val.pkl')
test_evaluator = val_evaluator

# training schedule
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=5, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Default setting for scaling LR automatically
auto_scale_lr = dict(base_batch_size=16)
