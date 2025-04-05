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
        projection_type='spherical',  # 'spherical' or 'cartesian'
        fusion_type='deep',           # 'early', 'late', or 'deep'
        use_attention=True,           # Whether to use attention
        norm_cfg=dict(type='BN', requires_grad=True)),
    
    pts_neck=dict(
        type='PseudoLidarImageFPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=4)
    )
