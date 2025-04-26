_base_ = ['./mvxnet_voxelnext_kitti_01.py']

# Override image backbone to ResNet-50 and adjust FPN inputs
model = dict(
    img_backbone=dict(
        _delete_=True,  # replace the inherited backbone instead of update
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),  # C3, C4, C5 feature maps
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    img_neck=dict(
        _delete_=True,
        type='mmdet.FPN',
        in_channels=[512, 1024, 2048],  # matching ResNet-50 C3–C5 channels
        out_channels=128,
        num_outs=3))
