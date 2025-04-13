_base_ = [
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/models/voxelnext_multimodal.py',  # Using our multimodal config
    '../_base_/schedules/cyclic-20e.py', 
    '../_base_/default_runtime.py'
]

# KITTI-specific parameters
voxel_size = [0.05, 0.05, 0.1]  # Smaller voxels for KITTI's higher density
point_cloud_range = [0, -40, -3, 70.4, 40, 1]  # KITTI's standard range

# KITTI class names (3-class)
class_names = ['Pedestrian', 'Cyclist', 'Car']

# Image backbone config (already in voxelnext_multimodal.py)
# Training schedule adjustments for KITTI
train_cfg = dict(
    val_interval=2)  # Validate more frequently on smaller dataset

test_cfg = dict(
    pts=dict(
        # Adjust post-processing for KITTI
        post_center_limit_range=[0, -40, -3, 70.4, 40, 1],
        max_per_img=100,  # Fewer objects expected in KITTI scenes
        score_threshold=0.1,
        nms_thr=0.2
    ))

# Data config
data = dict(
    samples_per_gpu=2,  # Reduced batch size for KITTI
    workers_per_gpu=2,
    train=dict(dataset=dict(class_names=class_names)),
    val=dict(class_names=class_names),
    test=dict(class_names=class_names))
