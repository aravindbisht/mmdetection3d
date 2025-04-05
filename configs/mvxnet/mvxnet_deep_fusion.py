_base_ = ['./mvxnet_true_pseudolidar.py']

model = dict(
    pts_backbone=dict(
        fusion_type='deep',  # Multi-level fusion
        use_attention=True  # With attention
    )
)

# Unique work directory
work_dir = './work_dirs/mvxnet_deep_fusion'
