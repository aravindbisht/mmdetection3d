import torch
from mmdet3d.apis import init_model, inference_detector
from mmdet3d.structures import PointCloud

# Initialize model
config = 'configs/mvxnet/mvxnet_voxelnext.py'
checkpoint = None  # Can specify a checkpoint path here
model = init_model(config, checkpoint, device='cuda:0')

# Create demo point cloud
points = torch.rand(1000, 4) * 100  # Random points with (x,y,z,intensity)
points[:, :3] -= 50  # Center around origin

# Create PointCloud object
pc = PointCloud(
    points=points,
    points_range=[-50, -50, -5, 50, 50, 3]  # Same as config
)

# Run inference
result, data = inference_detector(model, pc)

# Visualize results
model.show_results(data, result, show=True)
