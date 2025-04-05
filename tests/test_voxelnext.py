import torch
import pytest
from mmdet3d.models.backbones import VoxelNeXt
from mmdet3d.structures import PointCloud

@pytest.mark.parametrize('in_channels', [4, 5])
def test_voxelnext_backbone(in_channels):
    """Test VoxelNeXt backbone with different input channels."""
    # Create backbone
    backbone = VoxelNeXt(
        in_channels=in_channels,
        base_channels=64,
        out_indices=(0, 1, 2)
    )
    
    # Create test input
    pc = PointCloud(
        points=torch.rand(100, in_channels),
        points_range=[-50, -50, -5, 50, 50, 3]
    )
    voxels, num_points, coors = pc.to_voxel(
        voxel_size=[0.1, 0.1, 0.2],
        max_num_points=5
    )
    
    # Test forward
    input_tensor = spconv.SparseConvTensor(
        features=voxels,
        indices=coors,
        spatial_shape=[41, 1600, 1408],
        batch_size=1
    )
    outs = backbone(input_tensor)
    
    # Verify output
    assert len(outs) == 3
    for out in outs:
        assert isinstance(out, spconv.SparseConvTensor)
        assert out.features.shape[1] == 128  # Last channel size

if __name__ == '__main__':
    test_voxelnext_backbone(4)
