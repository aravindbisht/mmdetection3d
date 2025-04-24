from typing import Sequence, Tuple, List, Optional

import torch
from torch import Tensor, nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class SimpleVoxelScatter(nn.Module):
    """Convert sparse (voxel_features, coors) to a dense 5-D tensor.

    This minimal middle-encoder is meant to bridge DynamicVFE and a 3-D
    backbone that expects a dense tensor with shape (B, C, Z, Y, X).

    It performs **no** additional convolutions; it only scatters each voxel
    feature to its grid location.  Empty voxels are filled with zeros.

    Args:
        voxel_size (Sequence[float]): Voxel size (vx, vy, vz).
        point_cloud_range (Sequence[float]): Point cloud range in the form
            [x_min, y_min, z_min, x_max, y_max, z_max].
        downsample_stride (Sequence[int]): Downsample stride (ds_z, ds_y, ds_x).
    """

    def __init__(self,
                 voxel_size: Sequence[float] = (0.05, 0.05, 0.1),
                 point_cloud_range: Sequence[float] = (0, -40, -3, 70.4, 40, 1),
                 downsample_stride: Sequence[int] = (2, 4, 4)):
        super().__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.downsample_stride = downsample_stride

        # Pre-compute grid dimensions (Z, Y, X)
        x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
        vx, vy, vz = voxel_size
        ds_z, ds_y, ds_x = downsample_stride
        self.grid_size = (
            int(round((z_max - z_min) / vz) // ds_z),
            int(round((y_max - y_min) / vy) // ds_y),
            int(round((x_max - x_min) / vx) // ds_x),
        )  # (Z, Y, X)

    def forward(self,
                voxel_features: Tensor,
                coors: Tensor,
                batch_size: int) -> Tensor:
        """Scatter to dense.

        Args:
            voxel_features (Tensor): Shape (N, C).
            coors (Tensor): Int tensor (N, 4) in (batch, z, y, x) order.
            batch_size (int): Batch size.

        Returns:
            Tensor: Dense tensor with shape (B, C, Z, Y, X).
        """
        device = voxel_features.device
        N, C = voxel_features.shape
        Z, Y, X = self.grid_size
        output = voxel_features.new_zeros((batch_size, C, Z, Y, X))

        # Flatten indices for fast scatter
        bs_idx = coors[:, 0].long()
        ds_z, ds_y, ds_x = self.downsample_stride
        z_idx = (coors[:, 1] // ds_z).long()
        y_idx = (coors[:, 2] // ds_y).long()
        x_idx = (coors[:, 3] // ds_x).long()

        output[bs_idx, :, z_idx, y_idx, x_idx] = voxel_features
        return output
