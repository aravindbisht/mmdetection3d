import torch
from mmdet3d.registry import MODELS
from mmdet3d.models.middle_encoders import SparseEncoder
from mmcv.ops import SparseConvTensor

@MODELS.register_module()
class SparseEncoderSparseOutput(SparseEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparse_tensor = True
        
    def forward(self, voxel_features, coors, batch_size):
        # Check for empty inputs first
        if len(voxel_features) == 0 or len(coors) == 0:
            return self._create_empty_tensor(voxel_features, coors, batch_size)
            
        # Safely check coordinates (handle 1D case)
        if coors.dim() == 1:
            coors = coors.unsqueeze(0)
        
        # Validate coordinates
        try:
            valid_mask = (coors.min(dim=1).values > 0).all(dim=1)
            if not valid_mask.any():
                return self._create_empty_tensor(voxel_features, coors, batch_size)
        except IndexError:
            return self._create_empty_tensor(voxel_features, coors, batch_size)
            
        # Process through parent encoder
        try:
            features = super().forward(voxel_features[valid_mask], coors[valid_mask], batch_size)
            if features.numel() == 0:
                return self._create_empty_tensor(voxel_features, coors, batch_size)
                
            return SparseConvTensor(
                features=features,
                indices=coors[valid_mask].int(),
                spatial_shape=torch.Size(self.sparse_shape),
                batch_size=batch_size
            )
        except RuntimeError:
            return self._create_empty_tensor(voxel_features, coors, batch_size)

    def _create_empty_tensor(self, voxel_features, coors, batch_size):
        return SparseConvTensor(
            features=torch.zeros((0, self.output_channels), device=voxel_features.device),
            indices=torch.zeros((0, 4), device=coors.device, dtype=torch.int32),
            spatial_shape=torch.Size(self.sparse_shape),
            batch_size=batch_size
        )
