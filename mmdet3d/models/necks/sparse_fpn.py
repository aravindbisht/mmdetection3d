import torch
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmcv.ops import SparseConvTensor

@MODELS.register_module()
class SparseFPN(BaseModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Create projection for each input channel
        self.projs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=1)
            for in_ch, out_ch in zip(in_channels, out_channels)
        ])
        
    def forward(self, x):
        if isinstance(x, (list, tuple)):
            # Process each input separately
            return [proj(feat.mean(dim=2) if feat.dim() == 5 else feat)
                   for feat, proj in zip(x, self.projs)]
        elif isinstance(x, SparseConvTensor):
            # Convert sparse tensor to dense format
            dense_features = torch.zeros(
                (x.batch_size, x.features.size(1), *x.spatial_shape),
                device=x.features.device,
                dtype=x.features.dtype
            )
            for i, (idx, feat) in enumerate(zip(x.indices, x.features)):
                dense_features[idx[0], :, idx[1], idx[2], idx[3]] = feat
            return [self.projs[0](dense_features.mean(dim=2))]
        return [self.projs[0](x)]
