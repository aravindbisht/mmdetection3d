from typing import List, Tuple
import torch
from torch import Tensor, nn
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class BEVPoolNeck(BaseModule):
    """Collapse Z dimension of 5-D voxel features into a 2-D BEV map.

    Accepts either a single tensor (B, C, Z, Y, X) or a list / tuple of such
    tensors.  Output is a tuple of 4-D tensors (B, C, Y, X) ready for 2-D heads
    such as Anchor3DHead or CenterHead.

    Args:
        pool_type (str): 'max' or 'mean'. Defaults to 'max'.
    """

    def __init__(self, pool_type: str = 'max', init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert pool_type in ('max', 'mean')
        self.pool_type = pool_type

    def _collapse(self, x: Tensor) -> Tensor:
        if x.dim() == 5:
            if self.pool_type == 'max':
                x = x.max(dim=2)[0]
            else:
                x = x.mean(dim=2)
        return x

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            return tuple(self._collapse(t) for t in x)
        return self._collapse(x)
