from mmengine.model import BaseModule
import torch.nn as nn
from mmdet3d.registry import MODELS

@MODELS.register_module()
class IdentityBackbone(BaseModule):
    """A simple identity backbone that passes through input unchanged."""
    
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.identity = nn.Identity()
        
    def forward(self, x):
        # Ensure output is always a list for compatibility with heads expecting multi-level features
        out = self.identity(x)
        if not isinstance(out, (list, tuple)):
            return [out]
        return out