from mmengine.model import BaseModule
import torch.nn as nn

class IdentityBackbone(BaseModule):
    """A simple identity backbone that passes through input unchanged."""
    
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.identity = nn.Identity()
        
    def forward(self, x):
        return self.identity(x)