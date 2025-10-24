"""SqueezeNet V3: Torchvision-compatible backbone with full pretrained support.

This implementation uses torchvision's SqueezeNet1_1 as a base and adapts it for
feature extraction in MMDetection3D. It supports loading pre-trained weights from
torchvision and provides multi-scale feature extraction capabilities.
"""

import warnings
from typing import List, Optional, Sequence, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from mmengine.model import BaseModule
from mmengine.runner import load_checkpoint

from mmdet3d.registry import MODELS


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """Ensure all layers have a channel number that's divisible by 8.
    
    This function is taken from torchvision's implementation of MobileNetV2.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@MODELS.register_module()
class SQUEEZEv3(BaseModule):
    """SqueezeNet V3: Torchvision-compatible backbone with full pretrained support.

    This implementation uses torchvision's SqueezeNet1_1 as a base and adapts it
    for feature extraction in MMDetection3D. It provides multi-scale feature
    extraction and supports loading pre-trained weights from torchvision.

    Args:
        in_channels (int): Number of input channels. Must be 3 when using
            pre-trained weights. Default: 3.
        out_indices (Sequence[int], optional): Output from which stages.
            Default: (1, 3, 6, 10).
        out_channels (Sequence[int]): Output channels for multi-scale feature maps.
            Default: (64, 128, 256, 512).
        frozen_stages (int): Stages to be frozen (stop gradient and set eval mode).
            -1 means not freezing any parameters. Default: 1 (freeze stem + stage1).
        init_cfg (Optional[dict]): Initialization config dict.
            For pretrained: dict(type='Pretrained', checkpoint='torchvision://squeezenet1_1')
        pretrained (str, optional): Deprecated. Use init_cfg instead.
    """

    # Feature map indices for multi-scale outputs
    # These correspond to: after conv1, fire3, fire6, fire9
    arch_settings = {
        'squeezenet1_1': {
            'out_indices': (1, 3, 6, 10),
            'out_channels': (64, 128, 256, 512)
        }
    }

    def __init__(self,
                 in_channels: int = 3,
                 out_indices: Optional[Sequence[int]] = None,
                 out_channels: Optional[Sequence[int]] = None,
                 frozen_stages: int = 1,
                 init_cfg: Optional[Dict] = None,
                 pretrained: Optional[str] = None) -> None:
        
        # Handle deprecated pretrained parameter
        if isinstance(pretrained, str):
            warnings.warn(
                'DeprecationWarning: pretrained is deprecated, '
                'please use "init_cfg" instead.',
                DeprecationWarning)
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        
        # Set default init_cfg if not provided
        if init_cfg is None:
            init_cfg = dict(type='Pretrained', checkpoint='torchvision://squeezenet1_1')
            
        super().__init__(init_cfg=init_cfg)
        
        if in_channels != 3:
            raise ValueError('Input channels must be 3 when using pretrained weights')
            
        self.in_channels = in_channels
        self.arch = 'squeezenet1_1'
        self.out_indices = out_indices or self.arch_settings[self.arch]['out_indices']
        self.out_channels = out_channels or self.arch_settings[self.arch]['out_channels']
        self.frozen_stages = frozen_stages
        
        # Validate out_indices and out_channels
        if len(self.out_indices) != len(self.out_channels):
            raise ValueError('Length of out_indices and out_channels must match')
        
        # Load torchvision's SqueezeNet1_1
        self.squeezenet = models.squeezenet1_1(pretrained=False)
        
        # Extract the features
        self.features = nn.ModuleList(list(self.squeezenet.features.children()))
        
        # Initialize weights
        self.init_weights()
        
        # Freeze stages if needed
        self._freeze_stages()

    def _freeze_stages(self) -> None:
        """Freeze parameters of specified stages."""
        if self.frozen_stages < 0:
            return
            
        # Stage 0: First conv + ReLU + MaxPool (indices 0-2)
        if self.frozen_stages >= 0:
            for i in range(3):
                m = self.features[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
                    
        # Stage 1: First 3 fire modules (indices 3-5)
        if self.frozen_stages >= 1:
            for i in range(3, 6):
                m = self.features[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
                    
        # Stage 2: Next 3 fire modules (indices 6-8)
        if self.frozen_stages >= 2:
            for i in range(6, 9):
                m = self.features[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
                    
        # Stage 3: Last fire module + conv2d (indices 9-11)
        if self.frozen_stages >= 3:
            for i in range(9, 12):
                if i < len(self.features):
                    m = self.features[i]
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False

    def train(self, mode: bool = True) -> 'SQUEEZEv3':
        """Convert the model into training mode while keeping normalization layer
        frozen.

        Args:
            mode (bool): Whether to set training mode (True) or evaluation
                mode (False). Default: True.

        Returns:
            nn.Module: The model itself.
        """
        super().train(mode)
        self._freeze_stages()
        
        # Set batchnorm layers to eval mode when freezing stages
        if mode and self.frozen_stages > 0:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        
        return self

    def init_weights(self) -> None:
        """Initialize weights from pretrained model."""
        if not (hasattr(self, 'init_cfg') and self.init_cfg['type'] == 'Pretrained'):
            return
        
        # Load pretrained weights from torchvision
        pretrained = models.squeezenet1_1(pretrained=True)
        
        # Load the state dict
        state_dict = self.state_dict()
        pretrained_state_dict = pretrained.state_dict()
        
        # Filter out unnecessary keys and handle size mismatches
        for name, param in pretrained_state_dict.items():
            if name in state_dict and param.shape == state_dict[name].shape:
                state_dict[name] = param
            elif name.startswith('features'):
                # Handle features submodule
                new_name = name.replace('features.', '')
                if new_name in state_dict and param.shape == state_dict[new_name].shape:
                    state_dict[new_name] = param
        
        # Load the filtered state dict
        self.load_state_dict(state_dict, strict=False)
        
        print("=" * 80)
        print("✅ SQUEEZEv3: Pretrained weights loaded successfully!")
        print(f"   Architecture: {self.arch}")
        print(f"   Frozen stages: {self.frozen_stages}")
        print(f"   Output indices: {self.out_indices}")
        print(f"   Output channels: {self.out_channels}")
        print("=" * 80)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)

        Returns:
            list[torch.Tensor]: List of feature maps from the specified out_indices.
        """
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f'Input tensor has {x.shape[1]} channels but the model expects {self.in_channels} channels.'
            )
            
        outs = []
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
                if len(outs) == len(self.out_indices):
                    break
        
        # Ensure we return the exact number of feature maps expected
        if len(outs) != len(self.out_indices):
            raise ValueError(
                f'Expected {len(self.out_indices)} feature maps but got {len(outs)}. '
                'Check if out_indices are valid for the model architecture.'
            )
                    
        return outs


@MODELS.register_module()
class SQUEEZE_PretrainedV3(SQUEEZEv3):
    """Alias for SQUEEZEv3 with explicit pretrained naming.

    This is provided for backward compatibility and clarity.
    """
    pass

