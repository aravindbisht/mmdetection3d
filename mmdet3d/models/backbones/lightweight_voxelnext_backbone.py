import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmcv.ops import SubMConv3d
from mmdet3d.registry import MODELS

def channel_shuffle(x, groups):
    """Channel shuffle operation.
    
    Args:
        x (Tensor): Input tensor.
        groups (int): Number of groups to shuffle.
        
    Returns:
        Tensor: Shuffled tensor.
    """
    batch_size, channels, height, width, depth = x.size()
    channels_per_group = channels // groups
    
    # reshape
    x = x.view(batch_size, groups, channels_per_group, height, width, depth)
    
    # transpose
    x = torch.transpose(x, 1, 2).contiguous()
    
    # flatten
    x = x.view(batch_size, -1, height, width, depth)
    
    return x

@MODELS.register_module()
class LightweightVoxelNeXtBackbone(BaseModule):
    """Lightweight VoxelNeXt backbone for efficient 3D feature extraction.
    
    This backbone reduces parameters and computation while maintaining feature
    extraction capability through:
    1. Group convolutions with channel shuffling
    2. Channel reduction
    3. Sparse operations
    4. Efficient residual connections
    
    Args:
        in_channels (int): Number of input channels.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Stride of each layer.
        out_channels (list[int]): Number of output channels for each stage.
        sparse_shape (list[int]): Shape of sparse tensor.
        with_cp (bool): Use checkpoint or not.
        use_sparse_conv (bool): Use sparse convolution or not.
        groups (int): Number of groups for group convolution.
        use_subm_conv (bool): Whether to use SubMConv3d for all convolutions.
    """
    
    def __init__(self,
                 in_channels,
                 layer_nums,
                 layer_strides,
                 out_channels,
                 sparse_shape,
                 with_cp=False,
                 use_sparse_conv=True,
                 groups=4,
                 use_subm_conv=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.layer_nums = layer_nums
        self.layer_strides = layer_strides
        self.out_channels = out_channels
        self.sparse_shape = sparse_shape
        self.with_cp = with_cp
        self.use_sparse_conv = use_sparse_conv
        self.groups = groups
        self.use_subm_conv = use_subm_conv
        
        # Ensure channels are divisible by groups
        assert all(c % groups == 0 for c in out_channels), \
            'out_channels must be divisible by groups'
        
        # Build backbone layers
        self.blocks = nn.ModuleList()
        for i, layer_num in enumerate(layer_nums):
            block = nn.ModuleList()
            for j in range(layer_num):
                stride = layer_strides[i] if j == 0 else 1
                in_ch = in_channels if i == 0 and j == 0 else out_channels[i]
                out_ch = out_channels[i]
                
                # Use group convolution for efficiency
                block.append(
                    LightweightSparseBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        stride=stride,
                        sparse_shape=sparse_shape,
                        use_sparse_conv=use_sparse_conv,
                        groups=groups,
                        use_subm_conv=use_subm_conv))
            self.blocks.append(block)
    
    def forward(self, x):
        """Forward function.
        
        Args:
            x (Tensor): Input tensor with shape (N, C, H, W, D).
            
        Returns:
            list[Tensor]: List of feature maps.
        """
        outputs = []
        
        for i, block in enumerate(self.blocks):
            for j, layer in enumerate(block):
                if self.with_cp and not torch.onnx.is_in_onnx_export():
                    x = torch.utils.checkpoint.checkpoint(layer, x)
                else:
                    x = layer(x)
            outputs.append(x)
        
        return outputs

class LightweightSparseBlock(BaseModule):
    """Lightweight sparse block with group convolution.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride of the first convolution.
        sparse_shape (list[int]): Shape of sparse tensor.
        use_sparse_conv (bool): Use sparse convolution or not.
        groups (int): Number of groups for group convolution.
        use_subm_conv (bool): Whether to use SubMConv3d for all convolutions.
    """
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 sparse_shape,
                 use_sparse_conv=True,
                 groups=4,
                 use_subm_conv=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_sparse_conv = use_sparse_conv
        self.groups = groups
        self.use_subm_conv = use_subm_conv
        
        # Ensure channels are divisible by groups
        assert in_channels % groups == 0 and out_channels % groups == 0, \
            'channels must be divisible by groups'
        
        # Group convolution for efficiency
        if use_sparse_conv:
            # Use SubMConv3d with channel shuffling for group convolution
            self.conv = SubMConv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                indice_key='subm')
        else:
            self.conv = ConvModule(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=dict(type='BN3d'),
                act_cfg=dict(type='ReLU', inplace=True))
        
        # Batch normalization and activation
        self.norm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual connection if dimensions match
        self.residual = None
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
    
    def forward(self, x):
        """Forward function.
        
        Args:
            x (Tensor): Input tensor with shape (N, C, H, W, D).
            
        Returns:
            Tensor: Output tensor with shape (N, C, H, W, D).
        """
        identity = x
        
        # Apply group convolution with channel shuffling
        if self.use_sparse_conv:
            # Reshape for group convolution
            batch_size, channels, height, width, depth = x.size()
            x = x.view(batch_size, self.groups, -1, height, width, depth)
            x = torch.transpose(x, 1, 2).contiguous()
            x = x.view(batch_size, -1, height, width, depth)
            
            # Apply convolution
            out = self.conv(x)
            
            # Reshape back
            batch_size, channels, height, width, depth = out.size()
            out = out.view(batch_size, self.groups, -1, height, width, depth)
            out = torch.transpose(out, 1, 2).contiguous()
            out = out.view(batch_size, -1, height, width, depth)
        else:
            out = self.conv(x)
        
        # Apply normalization and activation
        if isinstance(out, tuple):
            out = out[0]
        out = self.norm(out)
        out = self.relu(out)
        
        # Add residual connection if applicable
        if self.residual is not None:
            out = out + self.residual(identity)
        
        return out 
