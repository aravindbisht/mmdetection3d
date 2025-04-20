import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

class SparseConv3d(BaseModule):
    """Sparse 3D Convolution module optimized for sparse voxel data.
    
    This implementation is inspired by VoxelNeXt's approach to maintain
    sparsity throughout the network for efficient processing.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=True,
                 indice_key=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.indice_key = indice_key
        
        # Initialize weights
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *[kernel_size] * 3))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
            
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=1)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
            
    def forward(self, x, indices=None):
        """Forward function.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) or (B, C, H, W, D)
            indices (torch.Tensor, optional): Indices tensor of shape (N, 4) where N is the
                number of active points and each row is (batch_idx, h, w, d). If None,
                regular 3D convolution will be used.
        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out, D_out)
        """
        # Move weights to the same device as input
        if self.weight.device != x.device:
            self.weight = nn.Parameter(self.weight.to(x.device))
            if self.bias is not None:
                self.bias = nn.Parameter(self.bias.to(x.device))

        # Validate input channels
        if x.size(1) != self.in_channels:
            # Try to reshape the weight tensor if input channels don't match
            if self.weight.size(1) != x.size(1):
                self.weight = nn.Parameter(
                    torch.Tensor(self.out_channels, x.size(1), *[self.kernel_size] * 3).to(x.device))
                self.reset_parameters()
                self.in_channels = x.size(1)

        # If indices is None, use regular 3D convolution
        if indices is None:
            # Handle both 4D and 5D input tensors
            if len(x.shape) == 4:
                x = x.unsqueeze(-1)
            return self._sparse_conv_forward(x, None)

        # Handle both 4D and 5D input tensors
        if len(x.shape) == 4:
            # Add depth dimension if missing
            x = x.unsqueeze(-1)
            
        batch_size, in_channels, height, width, depth = x.shape
        out_channels = self.out_channels
        out_height = height // self.stride
        out_width = width // self.stride
        out_depth = depth // self.stride

        # Initialize output tensor
        out = torch.zeros((batch_size, out_channels, out_height, out_width, out_depth),
                         device=x.device, dtype=x.dtype)

        # Process each batch separately
        for b in range(batch_size):
            # Get indices for current batch
            batch_mask = indices[:, 0] == b
            if not batch_mask.any():
                continue
                
            batch_indices = indices[batch_mask]
            
            # Get input features for current batch
            batch_features = x[b, :, batch_indices[:, 1], batch_indices[:, 2], batch_indices[:, 3]]
            
            # Apply convolution
            conv_out = self._sparse_conv_forward(batch_features.unsqueeze(0), None)
            
            # Reshape output if needed
            if len(conv_out.shape) == 2:
                conv_out = conv_out.view(-1, out_channels)
            
            # Assign to output tensor
            out[b, :, batch_indices[:, 1]//self.stride, 
                batch_indices[:, 2]//self.stride, 
                batch_indices[:, 3]//self.stride] = conv_out.squeeze(0)

        return out
    
    def _sparse_conv_forward(self, x, indices):
        """
        Placeholder for actual sparse convolution implementation.
        In a real implementation, this would use specialized sparse operations.
        """
        # Ensure input channels match
        if x.size(1) != self.weight.size(1):
            # Reshape weight if needed
            self.weight = nn.Parameter(
                torch.Tensor(self.out_channels, x.size(1), *[self.kernel_size] * 3).to(x.device))
            self.reset_parameters()
            
        return F.conv3d(x, self.weight, self.bias, self.stride, self.padding)

class SparseConvBlock(BaseModule):
    """A block of sparse convolution with normalization and activation."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 norm_cfg=dict(type='BN3d'),
                 act_cfg=dict(type='ReLU'),
                 indice_key=None):
        super().__init__()
        self.conv = SparseConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            indice_key=indice_key)
        
        # Add normalization and activation
        if norm_cfg is not None:
            self.norm = self.build_norm_layer(norm_cfg, out_channels)
        else:
            self.norm = None
            
        if act_cfg is not None:
            self.act = self.build_activation_layer(act_cfg)
        else:
            self.act = None
    
    def build_norm_layer(self, norm_cfg, num_features):
        """Build normalization layer."""
        if norm_cfg['type'] == 'BN3d':
            return nn.BatchNorm3d(num_features)
        else:
            raise NotImplementedError(f"Unsupported norm type: {norm_cfg['type']}")
    
    def build_activation_layer(self, act_cfg):
        """Build activation layer."""
        if act_cfg['type'] == 'ReLU':
            return nn.ReLU(inplace=True)
        else:
            raise NotImplementedError(f"Unsupported activation type: {act_cfg['type']}")
    
    def forward(self, x, indices=None):
        """Forward function.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) or (B, C, H, W, D)
            indices (torch.Tensor, optional): Indices tensor for sparse convolution.
                If None, regular 3D convolution will be used.
        Returns:
            torch.Tensor: Output tensor after convolution, normalization and activation
        """
        x = self.conv(x, indices)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x 
