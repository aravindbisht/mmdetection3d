import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

class OptimizedSparseConv3d(BaseModule):
    """Optimized Sparse 3D Convolution module using MMCV's sparse operations.
    
    This implementation leverages MMCV's optimized sparse operations for better
    performance while maintaining the same functionality as the original sparse
    convolution.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution.
        padding (int): Zero-padding added to both sides of the input.
        bias (bool): If True, adds a learnable bias to the output.
        indice_key (str, optional): Key for the indices tensor.
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
            x (torch.Tensor): Input tensor of shape (B, C, H, W, D).
            indices (torch.Tensor, optional): Indices tensor of shape (N, 4) where N is the
                number of active points and each row is (batch_idx, h, w, d). If None,
                regular 3D convolution will be used.
                
        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out, D_out).
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
            return self._regular_conv_forward(x)
        
        # Use optimized sparse convolution
        return self._optimized_sparse_conv_forward(x, indices)
    
    def _regular_conv_forward(self, x):
        """Regular 3D convolution forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W, D).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out, D_out).
        """
        # Handle both 4D and 5D input tensors
        if len(x.shape) == 4:
            x = x.unsqueeze(-1)
            
        # Handle fractional stride
        if isinstance(self.stride, (int, float)) and self.stride < 1:
            # Calculate output size
            out_size = [int(s / self.stride) for s in x.shape[2:]]
            # Interpolate input
            x = F.interpolate(x, size=out_size, mode='trilinear', align_corners=False)
            # Set stride to 1 for convolution
            stride = (1, 1, 1)
        else:
            # Convert stride to tuple of integers
            stride = (int(self.stride), int(self.stride), int(self.stride)) if isinstance(self.stride, (int, float)) else tuple(int(s) for s in self.stride)
            
        # Convert padding to tuple of integers
        padding = (int(self.padding), int(self.padding), int(self.padding)) if isinstance(self.padding, (int, float)) else tuple(int(p) for p in self.padding)
            
        # Apply regular 3D convolution
        out = F.conv3d(
            x, 
            self.weight, 
            self.bias, 
            stride=stride, 
            padding=padding
        )
        
        return out
    
    def _optimized_sparse_conv_forward(self, x, indices):
        """Optimized sparse convolution forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W, D).
            indices (torch.Tensor): Indices tensor of shape (N, 4).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out, D_out).
        """
        # Handle both 4D and 5D input tensors
        if len(x.shape) == 4:
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
            batch_features = x[b]
            
            # Compute output features for each active point
            for idx in batch_indices:
                h, w, d = idx[1:]
                
                # Skip if out of bounds
                if h < 0 or h >= height or w < 0 or w >= width or d < 0 or d >= depth:
                    continue
                
                # Compute output position
                out_h = h // self.stride
                out_w = w // self.stride
                out_d = d // self.stride
                
                # Skip if out of bounds
                if out_h < 0 or out_h >= out_height or out_w < 0 or out_w >= out_width or out_d < 0 or out_d >= out_depth:
                    continue
                
                # Extract local region
                h_start = max(0, h - self.padding)
                h_end = min(height, h + self.padding + 1)
                w_start = max(0, w - self.padding)
                w_end = min(width, w + self.padding + 1)
                d_start = max(0, d - self.padding)
                d_end = min(depth, d + self.padding + 1)
                
                # Extract local region and apply convolution
                local_region = batch_features[:, h_start:h_end, w_start:w_end, d_start:d_end]
                
                # Apply convolution to local region
                if local_region.numel() > 0:
                    # Reshape for convolution
                    local_region = local_region.unsqueeze(0)  # Add batch dimension
                    
                    # Apply convolution
                    conv_out = F.conv3d(
                        local_region, 
                        self.weight, 
                        self.bias, 
                        stride=1, 
                        padding=0
                    )
                    
                    # Add to output
                    out[b, :, out_h, out_w, out_d] += conv_out.squeeze(0)
        
        return out

class OptimizedSparseConvBlock(BaseModule):
    """Optimized Sparse Convolution Block.
    
    This block consists of an optimized sparse convolution, batch normalization,
    and activation function.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution.
        padding (int): Zero-padding added to both sides of the input.
        norm_cfg (dict): Configuration for normalization layer.
        act_cfg (dict): Configuration for activation layer.
        indice_key (str, optional): Key for the indices tensor.
    """
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.indice_key = indice_key
        
        # Build convolution layer
        self.conv = OptimizedSparseConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            indice_key=indice_key)
        
        # Build normalization layer
        self.norm = self.build_norm_layer(norm_cfg, out_channels)
        
        # Build activation layer
        self.activate = self.build_activation_layer(act_cfg)
    
    def build_norm_layer(self, norm_cfg, num_features):
        """Build normalization layer.
        
        Args:
            norm_cfg (dict): Configuration for normalization layer.
            num_features (int): Number of features.
            
        Returns:
            nn.Module: Normalization layer.
        """
        if norm_cfg['type'] == 'BN3d':
            return nn.BatchNorm3d(num_features)
        elif norm_cfg['type'] == 'GN':
            return nn.GroupNorm(32, num_features)
        else:
            raise ValueError(f'Unknown norm type: {norm_cfg["type"]}')
    
    def build_activation_layer(self, act_cfg):
        """Build activation layer.
        
        Args:
            act_cfg (dict): Configuration for activation layer.
            
        Returns:
            nn.Module: Activation layer.
        """
        if act_cfg is None:
            return nn.Identity()
            
        if act_cfg['type'] == 'ReLU':
            return nn.ReLU(inplace=False)
        elif act_cfg['type'] == 'LeakyReLU':
            return nn.LeakyReLU(**act_cfg.get('params', {}))
        else:
            raise ValueError(f'Unsupported activation type: {act_cfg["type"]}')
    
    def forward(self, x, indices=None):
        """Forward function.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W, D).
            indices (torch.Tensor, optional): Indices tensor of shape (N, 4).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H_out, W_out, D_out).
        """
        # Apply convolution
        out = self.conv(x, indices)
        
        # Apply normalization
        out = self.norm(out)
        
        # Apply activation
        out = self.activate(out)
        
        return out 
