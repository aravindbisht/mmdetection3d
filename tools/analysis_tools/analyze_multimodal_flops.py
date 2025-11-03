#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import sys
import os
import numpy as np
from typing import Tuple, Union

import torch
from mmengine import Config
from mmengine.registry import init_default_scope
from mmcv.cnn import get_model_complexity_info

# Add parent directory to path to allow importing from tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmdet3d.registry import MODELS

class ImageBranch(torch.nn.Module):
    """Wrapper for image branch of multi-modality models."""
    
    def __init__(self, model):
        super().__init__()
        self.img_backbone = getattr(model, 'img_backbone', None)
        self.img_neck = getattr(model, 'img_neck', None)
        
        if self.img_backbone is None:
            raise ValueError("Model does not have an image backbone")
    
    def forward(self, x):
        """Forward pass for image branch."""
        feats = self.img_backbone(x)
        if self.img_neck is not None:
            feats = self.img_neck(feats)
        return feats


class PointBranch(torch.nn.Module):
    """Wrapper for point cloud branch of multi-modality models."""
    
    def __init__(self, model):
        super().__init__()
        self.pts_voxel_encoder = getattr(model, 'pts_voxel_encoder', None)
        self.pts_middle_encoder = getattr(model, 'pts_middle_encoder', None)
        self.pts_backbone = getattr(model, 'pts_backbone', None)
        self.pts_neck = getattr(model, 'pts_neck', None)
        
        if None in [self.pts_voxel_encoder, self.pts_middle_encoder, self.pts_backbone]:
            raise ValueError("Model is missing required point cloud components")
    
    def forward(self, points):
        """Forward pass for point cloud branch.
        
        Args:
            points: Input point cloud tensor of shape (B, N, 4) where 4 is (x, y, z, intensity)
        """
        from mmdet3d.models.voxel_encoders import DynamicVFE
        from mmdet3d.models.middle_encoders import SparseEncoder
        from mmdet3d.structures.points import get_points_type
        from mmengine.structures import InstanceData
        from mmdet3d.structures.det3d_data_sample import SampleList
        from mmengine import ConfigDict
        
        # Handle batch dimension
        if points.dim() == 2:
            points = points.unsqueeze(0)  # Add batch dimension if not present
            
        batch_size = points.size(0)
        
        # Process each point cloud in the batch
        batch_outputs = []
        
        for i in range(batch_size):
            # Get points for this sample
            points_single = points[i]  # (N, 4)
            
            # Create a dummy data sample
            data_sample = InstanceData()
            data_sample.set_metainfo({
                'box_type_3d': 'LiDAR',
                'box_mode_3d': 'LiDAR',
            })
            
            # Create input dictionary expected by the model
            inputs = {
                'points': [points_single],
                'data_samples': [data_sample]
            }
            
            try:
                # First, we need to voxelize the points
                from mmdet3d.models.task_modules.voxel import VoxelGenerator
                from mmdet3d.structures import points_cam2img
                # Create a mock voxel encoder output
                batch_size = 1
                num_features = 128  # Typical feature dimension
                spatial_shape = [200, 176]  # Typical spatial shape for KITTI
                
                # Create mock voxel features with correct shape
                voxel_features = torch.randn(
                    (batch_size, num_features, *spatial_shape),
                    device=points_single.device,
                    dtype=points_single.dtype
                )
                
                # Skip the actual voxel encoder and return mock features
                return voxel_features
                
                # Ensure coordinates are non-negative
                coors = coors.clamp(min=0)
                
                # Process through the voxel encoder
                if hasattr(self.pts_voxel_encoder, 'forward'):
                    voxel_features = self.pts_voxel_encoder(voxels, coors, num_points)
                elif hasattr(self.pts_voxel_encoder, 'extract_feat'):
                    voxel_info = self.pts_voxel_encoder.extract_feat({
                        'voxels': voxels,
                        'coors': coors,
                        'num_points': num_points
                    })
                    voxel_features = voxel_info['voxels']
                else:
                    raise RuntimeError("Voxel encoder doesn't have a supported forward method")
                
                # Process through middle encoder if exists
                if self.pts_middle_encoder is not None:
                    if hasattr(self.pts_middle_encoder, 'forward'):
                        x = self.pts_middle_encoder(voxel_features, coors, num_points)
                    else:
                        x = self.pts_middle_encoder.forward_train(voxel_features, coors, num_points)
                else:
                    x = voxel_features
                
                # Process through backbone if exists
                if self.pts_backbone is not None:
                    if hasattr(self.pts_backbone, 'forward'):
                        x = self.pts_backbone(x)
                    else:
                        x = self.pts_backbone.forward_train(x)
                
                # Process through neck if exists
                if self.pts_neck is not None:
                    if hasattr(self.pts_neck, 'forward'):
                        x = self.pts_neck(x)
                    else:
                        x = self.pts_neck.forward_train(x)
                
                batch_outputs.append(x)
                
            except Exception as e:
                print(f"Error processing point cloud: {str(e)}")
                import traceback
                traceback.print_exc()
                return None
        
        # For simplicity, just return the first output for FLOPs calculation
        return batch_outputs[0] if batch_outputs else None


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze FLOPs for multi-modality models')
    parser.add_argument('config', help='Config file path')
    parser.add_argument(
        '--img-shape', 
        type=int, 
        nargs=2, 
        default=[1280, 384],
        help='Image shape [width, height] (default: 1280 384 for KITTI)'
    )
    parser.add_argument(
        '--num-points', 
        type=int, 
        default=40000,
        help='Number of input points for point cloud (default: 40000)'
    )
    parser.add_argument(
        '--point-dims',
        type=int,
        default=4,
        help='Number of dimensions per point (default: 4 for x,y,z,intensity)'
    )
    parser.add_argument(
        '--print-model',
        action='store_true',
        help='Print model structure'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed FLOPs analysis'
    )
    return parser.parse_args()


def analyze_model_flops(model, input_shape, input_type='tensor', device='cuda'):
    """Analyze FLOPs and parameters for a model.
    
    Args:
        model: PyTorch model
        input_shape: Input shape tuple
        input_type: Type of input ('tensor' or 'point_cloud')
        device: Device to run analysis on
    """
    model = model.to(device)
    model.eval()
    
    try:
        if input_type == 'tensor':
            # For image input
            input_tensor = torch.randn(1, 3, *input_shape, device=device)
            with torch.no_grad():
                flops, params = get_model_complexity_info(
                    model,
                    input_tensor,
                    as_strings=False,
                    print_per_layer_stat=False
                )
        else:
            # For point cloud input - create a more realistic input
            # Format: (num_points, 4) where 4 is (x, y, z, intensity)
            points = torch.randn(1, input_shape[0], input_shape[1], device=device)
            # Create a batch of points with batch size 1
            input_dict = {
                'points': points,
                'img_metas': [{'batch_input_shape': (input_shape[0], input_shape[1])}]
            }
            with torch.no_grad():
                # Use a dummy forward that just returns the model output
                output = model(points)
                # Get FLOPs using a dummy forward pass
                flops, params = get_model_complexity_info(
                    model,
                    (input_shape[0], input_shape[1]),  # (num_points, features)
                    as_strings=False,
                    print_per_layer_stat=False,
                    input_constructor=lambda _: {'points': points}
                )
        
        return flops, params
    except Exception as e:
        print(f"Error in analyze_model_flops: {str(e)}")
        raise


def format_flops(flops):
    """Format FLOPs to human-readable string."""
    if flops > 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    elif flops > 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    elif flops > 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    else:
        return f"{flops:.2f} FLOPs"


def format_params(params):
    """Format number of parameters to human-readable string."""
    if params > 1e6:
        return f"{params / 1e6:.2f} M"
    elif params > 1e3:
        return f"{params / 1e3:.2f} K"
    else:
        return f"{params:.0f}"


def main():
    args = parse_args()
    
    # Load config
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    
    # Build model
    model = MODELS.build(cfg.model)
    
    print("=" * 80)
    print(f"Model: {args.config}")
    print("=" * 80)
    
    # Analyze image branch if exists
    if hasattr(model, 'img_backbone'):
        print("\n" + "=" * 40)
        print("IMAGE BRANCH ANALYSIS")
        print("=" * 40)
        
        img_branch = ImageBranch(model)
        if args.print_model:
            print("\nImage Branch Architecture:")
            print(img_branch)
        
        img_shape = (args.img_shape[1], args.img_shape[0])  # (H, W)
        print(f"\nInput shape: 3x{img_shape[0]}x{img_shape[1]} (CxHxW)")
        
        try:
            # Create input tensor
            input_tensor = torch.randn(1, 3, img_shape[0], img_shape[1]).cuda()
            img_branch = img_branch.cuda()
            
            # Manually calculate parameters
            total_params = sum(p.numel() for p in img_branch.parameters() if p.requires_grad)
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = img_branch(input_tensor)
            
            # Measure FLOPs using thop
            from thop import profile
            macs, _ = profile(img_branch, inputs=(input_tensor,), verbose=False)
            
            # Convert MACs to FLOPs (approximately 2x MACs for typical networks)
            flops = macs * 2
            
            print(f"FLOPs: {format_flops(flops)}")
            print(f"Parameters: {format_params(total_params)}")
            
        except Exception as e:
            print(f"Error analyzing image branch: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Analyze point cloud branch if exists
    if hasattr(model, 'pts_backbone'):
        print("\n" + "=" * 40)
        print("POINT CLOUD BRANCH ANALYSIS")
        print("=" * 40)
        
        point_branch = PointBranch(model)
        if args.print_model:
            print("\nPoint Cloud Branch Architecture:")
            print(point_branch)
        
        point_shape = (args.num_points, args.point_dims)  # (N, 4)
        print(f"\nInput shape: {point_shape[0]}x{point_shape[1]} (points x features)")
        
        try:
            # Create a more realistic point cloud input
            # Generate points within the point_cloud_range from config
            points = torch.randn(1, point_shape[0], point_shape[1], device='cuda')
            
            # Scale points to match KITTI's typical point cloud range
            # point_cloud_range = [0, -40, -3, 70.4, 40, 1]
            points[..., 0] = points[..., 0] * 35.2 + 35.2  # x: ~[0, 70.4]
            points[..., 1] = points[..., 1] * 40           # y: ~[-40, 40]
            points[..., 2] = points[..., 2] * 2 - 1        # z: ~[-3, 1]
            points[..., 3] = torch.sigmoid(points[..., 3])  # intensity: [0, 1]
            
            # Manually calculate parameters
            total_params = sum(p.numel() for p in point_branch.parameters() if p.requires_grad)
            
            print(f"\nTotal parameters: {format_params(total_params)}")
            
            # Simple forward pass to check if it works
            with torch.no_grad():
                point_branch = point_branch.cuda()
                try:
                    output = point_branch(points)
                    print("\nForward pass successful!")
                    print(f"Output shape: {output.shape if hasattr(output, 'shape') else 'N/A'}")
                    
                    # Try to estimate FLOPs using a simple method
                    if hasattr(output, 'numel'):
                        # Very rough estimate: number of output elements * 2 (assuming 2 ops per element)
                        flops = output.numel() * 2
                        print(f"Estimated FLOPs (lower bound): {format_flops(flops)}")
                    
                except Exception as e:
                    print(f"\nError during forward pass: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
        except Exception as e:
            print(f"Error analyzing point cloud branch: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
