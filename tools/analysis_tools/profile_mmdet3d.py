# Copyright (c) OpenMMLab. All rights reserved.
import torch
import time
import argparse
from typing import Dict, Tuple, List, Optional, Union, Any
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes

def format_size(size: float) -> str:
    """Convert size to human readable format."""
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if size < 1024.0:
            return f"{size:.2f}{unit}"
        size /= 1024.0
    return f"{size:.2f}E"

def parse_args():
    parser = argparse.ArgumentParser(description='Profile MMDetection3D model')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--shape', type=int, nargs='+', 
                      default=[1, 3, 384, 1280, 40000, 4],
                      help='input shape: [batch, img_c, img_h, img_w, num_points, point_dims]')
    parser.add_argument('--device', type=str, default='cuda:0', 
                      help='device to use for profiling')
    parser.add_argument('--warmup', type=int, default=3,
                      help='number of warmup iterations')
    parser.add_argument('--iterations', type=int, default=10,
                      help='number of iterations to profile')
    return parser.parse_args()

class ModelProfiler:
    """Profiler for MMDetection3D models with support for multi-modality inputs."""
    
    def __init__(self, config_path: str, device: str = 'cuda:0'):
        """Initialize the profiler with a model config.
        
        Args:
            config_path: Path to the model config file
            device: Device to run the model on
        """
        self.device = torch.device(device)
        self.config = Config.fromfile(config_path)
        init_default_scope('mmdet3d')
        self.model = MODELS.build(self.config.model).to(self.device).eval()
        
    def prepare_inputs(self, input_shape: Tuple[int]) -> Dict:
        """Prepare dummy inputs for the MVXNet model.
        
        The input format is based on the KITTI dataset format that MVXNet expects.
        """
        batch, img_c, img_h, img_w, num_points, point_dims = input_shape
        
        # Prepare image input (C, H, W) in BGR format with proper normalization
        # The model expects BGR format with mean=[102.9801, 115.9465, 122.7717] and std=[1.0, 1.0, 1.0]
        # Generate random image in [0, 255] range with BGR channel order
        img = torch.randint(0, 256, (img_h, img_w, 3), dtype=torch.uint8, device=self.device)
        
        # Convert to float32 and normalize with mean and std
        img = img.float()  # Convert to float
        
        # Convert to BGR format (if not already)
        # Note: The model expects BGR format (not RGB) and bgr_to_rgb=False in config
        
        # Normalize using the mean and std from the config
        mean = torch.tensor([102.9801, 115.9465, 122.7717], device=self.device).view(1, 1, -1)
        std = torch.tensor([1.0, 1.0, 1.0], device=self.device).view(1, 1, -1)
        img = (img - mean) / std
        
        # Convert from HWC to CHW format
        img = img.permute(2, 0, 1)  # (C, H, W)
        
        # Add batch dimension for the model
        img = img.unsqueeze(0)  # (1, C, H, W)
        
        # Prepare point cloud input (N, 4) - [x, y, z, intensity]
        # For KITTI, the point cloud is in the camera coordinate system
        # Generate points within the expected range
        point_cloud_range = [0, -40, -3, 70.4, 40, 1]  # From config
        points = torch.zeros((num_points, 4), device=self.device)
        points[:, 0] = torch.rand(num_points, device=self.device) * (point_cloud_range[3] - point_cloud_range[0]) + point_cloud_range[0]  # x
        points[:, 1] = torch.rand(num_points, device=self.device) * (point_cloud_range[4] - point_cloud_range[1]) + point_cloud_range[1]  # y
        points[:, 2] = torch.rand(num_points, device=self.device) * (point_cloud_range[5] - point_cloud_range[2]) + point_cloud_range[2]  # z
        points[:, 3] = torch.rand(num_points, device=self.device)  # intensity
        
        # Create a voxel dictionary to simulate the output of the voxelization layer
        # These are dummy values and may need adjustment based on the actual model configuration
        voxel_dict = {
            'voxels': torch.randn(1000, 32, 4, device=self.device),  # (M, T=32, 4)
            'coors': torch.randint(0, 100, (1000, 4), device=self.device),  # (M, 4)
            'num_points': torch.randint(1, 33, (1000,), device=self.device),  # (M, )
            'coors_3d': torch.randn(1000, 3, device=self.device),  # (M, 3)
            'voxel_centers': torch.randn(1000, 3, device=self.device),  # (M, 3)
            'voxel_size': torch.tensor([0.1, 0.1, 0.2], device=self.device),  # (3, )
            'point_cloud_range': torch.tensor([0, -40, -3, 70.4, 40, 1], device=self.device),  # (6, )
            'batch_size': batch,
            'input_shape': (img_h, img_w)
        }
        
        # For MVXNet, we need to provide both points and image data
        # The model expects a list of dictionaries, one per sample in the batch
        img_metas = {
            'box_type_3d': 'LiDAR',
            'box_mode_3d': 0,  # 0 for LiDAR, 1 for Camera, 2 for Depth
            'img_shape': (img_h, img_w, img_c),
            'ori_shape': (img_h, img_w, img_c),
            'pad_shape': (img_h, img_w, img_c),
            'scale_factor': 1.0,
            'flip': False,
            'pcd_horizontal_flip': False,
            'pcd_vertical_flip': False,
            'pcd_trans': torch.zeros(3, device=self.device),
            'pcd_scale_factor': 1.0,
            'pcd_rotation': torch.eye(3, device=self.device),
            'pcd_rotation_angle': 0.0,
            'voxel_info': {
                'voxel_size': [0.1, 0.1, 0.2],
                'point_cloud_range': [0, -40, -3, 70.4, 40, 1],
                'max_num_points': 32,
                'max_voxels': 1000
            },
            'lidar2img': torch.eye(4, device=self.device, dtype=torch.float32),  # LiDAR to camera transformation
            'sample_idx': 0,
            'pts_filename': 'dummy.bin',
            'img_filename': 'dummy.png',
            'calib': {
                'P2': torch.eye(4, device=self.device, dtype=torch.float32),
                'R0_rect': torch.eye(4, device=self.device, dtype=torch.float32),
                'Tr_velo_to_cam': torch.eye(4, device=self.device, dtype=torch.float32),
                'Tr_imu_to_velo': torch.eye(4, device=self.device, dtype=torch.float32)
            },
            'pcd_scale_factor': 1.0
        }
        
        # For MVXNet, we need to wrap the inputs in a list (one item per sample in batch)
        return {
            'points': [points],  # List of point clouds (one per sample in batch)
            'img': [img],       # List of images (one per sample in batch)
            'img_metas': [img_metas],  # List of metadata (one per sample in batch)
            'voxel_dict': voxel_dict  # Voxelized point cloud data
        }
    
    def profile(self, input_shape: Tuple[int], warmup: int = 3, iterations: int = 10) -> Dict:
        """Profile the model with the given input shape.
        
        Args:
            input_shape: Input shape tuple
            warmup: Number of warmup iterations
            iterations: Number of iterations to profile
            
        Returns:
            Dictionary containing profiling results
        """
        # Prepare inputs
        inputs = self.prepare_inputs(input_shape)
        
        # For DynamicMVXFasterRCNN, we need to prepare the input in the expected format
        def run_forward():
            with torch.no_grad():
                # Create a Det3DDataSample
                data_sample = Det3DDataSample()
                
                # Set the metainfo
                data_sample.set_metainfo(inputs['img_metas'][0])
                
                # Create an InstanceData object for 3D instances
                instances_3d = InstanceData()
                instances_3d.bboxes_3d = LiDARInstance3DBoxes(torch.zeros((0, 7), device=self.device))  # Empty bboxes
                instances_3d.labels_3d = torch.zeros((0,), dtype=torch.long, device=self.device)  # Empty labels
                data_sample.gt_instances_3d = instances_3d
                
                # The model expects a dict with 'inputs' and 'data_samples' keys
                batch_data = {
                    'inputs': {
                        'points': [inputs['points'][0]],  # Original point cloud data
                        'voxels': inputs['voxel_dict'],   # Voxelized point cloud data
                        'img': inputs['img']              # Image data (1, C, H, W)
                    },
                    'data_samples': [data_sample]  # List of data samples, one per sample in batch
                }
                
                # Use test_step for inference
                return self.model.test_step(batch_data)
        
        # Warmup
        for _ in range(warmup):
            _ = run_forward()
        
        # Profile
        start_time = time.time()
        for _ in range(iterations):
            _ = run_forward()
        total_time = time.time() - start_time
        
        # Get model parameters
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model': self.model.__class__.__name__,
            'parameters': params,
            'parameters_str': f"{format_size(params)} parameters",
            'avg_inference_time': total_time / iterations,
            'fps': iterations / total_time,
            'device': str(self.device)
        }
    
    def profile_with_torch_profiler(self, input_shape: Tuple[int], warmup: int = 3, iterations: int = 10) -> Dict:
        """Profile the model using PyTorch's profiler.
        
        Args:
            input_shape: Input shape tuple
            warmup: Number of warmup iterations
            iterations: Number of iterations to profile
            
        Returns:
            Dictionary containing profiling results
        """
        # Prepare inputs
        inputs = self.prepare_inputs(input_shape)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(**inputs)
        
        # Profile with PyTorch profiler
        # Create a Det3DDataSample for profiling
        data_sample = Det3DDataSample()
        data_sample.set_metainfo(inputs['img_metas'][0])
        
        # Create an InstanceData object for 3D instances
        instances_3d = InstanceData()
        instances_3d.bboxes_3d = LiDARInstance3DBoxes(torch.zeros((0, 7), device=self.device))
        instances_3d.labels_3d = torch.zeros((0,), dtype=torch.long, device=self.device)
        data_sample.gt_instances_3d = instances_3d
        
        # Set point cloud data directly
        data_sample.points = inputs['points'][0]
        
        # Prepare batch data for profiling
        batch_data = {
            'inputs': inputs['img'],
            'data_samples': [data_sample]  # One sample per batch
        }
        
        # Create a wrapper function for profiling
        def run_forward():
            with torch.no_grad():
                return self.model.test_step(batch_data)
        
        # Warmup
        for _ in range(2):
            _ = run_forward()
        
        # Profile
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=iterations,
                repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profile'),
            with_stack=True,
            record_shapes=True,
            profile_memory=True,
            with_flops=True
        ) as prof:
            for i in range(iterations + 2):  # +2 for warmup
                _ = run_forward()
                prof.step()
        
        # Get model parameters
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Process profiler results
        events = prof.key_averages()
        cuda_events = [e for e in events if e.device_type == 1]  # CUDA events
        total_cuda_time = sum(e.self_cuda_time_total for e in cuda_events) / 1000  # ms to s
        
        return {
            'model': self.model.__class__.__name__,
            'parameters': params,
            'parameters_str': f"{format_size(params)} parameters",
            'avg_cuda_time': total_cuda_time / iterations,
            'fps': iterations / (total_cuda_time / 1000),  # Convert ms to s
            'device': str(self.device),
            'profiler_output': events.table(sort_by="cuda_time_total", row_limit=20)
        }

def print_profile_results(results: Dict, use_profiler: bool = False):
    """Print profiling results in a formatted way."""
    print("\n" + "="*80)
    print(f"{'MODEL PROFILE':^80}")
    print("="*80)
    
    print(f"\n{'Model:':<20} {results['model']}")
    print(f"{'Device:':<20} {results['device']}")
    print(f"\n{'Parameters:':<20} {results['parameters_str']}")
    
    if use_profiler:
        print(f"\n{'Avg CUDA Time:':<20} {results['avg_cuda_time']:.2f} ms")
        print(f"{'FPS:':<20} {results['fps']:.2f}")
        print("\n" + "="*80)
        print("DETAILED PROFILE (top 20 operators by CUDA time):")
        print("="*80)
        print(results['profiler_output'])
    else:
        print(f"\n{'Avg Inference Time:':<20} {results['avg_inference_time']*1000:.2f} ms")
        print(f"{'FPS:':<20} {results['fps']:.2f}")
    
    print("="*80)
    print("\nNote: For detailed profiling results, check the TensorBoard logs in the 'log/profile' directory.")
    print("="*80 + "\n")

def main():
    args = parse_args()
    
    # Validate input shape
    if len(args.shape) != 6:
        raise ValueError('Input shape must have 6 dimensions: [batch, img_channel, img_h, img_w, num_points, point_dims]')
    
    # Initialize profiler
    profiler = ModelProfiler(args.config, args.device)
    
    try:
        # Run basic profiling
        print("Running basic profiling...")
        basic_results = profiler.profile(
            input_shape=args.shape,
            warmup=args.warmup,
            iterations=args.iterations
        )
        print_profile_results(basic_results)
        
        # Run detailed profiling with PyTorch profiler
        print("\nRunning detailed profiling with PyTorch profiler...")
        detailed_results = profiler.profile_with_torch_profiler(
            input_shape=args.shape,
            warmup=args.warmup,
            iterations=args.iterations
        )
        print_profile_results(detailed_results, use_profiler=True)
        
    except Exception as e:
        print(f"Error during profiling: {e}")
        import traceback
        traceback.print_exc()
        
        # Try running without the profiler if there's an error
        print("\nFalling back to basic profiling...")
        try:
            basic_results = profiler.profile(
                input_shape=args.shape,
                warmup=args.warmup,
                iterations=args.iterations
            )
            print_profile_results(basic_results)
        except Exception as e2:
            print(f"Basic profiling also failed: {e2}")

if __name__ == '__main__':
    main()
