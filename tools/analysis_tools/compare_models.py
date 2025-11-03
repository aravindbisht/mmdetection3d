# compare_models.py
import time
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample

class ModelProfiler:
    def __init__(self, config_path, device='cuda:0'):
        self.device = device
        self.config = Config.fromfile(config_path)
        init_default_scope('mmdet3d')
        self.model = MODELS.build(self.config.model).to(self.device).eval()
        
    def prepare_inputs(self, input_shape):
        """Prepare dummy inputs and minimal data_samples for the model."""
        batch, img_c, img_h, img_w, num_points, _ = input_shape

        # Image input as CHW uint8 (preprocessor will normalize/pad)
        img = torch.randint(0, 256, (img_c, img_h, img_w),
                            dtype=torch.uint8, device=self.device)

        # Point cloud input (N, 4)
        point_cloud_range = [0, -40, -3, 70.4, 40, 1]
        points = torch.zeros((num_points, 4), device=self.device)
        points[:, 0] = torch.rand(num_points, device=self.device) * (point_cloud_range[3] - point_cloud_range[0]) + point_cloud_range[0]
        points[:, 1] = torch.rand(num_points, device=self.device) * (point_cloud_range[4] - point_cloud_range[1]) + point_cloud_range[1]
        points[:, 2] = torch.rand(num_points, device=self.device) * (point_cloud_range[5] - point_cloud_range[2]) + point_cloud_range[2]
        points[:, 3] = torch.rand(num_points, device=self.device)  # intensity

        # Minimal metainfo required by PointFusion and projection utilities
        data_sample = Det3DDataSample()
        data_sample.set_metainfo({
            'img_shape': (img_h, img_w, img_c),
            'ori_shape': (img_h, img_w, img_c),
            'scale_factor': (1.0, 1.0, 1.0, 1.0),
            'img_crop_offset': (0, 0),
            'flip': False,
            'lidar2img': torch.eye(4, device=self.device, dtype=torch.float32),
        })

        # Raw batch in dataloader style (preprocessor will convert 'img'->'imgs' and add voxels)
        raw_batch = {
            'inputs': {
                'points': [points],
                'img': [img]
            },
            'data_samples': [data_sample]
        }
        return raw_batch
    
    def profile(self, input_shape, warmup=3, iterations=10):
        """Profile the model with the given input shape."""
        raw_batch = self.prepare_inputs(input_shape)

        # Preprocess once to get padded shapes and voxel_dict
        processed = self.model.data_preprocessor(raw_batch, training=False)

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = self.model.predict(processed['inputs'], processed['data_samples'])

        # Profile
        start_time = time.time()
        for _ in range(iterations):
            with torch.no_grad():
                _ = self.model.predict(processed['inputs'], processed['data_samples'])
        avg_inference_time = (time.time() - start_time) / iterations
        
        # Get parameters
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'avg_inference_time': avg_inference_time,
            'fps': 1.0 / avg_inference_time,
            'params': f"{params/1e6:.2f}M",
            'device': str(self.device)
        }

def main():
    # Input shape: (batch, img_c, img_h, img_w, num_points, point_dims)
    input_shape = (1, 3, 384, 1280, 100000, 4)
    
    # Model configs to compare
    model_configs = {
        'MVXNet + Second FPN': 'configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py',
        'MVXNet + SqueezeFPN + FireRPFNet': 'configs/mvxnet/mvxnet_sqeezefpn_fire_rpfnet_kitti-3d-3class.py'
    }
    
    print("\nProfiling Models...")
    print("=" * 80)
    print(f"{'Model':<40} | {'Inference Time (ms)':<20} | {'FPS':<15} | {'Params':<15}")
    print("-" * 80)
    
    for name, config_path in model_configs.items():
        try:
            profiler = ModelProfiler(config_path)
            results = profiler.profile(input_shape)
            print(f"{name:<40} | {results['avg_inference_time']*1000:.2f} ms{'':<10} | {results['fps']:.2f}{'':<10} | {results['params']}")
        except Exception as e:
            print(f"Error profiling {name}: {str(e)}")

if __name__ == '__main__':
    main()