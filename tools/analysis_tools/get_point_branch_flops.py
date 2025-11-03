# tools/analysis_tools/get_point_branch_flops.py
import torch
import sys
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet3d.registry import MODELS
from mmcv.cnn import get_model_complexity_info

class PointBranch(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        print("Initializing PointBranch...")
        self.voxel_encoder = model.pts_voxel_encoder
        self.middle_encoder = model.pts_middle_encoder
        self.backbone = model.pts_backbone
        self.neck = getattr(model, 'pts_neck', None)
        print("PointBranch initialized with voxel_encoder, middle_encoder, backbone, and neck (if exists)")
        
    def forward(self, x):
        print(f"Forward pass with input shape: {x.shape}")
        try:
            print("Running voxel_encoder...")
            voxels, coors, num_points = self.voxel_encoder(x)
            print(f"voxel_encoder output - voxels: {voxels.shape}, coors: {coors.shape}, num_points: {num_points}")
            
            print("Running middle_encoder...")
            x = self.middle_encoder(voxels, coors, num_points)
            print(f"middle_encoder output: {x.shape if isinstance(x, torch.Tensor) else [t.shape if hasattr(t, 'shape') else t for t in x]}")
            
            print("Running backbone...")
            x = self.backbone(x)
            print(f"backbone output: {x[0].shape if isinstance(x, (list, tuple)) else x.shape}")
            
            if self.neck is not None:
                print("Running neck...")
                x = self.neck(x)
                print(f"neck output: {x[0].shape if isinstance(x, (list, tuple)) else x.shape}")
            return x
        except Exception as e:
            print(f"Error in forward pass: {str(e)}", file=sys.stderr)
            raise

def main():
    print("Starting point branch FLOPs calculation...")
    import argparse
    parser = argparse.ArgumentParser(description='Get FLOPs for point branch')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--num-points', type=int, default=40000, 
                       help='number of input points')
    args = parser.parse_args()

    print(f"Loading config from {args.config}...")
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    
    try:
        print("Building model...")
        model = MODELS.build(cfg.model).cuda()
        print("Model built successfully")
        
        print("Creating point branch...")
        point_branch = PointBranch(model).eval()
        print("Point branch created")
        
        input_shape = (args.num_points, 4)
        print(f"Creating input tensor with shape: {input_shape}")
        input_tensor = torch.randn(1, *input_shape).cuda()
        
        print("Calculating FLOPs...")
        flops, params = get_model_complexity_info(
            point_branch, 
            input_shape,
            as_strings=True,
            print_per_layer_stat=True
        )
        
        print('\n' + '=' * 60)
        print(f'Point branch input shape: {input_shape}')
        print(f'FLOPs: {flops}')
        print(f'Params: {params}')
        print('=' * 60)
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()