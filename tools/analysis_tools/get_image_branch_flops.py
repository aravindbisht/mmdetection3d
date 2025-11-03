# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet3d.registry import MODELS

try:
    from mmcv.cnn import get_model_complexity_info  # preferred
except Exception as e:  # pragma: no cover
    raise ImportError(
        'mmcv.cnn.get_model_complexity_info is required. Please install mmcv.') from e


class ImgBranch(torch.nn.Module):
    def __init__(self, backbone, neck=None):
        super().__init__()
        self.backbone = backbone
        self.neck = neck

    def forward(self, x):
        feats = self.backbone(x)
        if self.neck is not None:
            feats = self.neck(feats)
        # mmcv flops util only needs to run forward; output can be any tensor/list
        return feats


def parse_args():
    parser = argparse.ArgumentParser(description='FLOPs for image branch only')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--shape', type=int, nargs=2, default=[1280, 384],
        help='image size as [W H]. Default: 1280 384 (KITTI)')
    parser.add_argument(
        '--scope', type=str, default='mmdet3d', help='default scope to init')
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', args.scope))

    img_backbone_cfg = cfg.model.get('img_backbone', None)
    if img_backbone_cfg is None:
        raise RuntimeError('Config has no img_backbone section')
    img_neck_cfg = cfg.model.get('img_neck', None)

    backbone = MODELS.build(img_backbone_cfg)
    neck = MODELS.build(img_neck_cfg) if img_neck_cfg is not None else None

    net = ImgBranch(backbone, neck)
    net.eval()

    C = 3
    W, H = args.shape
    input_shape = (C, H, W)

    # mmcv returns human-readable strings (flops, params)
    flops, params = get_model_complexity_info(
        net, input_shape, as_strings=True, print_per_layer_stat=False)

    split = '=' * 40
    print(f"{split}\nImage input shape: {input_shape}\nFLOPs: {flops}\nParams: {params}\n{split}")
    print('Note: This counts only img_backbone + img_neck. Heads are not included.')


if __name__ == '__main__':
    main()
