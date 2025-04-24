from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS
import torch

@TRANSFORMS.register_module()
class PadGTBoxVelocity(BaseTransform):
    """Pad LiDAR/Cam depth boxes with zero velocity for datasets that do not
    provide it (e.g., KITTI).

    If the box dimension is 7 (x, y, z, dx, dy, dz, heading) this transform
    appends two zero columns so the tensor becomes 9-dim (… , vx, vy) expected
    by CenterPoint.
    """

    def __init__(self, keys = ('gt_bboxes_3d',)) -> None:
        self.keys = keys

    def transform(self, input_dict: dict) -> dict:
        for key in self.keys:
            if key not in input_dict:
                continue
            bboxes = input_dict[key]
            if bboxes is None:
                continue
            tensor = bboxes.tensor
            # Already has velocity
            if tensor.size(-1) >= 9:
                continue
            assert tensor.size(-1) == 7, (
                'Only support padding 7-dim boxes, ' f'got {tensor.size(-1)}')
            zeros = tensor.new_zeros(tensor.size(0), 2)
            padded = torch.cat([tensor, zeros], dim=-1)
            box_cls = bboxes.__class__
            input_dict[key] = box_cls(padded, box_dim=9, with_yaw=bboxes.with_yaw)
        return input_dict

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(keys={self.keys})'
