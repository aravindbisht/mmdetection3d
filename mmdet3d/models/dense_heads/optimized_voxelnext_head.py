import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.models.task_modules.coders import DeltaXYZWLHRBBoxCoder
from ..layers.optimized_sparse_conv import OptimizedSparseConvBlock

@MODELS.register_module()
class OptimizedVoxelNeXtHead(BaseModule):
    """Optimized VoxelNeXt head with improved performance.
    
    This head is an improved version of the VoxelNeXtHead that uses
    optimized sparse convolutions and better memory management.
    
    Args:
        in_channels (int): Number of input channels.
        feat_channels (int): Number of channels in the feature map.
        use_sparse_conv (bool): Whether to use sparse convolutions.
        num_classes (int): Number of classes.
        fusion_layer (dict): Configuration of fusion layer.
        train_cfg (dict): Configuration of training.
        test_cfg (dict): Configuration of testing.
        bbox_coder (dict): Configuration of bbox coder.
        loss_cls (dict): Configuration of classification loss.
        loss_bbox (dict): Configuration of bbox regression loss.
        loss_dir (dict): Configuration of direction classification loss.
        loss_iou (dict): Configuration of IoU loss.
    """
    def __init__(self,
                 in_channels=256,
                 feat_channels=256,
                 use_sparse_conv=True,
                 num_classes=3,
                 fusion_layer=None,
                 train_cfg=None,
                 test_cfg=dict(
                     use_rotate_nms=True,
                     nms_across_levels=False,
                     nms_pre=4096,
                     nms_thr=0.25,
                     score_thr=0.1,
                     min_bbox_size=0,
                     max_num=500,
                     nms_type='box3d_multiclass_nms',  # Options: 'box3d_multiclass_nms', 'aligned_3d_nms', 'circle_nms', 'nms_bev'
                     use_light_nms=True,  # Enable lightweight NMS
                     light_nms_thr=0.1,   # IoU threshold for lightweight NMS
                     chunk_size=10000,
                     pad_size_divisor=32),    # Added pad_size_divisor
                 bbox_coder=dict(
                     type='DeltaXYZWLHRBBoxCoder',
                     target_means=[0., 0., 0., 0., 0., 0., 0.],
                     target_stds=[1., 1., 1., 1., 1., 1., 1.]),
                 loss_cls=dict(
                     type='mmdet.FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='mmdet.SmoothL1Loss',
                     beta=1.0,
                     loss_weight=1.0),
                 loss_dir=dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=0.2),
                 loss_iou=dict(
                     type='IoULoss',
                     loss_weight=1.0)):
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.use_sparse_conv = use_sparse_conv
        self.num_classes = num_classes
        self.fusion_layer = fusion_layer
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # Build bbox coder using TASK_UTILS registry
        self.bbox_coder = TASK_UTILS.build(bbox_coder)

        # Build loss functions
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_dir = MODELS.build(loss_dir)
        self.loss_iou = MODELS.build(loss_iou)

        # Build fusion layer if specified
        if fusion_layer is not None:
            self.fusion_layer = MODELS.build(fusion_layer)
        else:
            self.fusion_layer = None

        # Build shared convolution layers
        if use_sparse_conv:
            self.shared_conv = OptimizedSparseConvBlock(
                in_channels,
                feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=dict(type='BN3d'),
                act_cfg=dict(type='ReLU', inplace=False))
        else:
            self.shared_conv = ConvModule(
                in_channels,
                feat_channels,
                3,
                stride=1,
                padding=1,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=dict(type='BN3d'),
                act_cfg=dict(type='ReLU', inplace=False))

        # Build classification head
        self.conv_cls = nn.Conv3d(feat_channels, num_classes, 1)

        # Build regression head
        self.conv_reg = nn.Conv3d(feat_channels, 7, 1)

        # Build direction classification head
        self.conv_dir_cls = nn.Conv3d(feat_channels, 2, 1)

    def forward(self, x):
        """Forward function.
        
        Args:
            x (list[torch.Tensor]): List of feature maps from backbone.
            
        Returns:
            tuple[list[torch.Tensor]]: Multi-level predictions.
                - cls_scores: List of classification scores.
                - bbox_preds: List of bbox predictions.
                - dir_cls_preds: List of direction classification predictions.
        """
        cls_scores = []
        bbox_preds = []
        dir_cls_preds = []
        
        for feat in x:
            # Create a new tensor to avoid in-place operations
            feat = feat.clone()
            
            # Apply shared convolution
            feat = self.shared_conv(feat)
            
            # Classification branch
            cls_score = self.conv_cls(feat)
            cls_scores.append(cls_score)
            
            # Regression branch
            bbox_pred = self.conv_reg(feat)
            bbox_preds.append(bbox_pred)
            
            # Direction classification branch
            dir_cls_pred = self.conv_dir_cls(feat)
            dir_cls_preds.append(dir_cls_pred)
        
        return cls_scores, bbox_preds, dir_cls_preds

    def loss(self, x, batch_data_samples):
        """Loss function.
        
        Args:
            x (list[torch.Tensor]): List of feature maps from backbone.
            batch_data_samples (list[:obj:`Det3DDataSample`]): The batch
                data samples.
            
        Returns:
            dict: A dictionary of loss components.
        """
        cls_scores, bbox_preds, dir_cls_preds = self(x)
        
        # Get ground truth
        gt_bboxes = []
        gt_labels = []
        for data_sample in batch_data_samples:
            # Access ground truth through gt_instances_3d
            gt_bboxes.append(data_sample.gt_instances_3d.bboxes_3d)
            gt_labels.append(data_sample.gt_instances_3d.labels_3d)
        
        # Get loss items
        loss_dict = {}
        
        # Classification loss
        loss_cls = []
        for cls_score in cls_scores:
            # Reshape cls_score to (N, C) for focal loss
            B, C, H, W, D = cls_score.shape
            cls_score = cls_score.permute(0, 2, 3, 4, 1).reshape(-1, C)
            
            # Create target labels for classification (one-hot encoding)
            target_labels = torch.zeros((B*H*W*D, C), device=cls_score.device)
            for i in range(B):
                for j in range(len(gt_labels[i])):
                    label = gt_labels[i][j]
                    target_labels[i*H*W*D + j, label] = 1
            
            loss_cls.append(self.loss_cls(cls_score, target_labels))
        loss_dict['loss_cls'] = sum(loss_cls)
        
        # Bbox regression loss
        loss_bbox = []
        for bbox_pred in bbox_preds:
            # Reshape bbox_pred to (N, 7) for regression loss
            B, C, H, W, D = bbox_pred.shape
            bbox_pred = bbox_pred.permute(0, 2, 3, 4, 1).reshape(-1, C)
            
            # Create target bboxes for regression
            target_bboxes = torch.zeros((B*H*W*D, C), device=bbox_pred.device)
            for i in range(B):
                for j in range(len(gt_bboxes[i])):
                    # Get the tensor data from the bbox
                    bbox_tensor = gt_bboxes[i][j].tensor
                    # Ensure we have enough dimensions
                    if bbox_tensor.dim() == 1 and bbox_tensor.size(0) < C:
                        # Pad with zeros if needed
                        padded_tensor = torch.zeros(C, device=bbox_tensor.device)
                        padded_tensor[:bbox_tensor.size(0)] = bbox_tensor
                        target_bboxes[i*H*W*D + j] = padded_tensor
                    else:
                        target_bboxes[i*H*W*D + j] = bbox_tensor[:C]
            
            loss_bbox.append(self.loss_bbox(bbox_pred, target_bboxes))
        loss_dict['loss_bbox'] = sum(loss_bbox)
        
        # Direction classification loss
        loss_dir = []
        for dir_cls_pred in dir_cls_preds:
            # Reshape dir_cls_pred to (N, 2) for direction classification
            B, C, H, W, D = dir_cls_pred.shape
            dir_cls_pred = dir_cls_pred.permute(0, 2, 3, 4, 1).reshape(-1, C)
            
            # Create target direction tensor based on heading
            target_dir = torch.zeros((B*H*W*D,), dtype=torch.long, device=dir_cls_pred.device)
            for i in range(B):
                for j in range(len(gt_bboxes[i])):
                    # Get the tensor data from the bbox
                    bbox_tensor = gt_bboxes[i][j].tensor
                    # Check if we have a heading value (usually at index 6)
                    if bbox_tensor.dim() == 1 and bbox_tensor.size(0) > 6:
                        heading = bbox_tensor[6]
                    else:
                        # Default to positive direction if heading not available
                        heading = 1.0
                    target_dir[i*H*W*D + j] = 1 if heading > 0 else 0
            
            loss_dir.append(self.loss_dir(dir_cls_pred, target_dir))
        loss_dict['loss_dir'] = sum(loss_dir)
        
        # IoU loss
        loss_iou = []
        for bbox_pred in bbox_preds:
            # Reshape bbox_pred to (N, 7) for IoU loss
            B, C, H, W, D = bbox_pred.shape
            bbox_pred = bbox_pred.permute(0, 2, 3, 4, 1).reshape(-1, C)
            
            # Create target bboxes for IoU loss
            target_bboxes = torch.zeros((B*H*W*D, C), device=bbox_pred.device)
            for i in range(B):
                for j in range(len(gt_bboxes[i])):
                    # Get the tensor data from the bbox
                    bbox_tensor = gt_bboxes[i][j].tensor
                    # Ensure we have enough dimensions
                    if bbox_tensor.dim() == 1 and bbox_tensor.size(0) < C:
                        # Pad with zeros if needed
                        padded_tensor = torch.zeros(C, device=bbox_tensor.device)
                        padded_tensor[:bbox_tensor.size(0)] = bbox_tensor
                        target_bboxes[i*H*W*D + j] = padded_tensor
                    else:
                        target_bboxes[i*H*W*D + j] = bbox_tensor[:C]
            
            loss_iou.append(self.loss_iou(bbox_pred, target_bboxes))
        loss_dict['loss_iou'] = sum(loss_iou)
        
        return loss_dict

    def _light_nms(self, cls_score, bbox_pred, dir_cls_pred, nms_thr=0.1, chunk_size=10000):
        """Lightweight NMS for 3D boxes that processes in chunks.
        
        Args:
            cls_score (torch.Tensor): Classification scores.
            bbox_pred (torch.Tensor): Bounding box predictions.
            dir_cls_pred (torch.Tensor): Direction classification predictions.
            nms_thr (float): IoU threshold for NMS.
            chunk_size (int): Chunk size for memory efficiency.
            
        Returns:
            dict: Dictionary containing filtered predictions.
        """
        # Get predicted scores and labels
        scores = torch.sigmoid(cls_score)
        labels = torch.argmax(scores, dim=1)
        
        # Get predicted directions
        dir_cls_pred = torch.argmax(dir_cls_pred, dim=1)
        
        # Create anchor tensor for bbox decoding
        B, C, H, W, D = bbox_pred.shape
        anchors = torch.zeros((B, H*W*D, 7), device=bbox_pred.device)
        anchors[..., 3:6] = 1.0  # Set default size to 1.0
        
        # Process bbox_pred in chunks to save memory
        bboxes_list = []
        
        # Pre-allocate memory for chunks
        num_chunks = (B * H * W * D + chunk_size - 1) // chunk_size
        bboxes_list = [None] * num_chunks
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, B * H * W * D)
            
            b_idx = start_idx // (H * W * D)
            h_idx = (start_idx % (H * W * D)) // (W * D)
            w_idx = (start_idx % (W * D)) // D
            d_idx = start_idx % D
            
            # Get chunk of bbox_pred
            bbox_chunk = bbox_pred[b_idx, :, h_idx, w_idx, d_idx].reshape(-1, C)
            anchor_chunk = anchors[b_idx, start_idx-b_idx*H*W*D:end_idx-b_idx*H*W*D]
            
            # Decode bboxes for this chunk
            bbox_chunk = self.bbox_coder.decode(anchor_chunk, bbox_chunk)
            bboxes_list[chunk_idx] = bbox_chunk
        
        # Concatenate all chunks
        bboxes = torch.cat(bboxes_list, dim=0)
        
        # Reshape dir_cls_pred to match bboxes shape
        if dir_cls_pred.dim() == 5:
            dir_cls_pred = dir_cls_pred.permute(0, 2, 3, 4, 1).reshape(B, H*W*D)
        else:
            dir_cls_pred = dir_cls_pred.reshape(B, H*W*D)
        
        # Apply direction correction
        bboxes[..., 6] = bboxes[..., 6] * (dir_cls_pred.view(-1).float() * 2 - 1)
        
        # Reshape predictions to match batch dimension
        bboxes = bboxes.reshape(B, H*W*D, -1)
        scores = scores.permute(0, 2, 3, 4, 1).reshape(B, H*W*D, -1)
        labels = labels.reshape(B, H*W*D)
        
        # Initialize output tensors
        keep = torch.zeros((B, H*W*D), dtype=torch.bool, device=bboxes.device)
        keep[:, 0] = True
        
        # Process NMS in chunks with vectorized operations
        for i in range(1, H*W*D):
            # Get current box
            curr_box = bboxes[:, i:i+1]
            curr_score = scores[:, i:i+1]
            
            # Calculate IoU with previous boxes in chunks
            max_iou = torch.zeros(B, 1, device=bboxes.device)
            
            # Process chunks in parallel
            for j in range(0, i, chunk_size):
                end_j = min(j + chunk_size, i)
                prev_boxes = bboxes[:, j:end_j]
                
                # Vectorized IoU calculation
                iou = self._light_box_iou_3d(curr_box, prev_boxes)
                max_iou = torch.max(max_iou, iou.max(dim=1, keepdim=True)[0])
            
            # Keep box if max IoU is below threshold
            keep[:, i] = max_iou.squeeze(-1) <= nms_thr
        
        # Apply keep mask efficiently
        keep_mask = keep.unsqueeze(-1)
        bboxes_sorted = bboxes[keep_mask.expand_as(bboxes)].reshape(-1, 7)
        scores_sorted = scores[keep_mask.expand_as(scores)].reshape(-1, scores.size(-1))
        labels_sorted = labels[keep].reshape(-1)
        dirs_sorted = dir_cls_pred[keep].reshape(-1)
        
        return {
            'bboxes': bboxes_sorted,
            'scores': scores_sorted,
            'labels': labels_sorted,
            'dirs': dirs_sorted
        }

    def _light_box_iou_3d(self, box1, box2):
        """Lightweight IoU calculation for 3D boxes.
        
        Args:
            box1 (torch.Tensor): First box.
            box2 (torch.Tensor): Second box.
            
        Returns:
            torch.Tensor: IoU values.
        """
        # Calculate intersection volume using simplified approach
        min_xyz = torch.max(box1[..., :3] - box1[..., 3:6] / 2,
                          box2[..., :3] - box2[..., 3:6] / 2)
        max_xyz = torch.min(box1[..., :3] + box1[..., 3:6] / 2,
                          box2[..., :3] + box2[..., 3:6] / 2)
        inter_xyz = torch.clamp(max_xyz - min_xyz, min=0)
        inter_vol = inter_xyz[..., 0] * inter_xyz[..., 1] * inter_xyz[..., 2]
        
        # Calculate union volume
        vol1 = box1[..., 3] * box1[..., 4] * box1[..., 5]
        vol2 = box2[..., 3] * box2[..., 4] * box2[..., 5]
        union_vol = vol1 + vol2 - inter_vol
        
        return inter_vol / (union_vol + 1e-6)

    def predict(self, x, batch_data_samples):
        """Predict function.
        
        Args:
            x (list[torch.Tensor]): List of feature maps from backbone.
            batch_data_samples (list[:obj:`Det3DDataSample`]): The batch
                data samples.
            
        Returns:
            list[:obj:`InstanceData`]: List of prediction results.
        """
        cls_scores, bbox_preds, dir_cls_preds = self(x)
        
        # Process each scale
        results_list = []
        
        for i in range(len(cls_scores)):
            # Get predictions
            cls_score = cls_scores[i]
            bbox_pred = bbox_preds[i]
            dir_cls_pred = dir_cls_preds[i]
            
            # Apply NMS based on configuration
            nms_type = self.test_cfg.get('nms_type', 'box3d_multiclass_nms')
            
            if self.test_cfg.get('use_light_nms', True):
                nms_out = self._light_nms(
                    cls_score, 
                    bbox_pred, 
                    dir_cls_pred,
                    nms_thr=self.test_cfg.get('light_nms_thr', 0.1),
                    chunk_size=self.test_cfg.get('chunk_size', 10000))
            else:
                # Use specified NMS type
                if nms_type == 'box3d_multiclass_nms':
                    nms_out = self._box3d_multiclass_nms(cls_score, bbox_pred, dir_cls_pred)
                elif nms_type == 'aligned_3d_nms':
                    nms_out = self._aligned_3d_nms(cls_score, bbox_pred, dir_cls_pred)
                elif nms_type == 'circle_nms':
                    nms_out = self._circle_nms(cls_score, bbox_pred, dir_cls_pred)
                elif nms_type == 'nms_bev':
                    nms_out = self._nms_bev(cls_score, bbox_pred, dir_cls_pred)
                else:
                    raise ValueError(f'Unknown NMS type: {nms_type}')
            
            # Create InstanceData objects for each sample
            for j in range(len(batch_data_samples)):
                # Get tensors from nms_out
                labels = nms_out['labels'][j]
                scores = nms_out['scores'][j]
                bboxes = nms_out['bboxes'][j]
                dirs = nms_out['dirs'][j]
                
                # Handle 0-d tensors and ensure consistent dimensions
                if labels.dim() == 0:
                    labels = labels.view(1)
                if scores.dim() == 0:
                    scores = scores.view(1)
                if bboxes.dim() == 0:
                    bboxes = bboxes.view(1, 7)  # Ensure 2D tensor with 7 dimensions
                elif bboxes.dim() == 1:
                    bboxes = bboxes.view(1, -1)  # Reshape to 2D tensor
                if dirs.dim() == 0:
                    dirs = dirs.view(1)
                
                # Ensure all tensors have the same length
                num_instances = len(labels)
                if len(scores) != num_instances:
                    scores = scores[:num_instances]
                if len(bboxes) != num_instances:
                    bboxes = bboxes[:num_instances]
                if len(dirs) != num_instances:
                    dirs = dirs[:num_instances]
                
                # Ensure bboxes have correct shape (N, 7)
                if bboxes.size(-1) != 7:
                    # Pad or truncate to 7 dimensions if needed
                    if bboxes.size(-1) < 7:
                        pad_size = 7 - bboxes.size(-1)
                        bboxes = torch.cat([bboxes, torch.zeros_like(bboxes[:, :1]).repeat(1, pad_size)], dim=-1)
                    else:
                        bboxes = bboxes[:, :7]
                
                # Convert bboxes to LiDARInstance3DBoxes
                from mmdet3d.structures import LiDARInstance3DBoxes
                bboxes = LiDARInstance3DBoxes(
                    bboxes,
                    box_dim=7,
                    origin=(0.5, 0.5, 0.5))
                
                # Create InstanceData object
                from mmengine.structures import InstanceData
                pred_instances = InstanceData()
                
                # Update predictions
                pred_instances.labels_3d = labels
                pred_instances.scores_3d = scores
                pred_instances.bboxes_3d = bboxes
                pred_instances.dirs_3d = dirs
                
                # Add to results list
                if j >= len(results_list):
                    results_list.append(pred_instances)
                else:
                    # Merge with existing results
                    existing = results_list[j]
                    if hasattr(existing, 'labels_3d') and existing.labels_3d is not None:
                        # Create new InstanceData with concatenated results
                        merged = InstanceData()
                        merged.labels_3d = torch.cat([existing.labels_3d, labels])
                        merged.scores_3d = torch.cat([existing.scores_3d, scores])
                        # Concatenate LiDARInstance3DBoxes
                        merged.bboxes_3d = LiDARInstance3DBoxes(
                            torch.cat([existing.bboxes_3d.tensor, bboxes.tensor]),
                            box_dim=7,
                            origin=(0.5, 0.5, 0.5))
                        merged.dirs_3d = torch.cat([existing.dirs_3d, dirs])
                        results_list[j] = merged
                    else:
                        # Replace with new results
                        results_list[j] = pred_instances
        
        return results_list

    def _box3d_multiclass_nms(self, cls_score, bbox_pred, dir_cls_pred):
        """Multi-class NMS for 3D boxes.
        
        Args:
            cls_score (torch.Tensor): Classification scores.
            bbox_pred (torch.Tensor): Bounding box predictions.
            dir_cls_pred (torch.Tensor): Direction classification predictions.
            
        Returns:
            dict: Dictionary containing filtered predictions.
        """
        from mmdet3d.models.layers import box3d_multiclass_nms
        
        # Get predicted scores and labels
        scores = torch.sigmoid(cls_score)
        labels = torch.argmax(scores, dim=1)
        
        # Get predicted directions
        dir_cls_pred = torch.argmax(dir_cls_pred, dim=1)
        
        # Create anchor tensor for bbox decoding
        B, C, H, W, D = bbox_pred.shape
        anchors = torch.zeros((B, H*W*D, 7), device=bbox_pred.device)
        anchors[..., 3:6] = 1.0  # Set default size to 1.0
        
        # Decode bboxes
        bboxes = self.bbox_coder.decode(anchors, bbox_pred.permute(0, 2, 3, 4, 1).reshape(B, H*W*D, -1))
        
        # Apply direction correction
        bboxes[..., 6] = bboxes[..., 6] * (dir_cls_pred.float() * 2 - 1)
        
        # Apply NMS
        nms_out = box3d_multiclass_nms(
            bboxes,
            scores.permute(0, 2, 3, 4, 1).reshape(B, H*W*D, -1),
            self.test_cfg.get('nms_thr', 0.25),
            self.test_cfg.get('score_thr', 0.1),
            self.test_cfg.get('max_num', 500))
        
        return {
            'bboxes': nms_out[0],
            'scores': nms_out[1],
            'labels': nms_out[2],
            'dirs': dir_cls_pred[nms_out[2]]
        }

    def _aligned_3d_nms(self, cls_score, bbox_pred, dir_cls_pred):
        """Aligned 3D NMS.
        
        Args:
            cls_score (torch.Tensor): Classification scores.
            bbox_pred (torch.Tensor): Bounding box predictions.
            dir_cls_pred (torch.Tensor): Direction classification predictions.
            
        Returns:
            dict: Dictionary containing filtered predictions.
        """
        from mmdet3d.models.layers import aligned_3d_nms
        
        # Similar to box3d_multiclass_nms but uses aligned_3d_nms
        # Implementation similar to _box3d_multiclass_nms but with aligned_3d_nms
        pass

    def _circle_nms(self, cls_score, bbox_pred, dir_cls_pred):
        """Circle NMS.
        
        Args:
            cls_score (torch.Tensor): Classification scores.
            bbox_pred (torch.Tensor): Bounding box predictions.
            dir_cls_pred (torch.Tensor): Direction classification predictions.
            
        Returns:
            dict: Dictionary containing filtered predictions.
        """
        from mmdet3d.models.layers import circle_nms
        
        # Similar to box3d_multiclass_nms but uses circle_nms
        # Implementation similar to _box3d_multiclass_nms but with circle_nms
        pass

    def _nms_bev(self, cls_score, bbox_pred, dir_cls_pred):
        """Bird's Eye View NMS.
        
        Args:
            cls_score (torch.Tensor): Classification scores.
            bbox_pred (torch.Tensor): Bounding box predictions.
            dir_cls_pred (torch.Tensor): Direction classification predictions.
            
        Returns:
            dict: Dictionary containing filtered predictions.
        """
        from mmdet3d.models.layers import nms_bev
        
        # Similar to box3d_multiclass_nms but uses nms_bev
        # Implementation similar to _box3d_multiclass_nms but with nms_bev
        pass

    def loss_by_feat(self,
                    cls_scores,
                    bbox_preds,
                    dir_cls_preds,
                    batch_gt_instances_3d,
                    batch_gt_instances_ignore=None,
                    **kwargs):
        """Loss function.

        Args:
            cls_scores (list[Tensor]): Classification scores for each scale level.
            bbox_preds (list[Tensor]): Box regression for each scale level.
            dir_cls_preds (list[Tensor]): Direction classification for each scale level.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances_3d. It usually includes ``bboxes_3d`` and ``labels_3d``
                attributes.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional): Batch of
                gt_instances_ignore. It includes ``bboxes_3d`` attribute.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        loss_dict = {}
        
        # Get ground truth data
        gt_labels_3d = [gt_instances_3d.labels_3d for gt_instances_3d in batch_gt_instances_3d]
        gt_bboxes_3d = [gt_instances_3d.bboxes_3d for gt_instances_3d in batch_gt_instances_3d]
        
        # Classification loss
        labels_3d = torch.cat(gt_labels_3d, dim=0)
        cls_scores = torch.cat(cls_scores, dim=0)
        loss_dict['loss_cls'] = self.loss_cls(cls_scores, labels_3d)
        
        # Bbox regression loss
        bbox_preds = torch.cat(bbox_preds, dim=0)
        bbox_targets = self.bbox_coder.encode(gt_bboxes_3d)
        loss_dict['loss_bbox'] = self.loss_bbox(bbox_preds, bbox_targets)
        
        # Direction classification loss
        dir_cls_preds = torch.cat(dir_cls_preds, dim=0)
        dir_labels = torch.cat([bbox.dir for bbox in gt_bboxes_3d], dim=0)
        loss_dict['loss_dir'] = self.loss_dir(dir_cls_preds, dir_labels)
        
        # IoU loss
        decoded_bboxes = self.bbox_coder.decode(bbox_preds)
        loss_dict['loss_iou'] = self.loss_iou(decoded_bboxes, gt_bboxes_3d)
        
        return loss_dict

    def predict_by_feat(self, cls_scores, bbox_preds, dir_cls_preds):
        """Predict bboxes by features.
        
        Args:
            cls_scores (list[torch.Tensor]): List of classification scores.
            bbox_preds (list[torch.Tensor]): List of bbox predictions.
            dir_cls_preds (list[torch.Tensor]): List of direction classification predictions.
            
        Returns:
            tuple[torch.Tensor]: Predictions.
                - bboxes: Predicted bboxes.
                - scores: Predicted scores.
        """
        # Get the last level predictions
        cls_score = cls_scores[-1]
        bbox_pred = bbox_preds[-1]
        dir_cls_pred = dir_cls_preds[-1]
        
        # Get predicted scores and labels
        scores = torch.sigmoid(cls_score)
        labels = torch.argmax(scores, dim=1)
        
        # Get predicted directions
        dir_cls_pred = torch.argmax(dir_cls_pred, dim=1)
        
        # Create anchor tensor for bbox decoding
        anchors = torch.zeros_like(bbox_pred)
        anchors[..., 3:6] = 1.0  # Set default size to 1.0
        
        # Decode bboxes
        bboxes = self.bbox_coder.decode(anchors, bbox_pred)
        
        # Reshape dir_cls_pred to match bboxes shape
        dir_cls_pred = dir_cls_pred.permute(0, 2, 3, 4).reshape(B, H*W*D)
        
        # Apply direction correction
        bboxes[..., 6] = bboxes[..., 6] * (dir_cls_pred.float().unsqueeze(-1) * 2 - 1)
        
        # Sort by scores
        scores = scores.permute(0, 2, 3, 4, 1).reshape(B, H*W*D, -1)
        scores, order = scores.sort(1, descending=True)
        bboxes = bboxes[torch.arange(B).view(-1, 1), order]
        labels = labels.permute(0, 2, 3, 4).reshape(B, H*W*D)[torch.arange(B).view(-1, 1), order]
        dirs = dir_cls_pred[torch.arange(B).view(-1, 1), order]
        
        # Apply NMS
        bboxes, scores = self._rotate_nms(bboxes, scores)
        
        # Convert to LiDARInstance3DBoxes
        bboxes = LiDARInstance3DBoxes(
            bboxes,
            box_dim=7,
            origin=(0.5, 0.5, 0.5))
        
        return bboxes, scores

    def _rotate_nms(self, cls_score, bbox_pred, dir_cls_pred):
        """Rotated NMS for 3D boxes.
        
        Args:
            cls_score (torch.Tensor): Classification scores.
            bbox_pred (torch.Tensor): Bounding box predictions.
            dir_cls_pred (torch.Tensor): Direction classification predictions.
            
        Returns:
            dict: Dictionary containing filtered predictions.
                - bboxes: Filtered bounding boxes
                - scores: Filtered scores
                - labels: Filtered labels
                - dirs: Filtered directions
        """
        # Get predicted scores and labels
        scores = torch.sigmoid(cls_score)
        labels = torch.argmax(scores, dim=1)
        
        # Get predicted directions
        dir_cls_pred = torch.argmax(dir_cls_pred, dim=1)
        
        # Create anchor tensor for bbox decoding
        # Reshape bbox_pred to get the batch and spatial dimensions
        B, C, H, W, D = bbox_pred.shape
        anchors = torch.zeros((B, H*W*D, 7), device=bbox_pred.device)
        anchors[..., 3:6] = 1.0  # Set default size to 1.0
        
        # Process bbox_pred in chunks to save memory
        chunk_size = min(10000, H*W*D)  # Adjust chunk size based on spatial dimensions
        bboxes_list = []
        
        for i in range(0, B * H * W * D, chunk_size):
            end_idx = min(i + chunk_size, B * H * W * D)
            b_idx = i // (H * W * D)
            h_idx = (i % (H * W * D)) // (W * D)
            w_idx = (i % (W * D)) // D
            d_idx = i % D
            
            # Get chunk of bbox_pred
            bbox_chunk = bbox_pred[b_idx, :, h_idx, w_idx, d_idx].reshape(-1, C)
            anchor_chunk = anchors[b_idx, i-b_idx*H*W*D:end_idx-b_idx*H*W*D]
            
            # Decode bboxes for this chunk
            bbox_chunk = self.bbox_coder.decode(anchor_chunk, bbox_chunk)
            bboxes_list.append(bbox_chunk)
            
            # Clear memory
            del bbox_chunk, anchor_chunk
            torch.cuda.empty_cache()
        
        # Concatenate all chunks
        bboxes = torch.cat(bboxes_list, dim=0)
        del bboxes_list
        torch.cuda.empty_cache()
        
        # Reshape dir_cls_pred to match bboxes shape
        if dir_cls_pred.dim() == 5:
            dir_cls_pred = dir_cls_pred.permute(0, 2, 3, 4, 1).reshape(B, H*W*D)
        else:
            dir_cls_pred = dir_cls_pred.reshape(B, H*W*D)
        
        # Apply direction correction in chunks
        for i in range(0, B * H * W * D, chunk_size):
            end_idx = min(i + chunk_size, B * H * W * D)
            bboxes[i:end_idx, 6] = bboxes[i:end_idx, 6] * (dir_cls_pred.view(-1)[i:end_idx].float() * 2 - 1)
        
        # Process scores in chunks
        scores = scores.permute(0, 2, 3, 4, 1).reshape(B, H*W*D, -1)
        scores, order = scores.sort(1, descending=True)
        
        # Reshape bboxes to match batch dimension
        bboxes = bboxes.reshape(B, H*W*D, -1)
        
        # Apply sorting to bboxes and labels in chunks
        bboxes_sorted = torch.zeros((B, H*W*D, 7), device=bboxes.device)
        labels_sorted = torch.zeros((B, H*W*D), device=labels.device)
        dirs_sorted = torch.zeros((B, H*W*D), device=dir_cls_pred.device)
        
        for i in range(0, B * H * W * D, chunk_size):
            end_idx = min(i + chunk_size, B * H * W * D)
            b_idx = i // (H * W * D)
            local_idx = i % (H * W * D)
            local_end = min(local_idx + (end_idx - i), H * W * D)
            
            # Get the indices for this chunk
            chunk_order = order[b_idx, local_idx:local_end]
            
            # Sort bboxes - ensure dimensions match
            bbox_chunk = bboxes[b_idx, chunk_order].reshape(-1, 7)
            bboxes_sorted[b_idx, local_idx:local_end] = bbox_chunk
            
            # Sort labels
            label_chunk = labels.reshape(B, H*W*D)[b_idx, chunk_order]
            labels_sorted[b_idx, local_idx:local_end] = label_chunk
            
            # Sort directions
            dir_chunk = dir_cls_pred[b_idx, chunk_order]
            dirs_sorted[b_idx, local_idx:local_end] = dir_chunk
            
            # Clear memory
            del bbox_chunk, label_chunk, dir_chunk
            torch.cuda.empty_cache()
        
        # Initialize output tensors
        keep = torch.zeros_like(scores, dtype=torch.bool)
        keep[:, 0] = True
        
        # Calculate IoU between boxes in chunks
        for i in range(1, H*W*D):
            iou = self._box_iou_3d(bboxes_sorted[:, i:i+1], bboxes_sorted[:, :i])
            if (iou <= 0.1).all():
                keep[:, i] = True
            
            # Clear memory periodically
            if i % 1000 == 0:
                torch.cuda.empty_cache()
        
        # Clear unnecessary tensors
        del bboxes, scores, labels, dir_cls_pred
        torch.cuda.empty_cache()
        
        # Reshape outputs to match expected format
        bboxes_sorted = bboxes_sorted.reshape(-1, bboxes_sorted.size(-1))
        labels_sorted = labels_sorted.reshape(-1)
        dirs_sorted = dirs_sorted.reshape(-1)
        
        return {
            'bboxes': bboxes_sorted[keep.reshape(-1)],
            'scores': scores[keep],
            'labels': labels_sorted[keep.reshape(-1)],
            'dirs': dirs_sorted[keep.reshape(-1)]
        }

    def _box_iou_3d(self, box1, box2):
        """Calculate IoU between 3D boxes.
        
        Args:
            box1 (torch.Tensor): First box.
            box2 (torch.Tensor): Second box.
            
        Returns:
            torch.Tensor: IoU values.
        """
        # Calculate intersection volume
        min_xyz = torch.max(box1[..., :3] - box1[..., 3:6] / 2,
                          box2[..., :3] - box2[..., 3:6] / 2)
        max_xyz = torch.min(box1[..., :3] + box1[..., 3:6] / 2,
                          box2[..., :3] + box2[..., 3:6] / 2)
        inter_xyz = torch.clamp(max_xyz - min_xyz, min=0)
        inter_vol = inter_xyz[..., 0] * inter_xyz[..., 1] * inter_xyz[..., 2]
        
        # Calculate union volume
        vol1 = box1[..., 3] * box1[..., 4] * box1[..., 5]
        vol2 = box2[..., 3] * box2[..., 4] * box2[..., 5]
        union_vol = vol1 + vol2 - inter_vol
        
        return inter_vol / (union_vol + 1e-6) 
