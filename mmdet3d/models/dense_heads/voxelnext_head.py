import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmdet.models.losses import FocalLoss, SmoothL1Loss
from mmdet3d.models.losses import RotatedIoU3DLoss
from mmdet3d.structures import Det3DDataSample
from mmdet3d.models.task_modules import Anchor3DRangeGenerator
from mmdet3d.models.task_modules.coders import DeltaXYZWLHRBBoxCoder
from mmdet3d.models.dense_heads.base_3d_dense_head import Base3DDenseHead
from mmdet3d.utils.typing_utils import InstanceList, SampleList
from typing import List, Optional, Tuple, Dict, Union
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures import LiDARInstance3DBoxes

@MODELS.register_module()
class VoxelNeXtHead(Base3DDenseHead):
    """Anchor-free 3D detection head inspired by VoxelNeXt.
    
    This head predicts objects directly from sparse voxel features without
    using anchors or center proxies.
    """
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 use_direction_classifier=True,
                 loss_cls=dict(
                     type='mmdet.FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='mmdet.SmoothL1Loss',
                     beta=1.0 / 9.0,
                     loss_weight=2.0),
                 loss_dir=dict(
                     type='mmdet.CrossEntropyLoss', use_sigmoid=False,
                     loss_weight=0.2),
                 loss_iou=dict(
                     type='RotatedIoU3DLoss',
                     loss_weight=1.0),
                 bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.use_direction_classifier = use_direction_classifier
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # Initialize bbox coder properly
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        
        # Build shared conv layers with additional batch normalization
        self.shared_conv = nn.Sequential(
            ConvModule(
                in_channels,
                feat_channels,
                3,
                padding=1,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=dict(type='BN3d'),
                act_cfg=dict(type='ReLU', inplace=False)),
            nn.BatchNorm3d(feat_channels),
            nn.ReLU(inplace=False),
            ConvModule(
                feat_channels,
                feat_channels,
                3,
                padding=1,
                conv_cfg=dict(type='Conv3d'),
                norm_cfg=dict(type='BN3d'),
                act_cfg=dict(type='ReLU', inplace=False)),
            nn.BatchNorm3d(feat_channels),
            nn.ReLU(inplace=False))
        
        # Classification head with batch normalization
        self.conv_cls = nn.Sequential(
            nn.Conv3d(feat_channels, num_classes, 1),
            nn.BatchNorm3d(num_classes))
        
        # Regression head with batch normalization
        self.conv_reg = nn.Sequential(
            nn.Conv3d(feat_channels, 7, 1),
            nn.BatchNorm3d(7))
        
        # Direction classifier with batch normalization
        if use_direction_classifier:
            self.conv_dir_cls = nn.Sequential(
                nn.Conv3d(feat_channels, 2, 1),
                nn.BatchNorm3d(2))
        
        # Loss functions
        loss_cls_copy = loss_cls.copy()
        loss_cls_copy.pop('type', None)
        self.loss_cls = FocalLoss(**loss_cls_copy)
        
        loss_bbox_copy = loss_bbox.copy()
        loss_bbox_copy.pop('type', None)
        self.loss_bbox = SmoothL1Loss(**loss_bbox_copy)
        
        loss_dir_copy = loss_dir.copy()
        loss_dir_copy.pop('type', None)
        loss_dir_copy.pop('use_sigmoid', None)
        self.loss_dir_weight = loss_dir_copy.pop('loss_weight', 0.2)
        self.loss_dir = nn.CrossEntropyLoss(**loss_dir_copy)
        
        loss_iou_copy = loss_iou.copy()
        loss_iou_copy.pop('type', None)
        self.loss_iou = RotatedIoU3DLoss(**loss_iou_copy)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights of the head."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward function.
        
        Args:
            x (list[Tensor]): List of 4D tensors of shape (N, C, H, W, D).
            
        Returns:
            tuple[list[Tensor]]: Multi-level predictions.
                - cls_scores (list[Tensor]): Classification scores for each level.
                - bbox_preds (list[Tensor]): Bbox predictions for each level.
                - dir_cls_preds (list[Tensor]): Direction classification for each level.
        """
        cls_scores = []
        bbox_preds = []
        dir_cls_preds = []
        
        for i, feat in enumerate(x):
            # Apply shared conv layers
            feat = feat.clone()  # Create a new tensor to avoid in-place operations
            feat = self.shared_conv(feat)
            
            # Classification prediction
            cls_score = self.conv_cls(feat)
            cls_scores.append(cls_score.clone())  # Create a new tensor
            
            # Bbox prediction
            bbox_pred = self.conv_reg(feat)
            bbox_preds.append(bbox_pred.clone())  # Create a new tensor
            
            # Direction classification
            if self.use_direction_classifier:
                dir_cls_pred = self.conv_dir_cls(feat)
                dir_cls_preds.append(dir_cls_pred.clone())  # Create a new tensor
            else:
                dir_cls_preds.append(None)
        
        return cls_scores, bbox_preds, dir_cls_preds
    
    def loss_by_feat(self, cls_scores, bbox_preds, dir_cls_preds, batch_gt_instances_3d, batch_gt_instances_ignore=None, batch_input_metas=None):
        """Loss function.
        
        Args:
            cls_scores (list[Tensor]): Classification scores for each level.
            bbox_preds (list[Tensor]): Bbox predictions for each level.
            dir_cls_preds (list[Tensor]): Direction classification for each level.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of gt instances.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional): Batch of gt instances to ignore.
            batch_input_metas (list[dict], optional): Batch input metas.
                
        Returns:
            dict: A dictionary of loss components.
        """
        # Get ground truth
        gt_labels_3d = [gt_instances_3d.labels_3d.clone() for gt_instances_3d in batch_gt_instances_3d]  # Create new tensors
        gt_bboxes_3d = [gt_instances_3d.bboxes_3d for gt_instances_3d in batch_gt_instances_3d]
        
        # Find the maximum number of ground truth objects in any sample
        max_num_gt = max([len(gt_labels) for gt_labels in gt_labels_3d])
        
        # Pad ground truth labels and boxes to the same size
        padded_gt_labels = []
        padded_gt_bboxes = []
        
        for i, (labels, bboxes) in enumerate(zip(gt_labels_3d, gt_bboxes_3d)):
            num_gt = len(labels)
            if num_gt == 0:
                # If no ground truth, create empty tensors
                padded_labels = torch.zeros(max_num_gt, dtype=labels.dtype, device=labels.device)
                padded_bboxes = torch.zeros((max_num_gt, 7), dtype=bboxes.tensor.dtype, device=bboxes.tensor.device)
            else:
                # Pad with zeros to match the maximum number of ground truth objects
                padded_labels = torch.zeros(max_num_gt, dtype=labels.dtype, device=labels.device)
                padded_labels[:num_gt] = labels
                
                padded_bboxes = torch.zeros((max_num_gt, 7), dtype=bboxes.tensor.dtype, device=bboxes.tensor.device)
                padded_bboxes[:num_gt] = bboxes.tensor
            
            padded_gt_labels.append(padded_labels)
            padded_gt_bboxes.append(padded_bboxes)
        
        # Convert ground truth to tensors
        gt_labels_3d = torch.stack(padded_gt_labels)  # (B, max_num_gt)
        gt_bboxes_3d = torch.stack(padded_gt_bboxes)  # (B, max_num_gt, 7)
        
        # Calculate losses
        losses = {}
        
        # Initialize loss components
        num_levels = len(cls_scores)
        cls_loss = []
        bbox_loss = []
        dir_loss = []
        iou_loss = []
        
        # Compute losses for each level
        for level in range(num_levels):
            # Reshape predictions to match ground truth
            cls_score = cls_scores[level].clone()  # (B, C, H, W, D)
            bbox_pred = bbox_preds[level].clone()  # (B, 7, H, W, D)
            if self.use_direction_classifier:
                dir_cls_pred = dir_cls_preds[level].clone()  # (B, 2, H, W, D)
            
            # Reshape predictions for loss computation
            B, C, H, W, D = cls_score.shape
            cls_score = cls_score.permute(0, 2, 3, 4, 1).reshape(-1, C)  # (B*H*W*D, C)
            bbox_pred = bbox_pred.permute(0, 2, 3, 4, 1).reshape(-1, 7)  # (B*H*W*D, 7)
            if self.use_direction_classifier:
                dir_cls_pred = dir_cls_pred.permute(0, 2, 3, 4, 1).reshape(-1, 2)  # (B*H*W*D, 2)
            
            # Create target labels for classification
            target_labels = torch.zeros((B*H*W*D, C), device=cls_score.device)
            for i in range(B):
                for j in range(max_num_gt):
                    if j < len(gt_labels_3d[i]) and gt_labels_3d[i][j] >= 0:  # Check if valid label
                        label = gt_labels_3d[i][j]
                        if label < C:  # Ensure label is within valid range
                            idx = i*H*W*D + j
                            if idx < target_labels.shape[0]:  # Check if index is within bounds
                                target_labels[idx, label] = 1
            
            # Create target bboxes for regression
            target_bboxes = torch.zeros((B*H*W*D, 7), device=bbox_pred.device)
            for i in range(B):
                for j in range(max_num_gt):
                    if j < len(gt_bboxes_3d[i]) and gt_bboxes_3d[i][j].sum() > 0:  # Check if valid bbox
                        idx = i*H*W*D + j
                        if idx < target_bboxes.shape[0]:  # Check if index is within bounds
                            target_bboxes[idx] = gt_bboxes_3d[i][j]
            
            # Create target direction for direction classification
            if self.use_direction_classifier:
                target_direction = torch.zeros((B*H*W*D,), dtype=torch.long, device=dir_cls_pred.device)
                for i in range(B):
                    for j in range(max_num_gt):
                        if j < len(gt_bboxes_3d[i]) and gt_bboxes_3d[i][j].sum() > 0:  # Check if valid bbox
                            idx = i*H*W*D + j
                            if idx < target_direction.shape[0]:  # Check if index is within bounds
                                # Use the last dimension (heading) to determine direction
                                heading = gt_bboxes_3d[i][j][-1]
                                target_direction[idx] = 1 if heading > 0 else 0
            
            # Classification loss
            cls_loss.append(self.loss_cls(cls_score, target_labels))
            
            # Bbox regression loss
            bbox_loss.append(self.loss_bbox(bbox_pred, target_bboxes))
            
            # Direction classification loss
            if self.use_direction_classifier:
                dir_loss.append(self.loss_dir(dir_cls_pred, target_direction))
            
            # IoU loss
            iou_loss.append(self.loss_iou(bbox_pred, target_bboxes))
        
        # Combine losses from all levels
        losses['loss_cls'] = sum(cls_loss) / num_levels
        losses['loss_bbox'] = sum(bbox_loss) / num_levels
        if self.use_direction_classifier:
            losses['loss_dir'] = sum(dir_loss) / num_levels * self.loss_dir_weight
        losses['loss_iou'] = sum(iou_loss) / num_levels
        
        return losses
    
    def predict_by_feat(self, cls_scores, bbox_preds, dir_cls_preds, batch_input_metas=None, cfg=None, rescale=False):
        """Predict function.
        
        Args:
            cls_scores (list[Tensor]): Classification scores for each level.
            bbox_preds (list[Tensor]): Bbox predictions for each level.
            dir_cls_preds (list[Tensor]): Direction classification for each level.
            batch_input_metas (list[dict], optional): Batch input metas.
            cfg (ConfigDict, optional): Test / postprocessing configuration.
            rescale (bool): Whether to rescale the results.
                
        Returns:
            list[:obj:`InstanceData`]: Detection results of the input images.
        """
        cfg = self.test_cfg if cfg is None else cfg
        
        # Post-process
        results = []
        for i in range(len(cls_scores)):
            # Get predictions
            cls_score = cls_scores[i]
            bbox_pred = bbox_preds[i]
            dir_cls_pred = dir_cls_preds[i] if self.use_direction_classifier else None
            
            # Reshape bbox_pred for decoding
            B, C, H, W, D = bbox_pred.shape
            bbox_pred = bbox_pred.permute(0, 2, 3, 4, 1).reshape(-1, 7)  # (B*H*W*D, 7)
            
            # Create anchor tensor for decoding
            # For anchor-free methods, we use a default anchor
            anchors = torch.zeros((bbox_pred.shape[0], 7), device=bbox_pred.device)
            # Set default values for anchors (can be adjusted based on your needs)
            anchors[:, 3:6] = 1.0  # Default width, length, height
            
            # Decode bbox predictions
            bboxes = self.bbox_coder.decode(anchors, bbox_pred)
            
            # Get scores
            scores = F.sigmoid(cls_score)
            scores = scores.permute(0, 2, 3, 4, 1).reshape(-1, self.num_classes)  # (B*H*W*D, num_classes)
            
            # Filter by score threshold
            score_thr = cfg.get('score_thr', 0.1)
            mask = scores > score_thr
            valid_indices = mask.any(dim=1)
            bboxes = bboxes[valid_indices]
            scores = scores[valid_indices]
            
            # Ensure scores have the correct format for KITTI
            # KITTI expects a 1D array of scores
            if len(scores) > 0:
                # Get the maximum score for each box
                scores = scores.max(dim=1)[0]
            
            # Get labels before NMS
            if len(scores) > 0 and len(cls_scores) > 0:
                cls_score = cls_scores[i]
                cls_score = F.sigmoid(cls_score)
                cls_score = cls_score.permute(0, 2, 3, 4, 1).reshape(-1, self.num_classes)
                cls_score = cls_score[valid_indices]
                labels = cls_score.argmax(dim=1)
            else:
                labels = scores.new_zeros(scores.size(0), dtype=torch.long)
            
            # Apply NMS to bboxes and scores
            if cfg.get('use_rotate_nms', True):
                nms_thr = cfg.get('nms_thr', 0.01)
                nms_type = cfg.get('nms_type', 'default')
                
                # Store original indices for later use
                original_indices = torch.arange(len(bboxes), device=bboxes.device)
                
                # Apply NMS
                bboxes, scores, keep_indices = self._rotate_nms(bboxes, scores, nms_thr, nms_type)
                
                # Filter labels using the same indices
                if len(bboxes) > 0:
                    labels = labels[keep_indices]
                else:
                    # If no boxes after NMS, create empty labels
                    labels = torch.zeros(0, dtype=torch.long, device=bboxes.device)
            
            # Convert bboxes to LiDARInstance3DBoxes
            if len(bboxes) > 0:
                # Ensure bboxes have the correct format for KITTI
                # KITTI format: (x, y, z, w, l, h, theta)
                # Make sure theta is in the correct range [-pi, pi]
                bboxes = bboxes.clone()
                bboxes[:, 6] = torch.atan2(torch.sin(bboxes[:, 6]), torch.cos(bboxes[:, 6]))
                
                # Create LiDARInstance3DBoxes with the correct origin
                bboxes = LiDARInstance3DBoxes(bboxes, box_dim=7, origin=(0.5, 0.5, 0.5))
                
                # Apply limit_yaw to normalize the heading angles
                bboxes.limit_yaw()
            else:
                # Create empty LiDARInstance3DBoxes
                bboxes = LiDARInstance3DBoxes(torch.zeros((0, 7), device=bboxes.device), 
                                             box_dim=7, origin=(0.5, 0.5, 0.5))
            
            # Create result
            result = InstanceData()
            result.bboxes_3d = bboxes
            result.scores_3d = scores
            
            # Set labels based on the class with the highest score
            result.labels_3d = labels
            
            results.append(result)
        
        return results
    
    def _rotate_nms(self, bboxes, scores, nms_thr, nms_type='default'):
        """Rotated NMS for 3D boxes.
        
        Args:
            bboxes (Tensor): 3D boxes of shape (N, 7) with format (x, y, z, w, l, h, theta).
            scores (Tensor): Scores of shape (N,).
            nms_thr (float): IoU threshold for NMS.
            nms_type (str): Type of NMS to use. Options are 'default', 'rotated', 'sparse'.
            
        Returns:
            tuple[Tensor]: Filtered boxes, scores, and keep indices.
        """
        if bboxes.shape[0] == 0:
            return bboxes, scores, torch.zeros(0, dtype=torch.long, device=bboxes.device)
            
        if nms_type == 'default':
            # Use the default implementation (no filtering)
            keep_indices = torch.arange(len(bboxes), device=bboxes.device)
            return bboxes, scores, keep_indices
        elif nms_type == 'rotated':
            return self._rotated_nms_impl(bboxes, scores, nms_thr)
        elif nms_type == 'sparse':
            return self._sparse_nms_impl(bboxes, scores, nms_thr)
        else:
            raise ValueError(f'Unknown NMS type: {nms_type}')
    
    def _rotated_nms_impl(self, bboxes, scores, nms_thr):
        """Implementation of rotated 3D NMS.
        
        Args:
            bboxes (Tensor): 3D boxes of shape (N, 7) with format (x, y, z, w, l, h, theta).
            scores (Tensor): Scores of shape (N,).
            nms_thr (float): IoU threshold for NMS.
            
        Returns:
            tuple[Tensor]: Filtered boxes, scores, and keep indices.
        """
        # Sort by score in descending order
        _, order = scores.sort(0, descending=True)
        bboxes = bboxes[order]
        scores = scores[order]
        
        # Initialize keep mask
        keep = torch.zeros_like(scores, dtype=torch.bool)
        keep[0] = True  # Keep the highest scoring box
        
        # Calculate IoU between boxes
        for i in range(1, bboxes.shape[0]):
            # Calculate IoU with all previously kept boxes
            ious = self._box3d_iou(bboxes[i:i+1], bboxes[keep])
            
            # If max IoU is below threshold, keep this box
            if ious.max() < nms_thr:
                keep[i] = True
        
        # Get the indices of the kept boxes in the original order
        keep_indices = order[keep]
        
        # Return kept boxes, scores, and keep indices
        return bboxes[keep], scores[keep], keep_indices
    
    def _sparse_nms_impl(self, bboxes, scores, nms_thr):
        """Implementation of sparse-aware 3D NMS.
        
        Args:
            bboxes (Tensor): 3D boxes of shape (N, 7) with format (x, y, z, w, l, h, theta).
            scores (Tensor): Scores of shape (N,).
            nms_thr (float): IoU threshold for NMS.
            
        Returns:
            tuple[Tensor]: Filtered boxes, scores, and keep indices.
        """
        # This is a simplified implementation of sparse NMS
        # In practice, you would need to implement this based on your specific requirements
        
        # Sort by score in descending order
        _, order = scores.sort(0, descending=True)
        bboxes = bboxes[order]
        scores = scores[order]
        
        # Initialize keep mask
        keep = torch.zeros_like(scores, dtype=torch.bool)
        keep[0] = True  # Keep the highest scoring box
        
        # Group boxes by spatial proximity
        # This is a simplified approach - in practice, you would use a more sophisticated
        # spatial hashing or clustering approach
        spatial_groups = []
        for i in range(bboxes.shape[0]):
            if keep[i]:
                # Find boxes in the same spatial group
                group = [i]
                for j in range(i+1, bboxes.shape[0]):
                    if not keep[j]:
                        # Calculate distance between box centers
                        dist = torch.norm(bboxes[i, :3] - bboxes[j, :3])
                        if dist < 2.0:  # Threshold for spatial proximity
                            group.append(j)
                            keep[j] = True
                spatial_groups.append(group)
        
        # Apply NMS within each spatial group
        final_keep = torch.zeros_like(scores, dtype=torch.bool)
        for group in spatial_groups:
            if len(group) == 1:
                final_keep[group[0]] = True
            else:
                group_bboxes = bboxes[group]
                group_scores = scores[group]
                
                # Apply NMS within the group
                group_keep = torch.zeros_like(group_scores, dtype=torch.bool)
                group_keep[0] = True  # Keep the highest scoring box
                
                for i in range(1, len(group)):
                    # Calculate IoU with all previously kept boxes in the group
                    ious = self._box3d_iou(group_bboxes[i:i+1], group_bboxes[group_keep])
                    
                    # If max IoU is below threshold, keep this box
                    if ious.max() < nms_thr:
                        group_keep[i] = True
                
                # Update final keep mask
                for i, idx in enumerate(group):
                    if group_keep[i]:
                        final_keep[idx] = True
        
        # Get the indices of the kept boxes in the original order
        keep_indices = order[final_keep]
        
        # Return kept boxes, scores, and keep indices
        return bboxes[final_keep], scores[final_keep], keep_indices
    
    def _box3d_iou(self, box1, box2):
        """Calculate 3D IoU between two 3D boxes.
        
        Args:
            box1 (Tensor): 3D boxes of shape (N, 7) with format (x, y, z, w, l, h, theta).
            box2 (Tensor): 3D boxes of shape (M, 7) with format (x, y, z, w, l, h, theta).
            
        Returns:
            Tensor: IoU of shape (N, M).
        """
        # Extract box parameters
        x1, y1, z1, w1, l1, h1, theta1 = box1.unbind(-1)
        x2, y2, z2, w2, l2, h2, theta2 = box2.unbind(-1)
        
        # Calculate volume
        vol1 = w1 * l1 * h1
        vol2 = w2 * l2 * h2
        
        # Calculate intersection
        # This is a simplified version - for accurate results, you need a more sophisticated
        # intersection calculation that handles rotated boxes
        x_overlap = torch.min(x1.unsqueeze(1) + w1.unsqueeze(1)/2, x2 + w2/2) - \
                    torch.max(x1.unsqueeze(1) - w1.unsqueeze(1)/2, x2 - w2/2)
        y_overlap = torch.min(y1.unsqueeze(1) + l1.unsqueeze(1)/2, y2 + l2/2) - \
                    torch.max(y1.unsqueeze(1) - l1.unsqueeze(1)/2, y2 - l2/2)
        z_overlap = torch.min(z1.unsqueeze(1) + h1.unsqueeze(1)/2, z2 + h2/2) - \
                    torch.max(z1.unsqueeze(1) - h1.unsqueeze(1)/2, z2 - h2/2)
        
        # Clamp negative values to 0
        x_overlap = torch.clamp(x_overlap, min=0)
        y_overlap = torch.clamp(y_overlap, min=0)
        z_overlap = torch.clamp(z_overlap, min=0)
        
        # Calculate intersection volume
        intersection = x_overlap * y_overlap * z_overlap
        
        # Calculate IoU
        union = vol1.unsqueeze(1) + vol2 - intersection
        iou = intersection / (union + 1e-6)
        
        return iou 
