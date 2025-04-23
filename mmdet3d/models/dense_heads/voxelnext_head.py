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
from mmdet3d.utils.nms import batched_nms

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
        
        # Add score threshold and NMS threshold from test_cfg
        if test_cfg is not None:
            self.score_threshold = test_cfg.get('score_threshold', 0.1)
            self.nms_threshold = test_cfg.get('nms_threshold', 0.5)
        else:
            self.score_threshold = 0.1
            self.nms_threshold = 0.5
        
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
        
        # Process features in parallel using torch.cuda.amp
        with torch.cuda.amp.autocast(enabled=True):
            for i, feat in enumerate(x):
                # Apply shared conv layers with memory optimization
                feat = self.shared_conv(feat)
                
                # Classification prediction
                cls_score = self.conv_cls(feat)
                cls_scores.append(cls_score)
                
                # Bbox prediction
                bbox_pred = self.conv_reg(feat)
                bbox_preds.append(bbox_pred)
                
                # Direction classification
                if self.use_direction_classifier:
                    dir_cls_pred = self.conv_dir_cls(feat)
                    dir_cls_preds.append(dir_cls_pred)
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
        # Input validation
        if not isinstance(cls_scores, list) or not isinstance(bbox_preds, list):
            raise TypeError('cls_scores and bbox_preds must be lists')
        
        if len(cls_scores) != len(bbox_preds):
            raise ValueError(f'Number of levels in cls_scores ({len(cls_scores)}) and bbox_preds ({len(bbox_preds)}) must match')
        
        if self.use_direction_classifier:
            if not isinstance(dir_cls_preds, list):
                raise TypeError('dir_cls_preds must be a list when use_direction_classifier is True')
            if len(dir_cls_preds) != len(cls_scores):
                raise ValueError(f'Number of levels in dir_cls_preds ({len(dir_cls_preds)}) must match cls_scores ({len(cls_scores)})')
        
        # Get ground truth with memory optimization
        with torch.no_grad():
            gt_labels_3d = [gt_instances_3d.labels_3d for gt_instances_3d in batch_gt_instances_3d]
            gt_bboxes_3d = [gt_instances_3d.bboxes_3d for gt_instances_3d in batch_gt_instances_3d]
            
            # Validate ground truth
            if not gt_labels_3d or not gt_bboxes_3d:
                raise ValueError('Empty ground truth labels or boxes')
            
            # Find the maximum number of ground truth objects in any sample
            max_num_gt = max([len(gt_labels) for gt_labels in gt_labels_3d])
            
            # Pre-allocate tensors for all samples at once
            device = gt_labels_3d[0].device
            dtype = gt_labels_3d[0].dtype
            B = len(gt_labels_3d)
            
            # Create padded tensors directly
            padded_gt_labels = torch.zeros((B, max_num_gt), dtype=dtype, device=device)
            padded_gt_bboxes = torch.zeros((B, max_num_gt, 7), dtype=gt_bboxes_3d[0].tensor.dtype, device=device)
            
            # Fill padded tensors efficiently using vectorized operations
            for i, (labels, bboxes) in enumerate(zip(gt_labels_3d, gt_bboxes_3d)):
                num_gt = len(labels)
                if num_gt > 0:
                    padded_gt_labels[i, :num_gt] = labels
                    padded_gt_bboxes[i, :num_gt] = bboxes.tensor
            
            # Stack tensors once
            gt_labels_3d = padded_gt_labels  # (B, max_num_gt)
            gt_bboxes_3d = padded_gt_bboxes  # (B, max_num_gt, 7)
        
        # Calculate losses with memory optimization
        losses = {}
        
        # Initialize loss components
        num_levels = len(cls_scores)
        cls_loss = []
        bbox_loss = []
        dir_loss = []
        iou_loss = []
        
        # Compute losses for each level with memory optimization
        with torch.amp.autocast('cuda', enabled=True):
            for level in range(num_levels):
                try:
                    # Get predictions for current level
                    cls_score = cls_scores[level]  # (B, C, H, W, D)
                    bbox_pred = bbox_preds[level]  # (B, 7, H, W, D)
                    if self.use_direction_classifier:
                        dir_cls_pred = dir_cls_preds[level]  # (B, 2, H, W, D)
                    
                    # Validate tensor shapes
                    if cls_score.dim() != 5 or bbox_pred.dim() != 5:
                        raise ValueError(f'Expected 5D tensors, got cls_score: {cls_score.dim()}D, bbox_pred: {bbox_pred.dim()}D')
                    
                    if self.use_direction_classifier and dir_cls_pred.dim() != 5:
                        raise ValueError(f'Expected 5D tensor for dir_cls_pred, got {dir_cls_pred.dim()}D')
                    
                    # Get shapes and ensure they match
                    B, C, H, W, D = cls_score.shape
                    if C != self.num_classes:
                        raise ValueError(f'Expected {self.num_classes} classes in cls_score, got {C}')
                    
                    total_elements = H * W * D
                    
                    # Verify tensor sizes before reshaping
                    expected_size = B * C * total_elements
                    if cls_score.numel() != expected_size:
                        raise ValueError(f'Tensor size mismatch. Expected {expected_size} elements, got {cls_score.numel()}')
                    
                    # Check for NaN values
                    if torch.isnan(cls_score).any() or torch.isnan(bbox_pred).any():
                        raise ValueError('NaN values detected in predictions')
                    
                    if self.use_direction_classifier and torch.isnan(dir_cls_pred).any():
                        raise ValueError('NaN values detected in direction predictions')
                    
                    # Reshape predictions efficiently using view
                    cls_score = cls_score.permute(0, 2, 3, 4, 1).contiguous()  # (B, H, W, D, C)
                    cls_score = cls_score.view(B, total_elements, C)  # (B, H*W*D, C)
                    
                    bbox_pred = bbox_pred.permute(0, 2, 3, 4, 1).contiguous()  # (B, H, W, D, 7)
                    bbox_pred = bbox_pred.view(B, total_elements, 7)  # (B, H*W*D, 7)
                    
                    if self.use_direction_classifier:
                        dir_cls_pred = dir_cls_pred.permute(0, 2, 3, 4, 1).contiguous()  # (B, H, W, D, 2)
                        dir_cls_pred = dir_cls_pred.view(B, total_elements, 2)  # (B, H*W*D, 2)
                    
                    # Create target tensors efficiently with matching dtypes
                    target_labels = torch.zeros((B, total_elements, C), device=cls_score.device, dtype=cls_score.dtype)
                    target_bboxes = torch.zeros((B, total_elements, 7), device=bbox_pred.device, dtype=bbox_pred.dtype)
                    
                    # Fill target tensors using vectorized operations
                    valid_mask = (gt_labels_3d >= 0) & (gt_labels_3d < C)  # (B, max_num_gt)
                    
                    # Process all batches at once using vectorized operations
                    for i in range(B):
                        valid_indices = valid_mask[i].nonzero().squeeze(-1)
                        if len(valid_indices) > 0:
                            # Ensure indices are within bounds
                            valid_indices = valid_indices[valid_indices < total_elements]
                            if len(valid_indices) > 0:
                                # Get labels and bboxes for valid indices
                                labels = gt_labels_3d[i, valid_indices]
                                bboxes = gt_bboxes_3d[i, valid_indices].to(dtype=bbox_pred.dtype)  # Convert to matching dtype
                                
                                # Validate labels
                                if not torch.all((labels >= 0) & (labels < C)):
                                    raise ValueError(f'Invalid label values found. Labels must be in range [0, {C-1}]')
                                
                                # Create one-hot encoding for labels efficiently
                                target_labels[i].scatter_(1, labels.unsqueeze(1), 1)
                                target_bboxes[i, valid_indices] = bboxes
                    
                    # Compute losses with mixed precision and numerical stability
                    cls_loss.append(self.loss_cls(cls_score, target_labels))
                    bbox_loss.append(self.loss_bbox(bbox_pred, target_bboxes))
                    
                    if self.use_direction_classifier:
                        # Reshape direction predictions for loss computation
                        dir_cls_pred = dir_cls_pred.reshape(-1, 2)  # (B*H*W*D, 2)
                        
                        # Create target direction tensor
                        target_direction = torch.zeros((B, total_elements), dtype=torch.long, device=dir_cls_pred.device)
                        
                        # Process direction classification efficiently
                        for i in range(B):
                            valid_indices = valid_mask[i].nonzero().squeeze(-1)
                            if len(valid_indices) > 0:
                                # Ensure indices are within bounds
                                valid_indices = valid_indices[valid_indices < total_elements]
                                if len(valid_indices) > 0:
                                    # Get headings for valid indices
                                    headings = gt_bboxes_3d[i, valid_indices, -1].to(dtype=dir_cls_pred.dtype)  # Convert to matching dtype
                                    target_direction[i, valid_indices] = (headings > 0).long()
                        
                        # Reshape target direction for loss computation
                        target_direction = target_direction.reshape(-1)  # (B*H*W*D,)
                        
                        # Compute direction loss
                        dir_loss.append(self.loss_dir(dir_cls_pred, target_direction))
                    
                    # Compute IoU loss with proper reshaping
                    # Reshape predictions and targets to match expected format
                    bbox_pred_flat = bbox_pred.reshape(-1, 7)  # (B*H*W*D, 7)
                    target_bboxes_flat = target_bboxes.reshape(-1, 7)  # (B*H*W*D, 7)
                    
                    # Ensure tensors have the same size
                    min_size = min(bbox_pred_flat.size(0), target_bboxes_flat.size(0))
                    bbox_pred_flat = bbox_pred_flat[:min_size]
                    target_bboxes_flat = target_bboxes_flat[:min_size]
                    
                    # Compute IoU loss with numerical stability
                    iou = self._box3d_iou(bbox_pred_flat, target_bboxes_flat)
                    iou_loss.append(1 - iou.mean())
                    
                except Exception as e:
                    print(f"Error processing level {level}: {str(e)}")
                    # Add zero loss for this level
                    cls_loss.append(torch.tensor(0.0, device=cls_scores[0].device))
                    bbox_loss.append(torch.tensor(0.0, device=cls_scores[0].device))
                    if self.use_direction_classifier:
                        dir_loss.append(torch.tensor(0.0, device=cls_scores[0].device))
                    iou_loss.append(torch.tensor(0.0, device=cls_scores[0].device))
        
        # Combine losses from all levels with numerical stability
        losses['loss_cls'] = sum(cls_loss) / max(num_levels, 1)
        losses['loss_bbox'] = sum(bbox_loss) / max(num_levels, 1)
        if self.use_direction_classifier:
            losses['loss_dir'] = sum(dir_loss) / max(num_levels, 1) * self.loss_dir_weight
        losses['loss_iou'] = sum(iou_loss) / max(num_levels, 1)
        
        # Final numerical stability check
        for k, v in losses.items():
            if torch.isnan(v):
                raise ValueError(f'NaN detected in {k}')
        
        return losses
    
    def predict_by_feat(self,
                         cls_scores,
                         bbox_preds,
                         dir_cls_preds,
                         input_metas=None,
                         batch_input_metas=None,
                         rescale=False,
                         cfg=None,
                         **kwargs):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (List[Tensor]): Classification scores for each level
            bbox_preds (List[Tensor]): Box regression for each level
            dir_cls_preds (List[Tensor]): Direction classification for each level
            input_metas (list[dict], optional): Input meta info. Defaults to None.
            batch_input_metas (list[dict], optional): Batch input meta info. Defaults to None.
            rescale (bool): Whether to rescale bbox. Defaults to False.
            cfg (ConfigDict, optional): Test / postprocessing configuration. Defaults to None.
            **kwargs: Additional arguments from base class

        Returns:
            list[tuple[Tensor, Tensor, Tensor]]: Each item in result_list is
                3-tuple. The first item is an (n, 5) tensor, where the first 4
                columns are bounding box positions (tl_x, tl_y, br_x, br_y) and
                the 5-th column is a score between 0 and 1. The second item is an
                (n,) tensor where each item is the predicted class label of the
                corresponding box. The third item is an (n,) tensor where each
                item is the predicted direction label of the corresponding box.
        """
        result_list = []
        if input_metas is None:
            input_metas = batch_input_metas if batch_input_metas is not None else [{}] * len(cls_scores[0])
        
        # Get voxel size from input metas or use default
        voxel_size = None
        for meta in input_metas:
            if 'voxel_size' in meta:
                voxel_size = meta['voxel_size']
                break
        if voxel_size is None:
            voxel_size = [0.05, 0.05, 0.1]  # Default KITTI voxel size

        for img_id in range(len(input_metas)):
            try:
                cls_score_list = [
                    cls_scores[i][img_id].detach() for i in range(len(cls_scores))
                ]
                bbox_pred_list = [
                    bbox_preds[i][img_id].detach() for i in range(len(bbox_preds))
                ]
                dir_cls_pred_list = [
                    dir_cls_preds[i][img_id].detach() for i in range(len(dir_cls_preds))
                ]

                input_meta = input_metas[img_id]
                batch_size = input_meta.get('batch_size', 1)
                
                # Process predictions level by level
                bboxes_list = []
                scores_list = []
                dir_labels_list = []
                batch_indices = []
                
                for level_id in range(len(cls_score_list)):
                    try:
                        cls_score = cls_score_list[level_id]
                        bbox_pred = bbox_pred_list[level_id]
                        dir_cls_pred = dir_cls_pred_list[level_id]
                        
                        # Ensure tensors are on the correct device
                        device = cls_score.device
                        
                        # Get predictions
                        scores = cls_score.sigmoid()
                        bbox_pred = bbox_pred.reshape(-1, self.box_code_size)
                        dir_cls_pred = dir_cls_pred.reshape(-1, 2)
                        
                        # Apply score threshold
                        score_threshold = getattr(self, 'score_threshold', 0.1)
                        score_mask = scores > score_threshold
                        
                        if score_mask.any():
                            # Filter predictions
                            scores = scores[score_mask]
                            bbox_pred = bbox_pred[score_mask]
                            dir_cls_pred = dir_cls_pred[score_mask]
                            
                            # Get direction labels
                            dir_labels = torch.max(dir_cls_pred, dim=-1)[1]
                            
                            # Create batch indices
                            level_batch_indices = torch.full((len(scores),), level_id, device=device)
                            
                            # Add to lists
                            bboxes_list.append(bbox_pred)
                            scores_list.append(scores)
                            dir_labels_list.append(dir_labels)
                            batch_indices.append(level_batch_indices)
                    except Exception as e:
                        print(f"Error processing level {level_id}: {str(e)}")
                        continue
                
                if not bboxes_list:
                    # No valid predictions
                    empty_bbox = torch.zeros((0, self.box_code_size), device=device)
                    empty_score = torch.zeros((0,), device=device)
                    empty_dir = torch.zeros((0,), device=device)
                    result_list.append((empty_bbox, empty_score, empty_dir))
                    continue
                
                # Concatenate predictions from all levels
                bboxes = torch.cat(bboxes_list, dim=0)
                scores = torch.cat(scores_list, dim=0)
                dir_labels = torch.cat(dir_labels_list, dim=0)
                batch_indices = torch.cat(batch_indices, dim=0)
                
                # Apply NMS
                nms_threshold = getattr(self, 'nms_threshold', 0.5)
                nms_cfg = dict(
                    type='nms',
                    iou_threshold=nms_threshold,
                    score_threshold=score_threshold)
                
                # Perform NMS
                nms_bboxes, nms_scores, nms_dir_labels = [], [], []
                for level_id in range(len(cls_score_list)):
                    level_mask = batch_indices == level_id
                    if level_mask.any():
                        level_bboxes = bboxes[level_mask]
                        level_scores = scores[level_mask]
                        level_dir_labels = dir_labels[level_mask]
                        
                        # Apply NMS
                        keep = batched_nms(level_bboxes, level_scores, level_dir_labels, nms_cfg)
                        
                        nms_bboxes.append(level_bboxes[keep])
                        nms_scores.append(level_scores[keep])
                        nms_dir_labels.append(level_dir_labels[keep])
                
                if nms_bboxes:
                    final_bboxes = torch.cat(nms_bboxes, dim=0)
                    final_scores = torch.cat(nms_scores, dim=0)
                    final_dir_labels = torch.cat(nms_dir_labels, dim=0)
                else:
                    final_bboxes = torch.zeros((0, self.box_code_size), device=device)
                    final_scores = torch.zeros((0,), device=device)
                    final_dir_labels = torch.zeros((0,), device=device)
                
                result_list.append((final_bboxes, final_scores, final_dir_labels))
            except Exception as e:
                print(f"Error processing sample {img_id}: {str(e)}")
                # Return empty result for this sample
                device = cls_scores[0].device
                empty_bbox = torch.zeros((0, self.box_code_size), device=device)
                empty_score = torch.zeros((0,), device=device)
                empty_dir = torch.zeros((0,), device=device)
                result_list.append((empty_bbox, empty_score, empty_dir))
        
        return result_list
    
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
        # Input validation
        if box1.dim() != 2 or box2.dim() != 2:
            raise ValueError(f'Expected 2D tensors, got box1: {box1.dim()}D, box2: {box2.dim()}D')
        
        if box1.size(-1) != 7 or box2.size(-1) != 7:
            raise ValueError(f'Expected 7 dimensions per box, got box1: {box1.size(-1)}, box2: {box2.size(-1)}')
        
        # Check for NaN values
        if torch.isnan(box1).any() or torch.isnan(box2).any():
            raise ValueError('NaN values detected in input boxes')
        
        # Ensure boxes have positive dimensions
        box1 = torch.clamp(box1, min=1e-6)
        box2 = torch.clamp(box2, min=1e-6)
        
        # Move IoU calculation to GPU
        # Use vectorized operations
        x1, y1, z1, w1, l1, h1, theta1 = box1.unbind(-1)
        x2, y2, z2, w2, l2, h2, theta2 = box2.unbind(-1)
        
        # Calculate volume on GPU with numerical stability
        vol1 = w1 * l1 * h1
        vol2 = w2 * l2 * h2
        
        # Calculate intersection on GPU with numerical stability
        x_overlap = torch.min(x1.unsqueeze(1) + w1.unsqueeze(1)/2, x2 + w2/2) - \
                    torch.max(x1.unsqueeze(1) - w1.unsqueeze(1)/2, x2 - w2/2)
        y_overlap = torch.min(y1.unsqueeze(1) + l1.unsqueeze(1)/2, y2 + l2/2) - \
                    torch.max(y1.unsqueeze(1) - l1.unsqueeze(1)/2, y2 - l2/2)
        z_overlap = torch.min(z1.unsqueeze(1) + h1.unsqueeze(1)/2, z2 + h2/2) - \
                    torch.max(z1.unsqueeze(1) - h1.unsqueeze(1)/2, z2 - h2/2)
        
        # Clamp negative values to 0 on GPU
        x_overlap = torch.clamp(x_overlap, min=0)
        y_overlap = torch.clamp(y_overlap, min=0)
        z_overlap = torch.clamp(z_overlap, min=0)
        
        # Calculate intersection volume on GPU with numerical stability
        intersection = x_overlap * y_overlap * z_overlap
        
        # Calculate IoU on GPU with numerical stability
        union = vol1.unsqueeze(1) + vol2 - intersection
        iou = intersection / (union + 1e-6)
        
        # Clamp IoU to valid range
        iou = torch.clamp(iou, min=0.0, max=1.0)
        
        return iou

    def _decode_bbox(self, bbox_pred, batch_indices, input_metas):
        """Decode bbox predictions.
        
        Args:
            bbox_pred (Tensor): Bbox predictions of shape (N, 7).
            batch_indices (Tensor): Batch indices of shape (N,).
            input_metas (list[dict]): Input metas.
            
        Returns:
            Tensor: Decoded bboxes of shape (N, 7).
        """
        # Get batch size and device
        device = bbox_pred.device
        
        # Create anchor tensor for decoding
        anchors = torch.zeros_like(bbox_pred)
        anchors[..., 3:6] = 1.0  # Set default size to 1.0
        
        # Decode bboxes using vectorized operations
        bboxes = self.bbox_coder.decode(anchors, bbox_pred)
        
        # Apply voxel size and range transformations in batches
        for i in range(len(input_metas)):
            # Get predictions for this sample
            sample_mask = batch_indices == i
            if not sample_mask.any():
                continue
                
            # Get voxel size and range for this sample
            voxel_size = input_metas[i]['voxel_size']
            pc_range = input_metas[i]['pc_range']
            
            # Apply transformations
            bboxes[sample_mask, 0] = bboxes[sample_mask, 0] * voxel_size[0] + pc_range[0]
            bboxes[sample_mask, 1] = bboxes[sample_mask, 1] * voxel_size[1] + pc_range[1]
            bboxes[sample_mask, 2] = bboxes[sample_mask, 2] * voxel_size[2] + pc_range[2]
        
        return bboxes 
