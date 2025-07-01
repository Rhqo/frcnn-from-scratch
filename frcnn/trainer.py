
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from frcnn.models.faster_rcnn import FasterRCNN
from frcnn.losses import smooth_l1_loss, cross_entropy_loss
from frcnn.utils.bbox_tools import bbox2loc, bbox_iou
from frcnn.utils.anchors import generate_anchors, generate_base_anchors


class FasterRCNNTrainer(nn.Module):
    """
    Manages the training of the Faster R-CNN model.

    This class handles the forward pass, loss calculation, and target assignment
    for both the RPN and the Fast R-CNN head.

    Args:
        faster_rcnn (FasterRCNN): The Faster R-CNN model to train.
        optimizer (torch.optim.Optimizer): The optimizer for training.
    """

    def __init__(self, faster_rcnn: FasterRCNN, optimizer: optim.Optimizer):
        super(FasterRCNNTrainer, self).__init__()
        self.faster_rcnn = faster_rcnn
        self.optimizer = optimizer

        # Loss functions
        self.rpn_cls_loss_func = cross_entropy_loss
        self.rpn_reg_loss_func = smooth_l1_loss
        self.roi_cls_loss_func = cross_entropy_loss
        self.roi_reg_loss_func = smooth_l1_loss

        # RPN parameters (from FasterRCNN model)
        self.feat_stride = faster_rcnn.feat_stride
        self.n_anchor = faster_rcnn.rpn.n_anchor

    def forward(self, imgs, bboxes, labels, scale):
        """
        Forward pass for training.

        Args:
            imgs (torch.Tensor): Input image batch.
            bboxes (torch.Tensor): Ground truth bounding boxes for each image.
            labels (torch.Tensor): Ground truth labels for each bounding box.
            scale (float): Image scale factor.

        Returns:
            tuple: Tuple of losses (rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss).
        """
        n = imgs.shape[0]
        if n != 1:
            raise ValueError("Batch size must be 1 for now.")

        img_size = imgs.shape[2:]

        # 1. Feature Extraction
        feature_map = self.faster_rcnn.extractor(imgs)

        # 2. RPN forward pass
        rpn_cls_scores, rpn_bbox_preds = self.faster_rcnn.rpn(feature_map)

        # 3. Generate anchors
        base_anchors = generate_base_anchors()
        feat_height, feat_width = feature_map.shape[2], feature_map.shape[3]
        anchors = torch.from_numpy(generate_anchors(base_anchors, self.feat_stride, feat_height, feat_width)).float().to(imgs.device)

        # --- RPN Target Assignment ---
        # Convert ground truth boxes to numpy for IoU calculation
        bbox = bboxes[0] # Assuming batch size 1
        label = labels[0] # Assuming batch size 1

        # Calculate IoU between anchors and ground truth boxes
        # iou: (num_anchors, num_gt_boxes)
        iou = bbox_iou(anchors, bbox)

        # Assign labels to anchors
        # -1: ignore, 0: background, 1: foreground
        gt_max_iou = iou.max(axis=1).values # IoU of each anchor with its best matching gt_box
        gt_max_iou_argmax = iou.argmax(axis=1) # Index of the best matching gt_box for each anchor

        anchor_labels = torch.ones((anchors.shape[0],), dtype=torch.long, device=imgs.device) * -1 # Initialize all to ignore

        # Assign background labels (IoU < 0.3)
        anchor_labels[gt_max_iou < 0.3] = 0

        # Assign foreground labels (IoU > 0.7 or anchor is best match for any gt_box)
        anchor_labels[gt_max_iou >= 0.7] = 1
        
        # For each ground truth box, assign its best matching anchor as foreground
        gt_max_iou_anchor = iou.max(dim=0).values # IoU of each gt_box with its best matching anchor
        gt_max_iou_anchor_argmax = iou.argmax(dim=0) # Index of the best matching anchor for each gt_box
        anchor_labels[gt_max_iou_anchor_argmax] = 1

        # Sample anchors to balance foreground/background
        # Max 256 anchors per image, 128 foreground, 128 background
        n_pos = 128
        pos_index = torch.where(anchor_labels == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = pos_index[torch.randperm(len(pos_index))[:len(pos_index) - n_pos]]
            anchor_labels[disable_index] = -1

        n_neg = 256 - torch.sum(anchor_labels == 1)
        neg_index = torch.where(anchor_labels == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = neg_index[torch.randperm(len(neg_index))[:len(neg_index) - n_neg]]
            anchor_labels[disable_index] = -1

        # Calculate RPN regression targets
        # For foreground anchors, calculate bbox regression targets relative to their matched gt_box
        rpn_loc_targets = bbox2loc(anchors, bbox[gt_max_iou_argmax])

        # Create weights for RPN regression loss
        # Only foreground anchors contribute to regression loss
        rpn_loc_weights = torch.zeros(anchors.shape, dtype=torch.float32, device=imgs.device)
        rpn_loc_weights[anchor_labels == 1, :] = 1

        # Reshape RPN outputs for loss calculation
        rpn_cls_scores_2d = rpn_cls_scores.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        rpn_bbox_preds_2d = rpn_bbox_preds.permute(0, 2, 3, 1).contiguous().view(-1, 4)

        # RPN losses
        rpn_cls_loss = self.rpn_cls_loss_func(rpn_cls_scores_2d, anchor_labels)
        rpn_loc_loss = self.rpn_reg_loss_func(
            rpn_bbox_preds_2d, rpn_loc_targets, rpn_loc_weights, rpn_loc_weights
        )

        # 4. Generate proposals from RPN outputs
        # Set model to eval mode for proposal generation to use test NMS parameters
        self.faster_rcnn.eval()
        rois = self.faster_rcnn.proposal_layer(rpn_cls_scores.detach(), rpn_bbox_preds.detach(), img_size)
        self.faster_rcnn.train() # Set back to train mode

        # --- Fast R-CNN Target Assignment ---
        # Sample RoIs and assign targets for Fast R-CNN
        sample_rois, gt_roi_loc_targets, gt_roi_labels = self._proposal_target_creator(
            rois, bboxes[0], labels[0], self.faster_rcnn.num_classes
        )

        # Forward pass through Fast R-CNN head with sampled RoIs
        roi_cls_scores, roi_bbox_preds = self.faster_rcnn.head(
            self.faster_rcnn.roi_pooling(feature_map, sample_rois)
        )

        # RoI losses
        roi_cls_loss = self.roi_cls_loss_func(roi_cls_scores, gt_roi_labels)
        
        # Only positive samples contribute to regression loss
        n_pos = torch.sum(gt_roi_labels > 0).item()
        roi_loc_targets = gt_roi_loc_targets[:n_pos]
        roi_bbox_preds = roi_bbox_preds.view(-1, self.faster_rcnn.num_classes, 4)
        roi_bbox_preds = roi_bbox_preds[torch.arange(n_pos), gt_roi_labels[:n_pos]]

        roi_loc_loss = self.roi_reg_loss_func(
            roi_bbox_preds, roi_loc_targets,
            torch.ones_like(roi_loc_targets), torch.ones_like(roi_loc_targets)
        )

        return rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss

    def _proposal_target_creator(self, rois, gt_bbox, gt_label, num_classes,
                                 n_sample=128, pos_ratio=0.25, pos_iou_thresh=0.5, neg_iou_thresh_high=0.5, neg_iou_thresh_low=0.0):
        """
        Samples RoIs and assigns targets for the Fast R-CNN head.

        Args:
            rois (torch.Tensor): Region proposals from RPN (n_proposals, 5).
            gt_bbox (torch.Tensor): Ground truth bounding boxes (n_gt, 4).
            gt_label (torch.Tensor): Ground truth labels (n_gt,).
            num_classes (int): Total number of classes including background.
            n_sample (int): Total number of RoIs to sample.
            pos_ratio (float): Ratio of positive samples in the sampled RoIs.
            pos_iou_thresh (float): IoU threshold for positive samples.
            neg_iou_thresh_high (float): Upper IoU threshold for negative samples.
            neg_iou_thresh_low (float): Lower IoU threshold for negative samples.

        Returns:
            tuple:
                - sample_rois (torch.Tensor): Sampled RoIs (n_sample, 5).
                - gt_roi_loc_targets (torch.Tensor): Ground truth regression targets for sampled positive RoIs.
                - gt_roi_labels (torch.Tensor): Ground truth labels for sampled RoIs.
        """
        # Calculate IoU between proposals and ground truth boxes
        # iou: (num_rois, num_gt_boxes)
        iou = bbox_iou(rois[:, 1:], gt_bbox) # Exclude batch index from rois for IoU calculation

        # For each RoI, find the ground truth box with the maximum IoU
        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1).values

        # Assign labels to RoIs
        # Initialize all labels to background (0)
        labels = torch.zeros((rois.shape[0],), dtype=torch.long, device=rois.device)
        # Assign foreground labels (1 to num_classes-1)
        labels[max_iou >= pos_iou_thresh] = gt_label[gt_assignment[max_iou >= pos_iou_thresh]]

        # Sample positive RoIs
        pos_index = torch.where(max_iou >= pos_iou_thresh)[0]
        n_pos_actual = int(n_sample * pos_ratio)
        if len(pos_index) > n_pos_actual:
            disable_index = pos_index[torch.randperm(len(pos_index))[:len(pos_index) - n_pos_actual]]
            labels[disable_index] = 0 # Change to background if too many positive

        # Sample negative RoIs
        neg_index = torch.where((max_iou < neg_iou_thresh_high) & (max_iou >= neg_iou_thresh_low))[0]
        n_neg_actual = n_sample - torch.sum(labels > 0).item() # Remaining slots for negative samples
        if len(neg_index) > n_neg_actual:
            disable_index = neg_index[torch.randperm(len(neg_index))[:len(neg_index) - n_neg_actual]]
            labels[disable_index] = -1 # Ignore if too many negative

        # Select sampled RoIs
        keep_index = torch.where(labels != -1)[0]
        sample_rois = rois[keep_index]
        gt_roi_labels = labels[keep_index]

        # Calculate regression targets for positive samples
        pos_sample_rois = sample_rois[gt_roi_labels > 0]
        pos_gt_bbox = gt_bbox[gt_assignment[keep_index][gt_roi_labels > 0]]
        gt_roi_loc_targets = bbox2loc(pos_sample_rois[:, 1:], pos_gt_bbox)

        return sample_rois, gt_roi_loc_targets, gt_roi_labels
