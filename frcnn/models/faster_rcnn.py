
import torch
import torch.nn as nn
import torchvision.ops as ops

from frcnn.models.resnet50 import get_resnet50_base_net
from frcnn.utils.bbox_tools import loc2bbox
from frcnn.models.rpn import RPN
from frcnn.models.roi_pooling import RoIPooling
from frcnn.models.proposal_layer import ProposalLayer
from frcnn.models.fast_rcnn_head import FastRCNNHead


class FasterRCNN(nn.Module):
    """
    The complete Faster R-CNN model.

    This model integrates the VGG16 base network, the Region Proposal Network (RPN),
    the RoI Pooling layer, and the Fast R-CNN head.

    Args:
        num_classes (int): Number of object classes (including background).
        feat_stride (int): The stride of the feature map relative to the input image.
        rpn_nms_thresh (float): The NMS threshold for RPN proposals.
        rpn_train_pre_nms (int): Number of top RPN proposals to keep before NMS during training.
        rpn_train_post_nms (int): Number of top RPN proposals to keep after NMS during training.
        rpn_test_pre_nms (int): Number of top RPN proposals to keep before NMS during testing.
        rpn_test_post_nms (int): Number of top RPN proposals to keep after NMS during testing.
        roi_output_size (tuple): The fixed output size of the RoI Pooling layer (height, width).
    """

    def __init__(self, num_classes=21, feat_stride=16,
                 rpn_nms_thresh=0.7, rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
                 rpn_test_pre_nms=6000, rpn_test_post_nms=300,
                 roi_output_size=(7, 7)):
        super(FasterRCNN, self).__init__()

        self.num_classes = num_classes
        self.feat_stride = feat_stride

        # 1. Base Network (Feature Extractor)
        self.extractor = get_resnet50_base_net()

        # 2. Region Proposal Network (RPN)
        self.rpn = RPN(in_channels=2048, mid_channels=512, n_anchor=9)

        # 3. Proposal Layer (converts RPN outputs to proposals)
        self.proposal_layer = ProposalLayer(
            feat_stride=feat_stride, n_anchor=9, nms_thresh=rpn_nms_thresh,
            n_train_pre_nms=rpn_train_pre_nms, n_train_post_nms=rpn_train_post_nms,
            n_test_pre_nms=rpn_test_pre_nms, n_test_post_nms=rpn_test_post_nms
        )

        # 4. RoI Pooling Layer
        self.roi_pooling = RoIPooling(output_size=roi_output_size, spatial_scale=1.0 / feat_stride)

        # 5. Fast R-CNN Head
        self.head = FastRCNNHead(in_channels=2048, num_classes=num_classes, roi_output_size=roi_output_size)

    def forward(self, x: torch.Tensor, img_size: tuple):
        """
        Forward pass of the Faster R-CNN model.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, H, W).
            img_size (tuple): Original image size (height, width).

        Returns:
            List[Dict[str, torch.Tensor]]: A list of dictionaries, where each dictionary
                represents the detections for one image and contains:
                - 'boxes' (torch.Tensor): Bounding box coordinates in (x1, y1, x2, y2) format.
                - 'labels' (torch.Tensor): Predicted class labels.
                - 'scores' (torch.Tensor): Prediction scores.
        """
        # 1. Feature Extraction
        feature_map = self.extractor(x)

        # 2. RPN forward pass
        rpn_cls_scores, rpn_bbox_preds = self.rpn(feature_map)

        # 3. Generate proposals from RPN outputs
        rois = self.proposal_layer(rpn_cls_scores, rpn_bbox_preds, img_size)

        # 4. RoI Pooling
        pooled_features = self.roi_pooling(feature_map, rois)

        # 5. Fast R-CNN Head forward pass
        roi_cls_scores, roi_bbox_preds = self.head(pooled_features)

        # Post-processing for evaluation/inference
        # Apply softmax to RoI classification scores
        roi_probs = torch.softmax(roi_cls_scores, dim=1)
        
        # Get the predicted class labels and scores
        scores, labels = roi_probs.max(dim=1)
        
        # Remove background class (label 0)
        # Keep only foreground classes
        foreground_indices = (labels != 0).nonzero(as_tuple=True)[0]
        
        if foreground_indices.numel() == 0:
            # No foreground objects detected
            return [{
                'boxes': torch.empty((0, 4), dtype=torch.float32, device=x.device),
                'labels': torch.empty((0,), dtype=torch.int64, device=x.device),
                'scores': torch.empty((0,), dtype=torch.float32, device=x.device)
            }]

        scores = scores[foreground_indices]
        labels = labels[foreground_indices]
        roi_bbox_preds = roi_bbox_preds[foreground_indices]
        rois = rois[foreground_indices] # Filter rois as well

        # Convert rois from (x1, y1, x2, y2) to (x, y, w, h)
        rois_xywh = torch.stack([
            rois[:, 0],
            rois[:, 1],
            rois[:, 2] - rois[:, 0],
            rois[:, 3] - rois[:, 1]
        ], dim=1)

        # Select the bbox_preds corresponding to the predicted labels
        selected_roi_bbox_preds = torch.empty((foreground_indices.numel(), 4), dtype=torch.float32, device=x.device)
        for i, label_idx in enumerate(labels):
            selected_roi_bbox_preds[i] = roi_bbox_preds[i, label_idx * 4 : (label_idx + 1) * 4]

        # Decode bounding box predictions
        decoded_boxes = loc2bbox(rois_xywh, selected_roi_bbox_preds)
        
        # Clip boxes to image boundaries
        img_h, img_w = img_size
        decoded_boxes[:, 0::2] = decoded_boxes[:, 0::2].clamp(min=0, max=img_w)
        decoded_boxes[:, 1::2] = decoded_boxes[:, 1::2].clamp(min=0, max=img_h)

        # Apply NMS
        keep = ops.nms(decoded_boxes, scores, iou_threshold=0.5)

        final_boxes = decoded_boxes[keep]
        final_labels = labels[keep]
        final_scores = scores[keep]
        return [{
            'boxes': final_boxes,
            'labels': final_labels,
            'scores': final_scores
        }]
