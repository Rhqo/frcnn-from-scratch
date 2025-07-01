
import torch
import torch.nn as nn
import numpy as np
from torchvision.ops import nms

from frcnn.utils.anchors import generate_anchors, generate_base_anchors


class ProposalLayer(nn.Module):
    """
    Converts RPN outputs (class scores and bounding box deltas) into object proposals.

    This layer performs the following steps:
    1. Generates anchors for the given feature map size.
    2. Applies the RPN's bounding box deltas to the anchors.
    3. Clips the proposals to the image boundaries.
    4. Removes proposals that are too small.
    5. Sorts proposals by their objectness score.
    6. Applies Non-Maximum Suppression (NMS) to remove redundant proposals.

    Args:
        feat_stride (int): The stride of the feature map relative to the input image.
        n_anchor (int): The number of anchors per feature map location.
        nms_thresh (float): The NMS threshold to use.
        n_train_pre_nms (int): Number of top proposals to keep before NMS during training.
        n_train_post_nms (int): Number of top proposals to keep after NMS during training.
        n_test_pre_nms (int): Number of top proposals to keep before NMS during testing.
        n_test_post_nms (int): Number of top proposals to keep after NMS during testing.
    """

    def __init__(self, feat_stride=16, n_anchor=9, nms_thresh=0.7, 
                 n_train_pre_nms=12000, n_train_post_nms=2000,
                 n_test_pre_nms=6000, n_test_post_nms=300):
        super(ProposalLayer, self).__init__()
        self.feat_stride = feat_stride
        self.n_anchor = n_anchor
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms

    def forward(self, rpn_cls_scores, rpn_bbox_preds, img_size):
        """
        Forward pass of the proposal layer.

        Args:
            rpn_cls_scores (torch.Tensor): RPN classification scores of shape 
                                           (batch_size, n_anchor * 2, height, width).
            rpn_bbox_preds (torch.Tensor): RPN bounding box predictions of shape
                                           (batch_size, n_anchor * 4, height, width).
            img_size (tuple): The size of the input image (height, width).

        Returns:
            torch.Tensor: The final region proposals of shape (n_proposals, 5).
                          Each row is (batch_index, x1, y1, x2, y2).
        """
        # Generate anchors
        base_anchors = generate_base_anchors()
        feat_height, feat_width = rpn_cls_scores.shape[2], rpn_cls_scores.shape[3]
        anchors = torch.from_numpy(generate_anchors(base_anchors, self.feat_stride, feat_height, feat_width)).float().to(rpn_cls_scores.device)

        # Reshape scores and predictions
        rpn_cls_scores = rpn_cls_scores.permute(0, 2, 3, 1).contiguous().view(1, -1, 2)
        rpn_bbox_preds = rpn_bbox_preds.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)

        # Get foreground scores
        scores = nn.functional.softmax(rpn_cls_scores, dim=2)[:, :, 1]

        # Apply bounding box deltas to anchors
        proposals = self._bbox_transform_inv(anchors, rpn_bbox_preds.squeeze(0))

        # Clip proposals to image boundaries
        proposals[:, 0::2] = torch.clamp(proposals[:, 0::2], 0, img_size[1] - 1)
        proposals[:, 1::2] = torch.clamp(proposals[:, 1::2], 0, img_size[0] - 1)

        # Remove proposals that are too small
        min_size = 16
        keep = self._filter_proposals(proposals, min_size)
        proposals = proposals[keep, :]
        scores = scores.squeeze(0)[keep]

        # Sort proposals by score and apply NMS
        order = torch.argsort(scores, descending=True)
        pre_nms_top_n = self.n_train_pre_nms if self.training else self.n_test_pre_nms
        order = order[:pre_nms_top_n]
        proposals = proposals[order, :]
        scores = scores[order]

        keep = nms(proposals, scores, self.nms_thresh)
        post_nms_top_n = self.n_train_post_nms if self.training else self.n_test_post_nms
        keep = keep[:post_nms_top_n]
        proposals = proposals[keep, :]

        # Add batch index to proposals
        batch_inds = torch.zeros((proposals.shape[0], 1), dtype=torch.float32).to(proposals.device)
        proposals = torch.cat([batch_inds, proposals], 1)

        return proposals

    def _bbox_transform_inv(self, boxes, deltas):
        """Applies bounding box regression deltas to the anchors."""
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        dx = deltas[:, 0::4]
        dy = deltas[:, 1::4]
        dw = deltas[:, 2::4]
        dh = deltas[:, 3::4]

        pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
        pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
        pred_w = torch.exp(dw) * widths.unsqueeze(1)
        pred_h = torch.exp(dh) * heights.unsqueeze(1)

        pred_boxes = deltas.clone()
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes

    def _filter_proposals(self, proposals, min_size):
        """Removes proposals that are smaller than a minimum size."""
        ws = proposals[:, 2] - proposals[:, 0] + 1
        hs = proposals[:, 3] - proposals[:, 1] + 1
        keep = torch.where((ws >= min_size) & (hs >= min_size))[0]
        return keep
