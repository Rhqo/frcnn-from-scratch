
import torch
import torch.nn as nn

from frcnn.models.vgg16 import get_vgg16_base_net
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
        self.extractor = get_vgg16_base_net()

        # 2. Region Proposal Network (RPN)
        self.rpn = RPN(in_channels=512, mid_channels=512, n_anchor=9)

        # 3. Proposal Layer (converts RPN outputs to proposals)
        self.proposal_layer = ProposalLayer(
            feat_stride=feat_stride, n_anchor=9, nms_thresh=rpn_nms_thresh,
            n_train_pre_nms=rpn_train_pre_nms, n_train_post_nms=rpn_train_post_nms,
            n_test_pre_nms=rpn_test_pre_nms, n_test_post_nms=rpn_test_post_nms
        )

        # 4. RoI Pooling Layer
        self.roi_pooling = RoIPooling(output_size=roi_output_size, spatial_scale=1.0 / feat_stride)

        # 5. Fast R-CNN Head
        self.head = FastRCNNHead(in_channels=512, num_classes=num_classes, roi_output_size=roi_output_size)

    def forward(self, x: torch.Tensor, img_size: tuple):
        """
        Forward pass of the Faster R-CNN model.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, H, W).
            img_size (tuple): Original image size (height, width).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - rpn_cls_scores (torch.Tensor): RPN classification scores.
                - rpn_bbox_preds (torch.Tensor): RPN bounding box predictions.
                - roi_cls_scores (torch.Tensor): RoI classification scores.
                - roi_bbox_preds (torch.Tensor): RoI bounding box predictions.
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

        return rpn_cls_scores, rpn_bbox_preds, roi_cls_scores, roi_bbox_preds
