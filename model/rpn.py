import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import get_iou, \
                        apply_regression_pred_to_anchors_or_proposals, \
                        clamp_boxes_to_image_boundary, \
                        boxes_to_transformation_targets, \
                        sample_positive_negative


class RegionProposalNetwork(nn.Module):
    """
    RPN with following layers on the feature map
        1. 3x3 conv layer followed by Relu
        2. 1x1 classification conv with num_anchors(num_scales x num_aspect_ratios) output channels
        3. 1x1 classification conv with 4 x num_anchors output channels

    Classification is done via one value indicating probability of foreground
    with sigmoid applied during inference
    """
    
    def __init__(self, config):
        super(RegionProposalNetwork, self).__init__()
        self.in_channels = config.backbone_out_channels
        self.scales = config.scales
        self.aspect_ratios = config.aspect_ratios

        self.low_iou_threshold = config.rpn_bg_threshold
        self.high_iou_threshold = config.rpn_fg_threshold
        self.rpn_nms_threshold = config.rpn_nms_threshold

        self.rpn_batch_size = config.rpn_batch_size
        self.rpn_pos_count = int(config.rpn_pos_fraction * self.rpn_batch_size)
        self.rpn_topk = config.rpn_train_topk if self.training else config.rpn_test_topk
        self.rpn_prenms_topk = config.rpn_train_prenms_topk if self.training else config.rpn_test_prenms_topk

        self.num_anchors = len(self.scales) * len(self.aspect_ratios)
        
        # 3x3 conv layer
        self.rpn_conv = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1)
        
        # 1x1 classification conv layer
        self.cls_layer = nn.Conv2d(self.in_channels, self.num_anchors, kernel_size=1, stride=1)
        
        # 1x1 regression
        self.bbox_reg_layer = nn.Conv2d(self.in_channels, self.num_anchors * 4, kernel_size=1, stride=1)
        
        for layer in [self.rpn_conv, self.cls_layer, self.bbox_reg_layer]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)
    
    def generate_anchors(self, image, feat):
        """
        Method to generate anchors. 
        1. generate one set of zero-centred anchors (using the scales and aspect ratios provided)
        2. then generate shift values in x,y axis for all featuremap locations
        3. The single zero centred anchors generated are replicated and shifted accordingly
            to generate anchors for all feature map locations.
        
        Note that these anchors are generated such that their centre is top left corner of the
        feature map cell rather than the centre of the feature map cell.

        :param image: (N, C, H, W) tensor
        :param feat: (N, C_feat, H_feat, W_feat) tensor
        :return: anchor boxes of shape (H_feat * W_feat * num_anchors_per_location, 4)
        """
        # 1. generate zero-cented base anchors
        grid_h, grid_w = feat.shape[-2:]            # feature map size
        image_h, image_w = image.shape[-2:]         # original image size
        
        # For the vgg16 case stride would be 16 for both h and w
        stride_h = torch.tensor(image_h // grid_h, dtype=torch.int64, device=feat.device)
        stride_w = torch.tensor(image_w // grid_w, dtype=torch.int64, device=feat.device)
        
        scales = torch.as_tensor(self.scales, dtype=feat.dtype, device=feat.device)
        aspect_ratios = torch.as_tensor(self.aspect_ratios, dtype=feat.dtype, device=feat.device)

        # The below code ensures h/w = aspect_ratios and h*w=1
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios
        
        # Actual width and height for all combinations of scales and latios
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)
        
        # Now we make all 9 anchors zero centred -> x1, y1, x2, y2 = -w/2, -h/2, w/2, h/2
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        base_anchors = base_anchors.round()
        
        # 2. generate shift values
        # Get the shifts in x axis (0, 1,..., W_feat-1) * stride_w
        shifts_x = torch.arange(0, grid_w, dtype=torch.int32, device=feat.device) * stride_w

        # Get the shifts in y axis (0, 1,..., H_feat-1) * stride_h
        shifts_y = torch.arange(0, grid_h, dtype=torch.int32, device=feat.device) * stride_h
        
        # Create a grid using these shifts
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        
        shifts_x = shifts_x.reshape(-1)
        shifts_y = shifts_y.reshape(-1)
        # Convert to a tensor in the form of [shift_x, shift_y, ...] 
        # so that it can be applied equally to all (x1, y1, x2, y2)
        shifts = torch.stack((shifts_x, shifts_y, shifts_x, shifts_y), dim=1)
        # shifts -> (H_feat * W_feat, 4)
        
        # 3. base anchors generated are replicated and shifted for all feature map locations.
        # Add shifts to each of the base anchors
        # anchors -> (H_feat * W_feat, num_anchors_per_location, 4) -> (H_feat * W_feat * num_anchors_per_location, 4)
        anchors = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4))
        anchors = anchors.reshape(-1, 4)
        
        return anchors
    
    def assign_targets_to_anchors(self, anchors, gt_boxes):
        """
        1. For each anchor assign a ground truth box based on the IOU.
        2. Also creates classification labels to be used for training
            a. foreground: label=1 for anchors where maximum IOU with a gtbox > high_iou_threshold
            b. background: label=0 for anchors where maximum IOU with a gtbox < low_iou_threshold
            c. ignore: label=-1 for anchors where maximum IOU with a gtbox between (low_iou_threshold, high_iou_threshold)

        input
            anchors: (num_anchors_in_image, 4) all anchor boxes
            gt_boxes: (num_gt_boxes_in_image, 4) all ground truth boxes
        
        return
            label: (num_anchors_in_image) {-1/0/1}
            matched_gt_boxes: (num_anchors_in_image, 4) coordinates of assigned gt_box to each anchor
                Even (background/to_be_ignored anchors) will be assigned some ground truth box.
                It's fine, we will use label to differentiate those instances later
        """
        
        # Calculate IoU between all gt_boxes and all anchor_boxes
        iou_matrix = get_iou(gt_boxes, anchors)
        
        # Find the "gt_box" with the highest IoU value for "each anchor".
        best_match_iou, best_match_gt_idx = iou_matrix.max(dim=0)
        best_match_gt_idx_pre_thresholding = best_match_gt_idx.clone()
        
        # Based on threshold, update the values of best_match_gt_idx
        below_low_threshold = best_match_iou < self.low_iou_threshold      # background
        between_thresholds = (best_match_iou >= self.low_iou_threshold) & (best_match_iou < self.high_iou_threshold)    # ignore
        best_match_gt_idx[below_low_threshold] = -1
        best_match_gt_idx[between_thresholds] = -2
        
        # Find the "anchor box" with the highest IoU value for "gt_box".
        # Forcing these "most relevant" anchors to be a positive sample
        best_anchor_iou_for_gt, _ = iou_matrix.max(dim=1)
        
        # Even if the IoU value is lower than 0.7, this anchor is gt_box.
        # ex) gt_pred_pair_with_highest_iou -> [0, 0, 0, 1, 1, 1], [8896, 8905, 8914, 10472, 10805, 11138]
        # label 0: [8896, 8905, 8914], label 1: [10472, 10805, 11138] (same iou)
        gt_pred_pair_with_highest_iou = torch.where(iou_matrix == best_anchor_iou_for_gt[:, None])
        
        # Get all the anchors indexes to update
        pred_inds_to_update = gt_pred_pair_with_highest_iou[1]
        
        # Reassign the gt_box index that was originally paired to these anchors.
        best_match_gt_idx[pred_inds_to_update] = best_match_gt_idx_pre_thresholding[pred_inds_to_update]
        
        # making the negative index (-1, -2) in best_match_gt_idx to zero.
        matched_gt_boxes = gt_boxes[best_match_gt_idx.clamp(min=0)]
        
        # a. foreground anchor labels as 1
        labels = best_match_gt_idx >= 0
        labels = labels.to(dtype=torch.float32)
        
        # b. background anchor labels as 0
        background_anchors = best_match_gt_idx == -1
        labels[background_anchors] = 0.0
        
        # c. ignore anchor labels as -1
        ignored_anchors = best_match_gt_idx == -2
        labels[ignored_anchors] = -1.0
        
        return labels, matched_gt_boxes

    def filter_proposals(self, proposals, cls_scores, image_shape):
        """
        1. Pre NMS topK filtering
        2. Make proposals valid by clamping coordinates(0, width/height)
        2. Small Boxes filtering based on width and height
        3. NMS
        4. Post NMS topK filtering

        input
            proposals: (num_anchors_in_image, 4)
            cls_scores: (num_anchors_in_image, 4) these are cls logits
            image_shape: resized image shape needed to clip proposals to image boundary
        return
            proposals: (num_filtered_proposals, 4)
            cls_scores: num_filtered_proposals
        """
        # Pre NMS Filtering
        cls_scores = cls_scores.reshape(-1)
        cls_scores = torch.sigmoid(cls_scores)                                      # 0-1 mapping
        _, top_n_idx = cls_scores.topk(min(self.rpn_prenms_topk, len(cls_scores)))  # filter by number of "rpn_pre_nms_topk"
        
        cls_scores = cls_scores[top_n_idx]
        proposals = proposals[top_n_idx]
        
        # Clamp boxes to image boundary
        proposals = clamp_boxes_to_image_boundary(proposals, image_shape)           # bring the bounding box into the image
        
        # Small boxes based on width and height filtering
        min_size = 16
        ws, hs = proposals[:, 2] - proposals[:, 0], proposals[:, 3] - proposals[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
        keep = torch.where(keep)[0]
        proposals = proposals[keep]
        cls_scores = cls_scores[keep]
        
        # NMS based on objectness scores
        keep_mask = torch.zeros_like(cls_scores, dtype=torch.bool)
        keep_indices = torch.ops.torchvision.nms(proposals, cls_scores, self.rpn_nms_threshold) # filter values below rpn_nms_threshold
        keep_mask[keep_indices] = True
        keep_indices = torch.where(keep_mask)[0]

        # Sort by objectness (descending)
        post_nms_keep_indices = keep_indices[cls_scores[keep_indices].sort(descending=True)[1]]
        
        # Post NMS topk filtering
        proposals, cls_scores = (proposals[post_nms_keep_indices[:self.rpn_topk]],      # filter by number of "rpn_topk"
                                 cls_scores[post_nms_keep_indices[:self.rpn_topk]])
        
        return proposals, cls_scores
    
    def forward(self, image, feat, target=None):
        """
        1. Call RPN specific conv layers to generate classification and bbox transformation predictions for anchors
        2. Generate anchors for entire image
        3. Transform generated anchors based on predicted bbox transformation to generate proposals
        4. Filter proposals
        5. For training additionally we do the following:
            a. Assign target ground truth labels and boxes to each anchors
            b. Sample positive and negative anchors
            c. Compute classification loss using sampled pos/neg anchors
            d. Compute Localization loss using sampled pos anchors
        """
        # Call RPN layers
        rpn_feat = nn.ReLU()(self.rpn_conv(feat))           # feature map
        cls_scores = self.cls_layer(rpn_feat)               # objecness score
        box_transform_pred = self.bbox_reg_layer(rpn_feat)  # bbox transformation prediction

        # Generate anchors
        anchors = self.generate_anchors(image, feat)
        
        # cls_score -> (Batch_Size, Number of Anchors per location, H_feat, W_feat) 
        # -> (Batch_Size * H_feat * W_feat * Number of Anchors per location, 1)
        number_of_anchors_per_location = cls_scores.size(1)
        cls_scores = cls_scores.permute(0, 2, 3, 1)
        cls_scores = cls_scores.reshape(-1, 1)
        
        # box_transform_pred -> (Batch_Size, Number of Anchors per location * 4, H_feat, W_feat)
        # -> (Batch_Size * H_feat * W_feat * Number of Anchors per location, 4)
        box_transform_pred = box_transform_pred.view(
            box_transform_pred.size(0),
            number_of_anchors_per_location,
            4,
            rpn_feat.shape[-2],
            rpn_feat.shape[-1])
        box_transform_pred = box_transform_pred.permute(0, 3, 4, 1, 2)
        box_transform_pred = box_transform_pred.reshape(-1, 4)
        
        # apply the "box_transform_pred" predicted by the RPN to the "anchors" in the fixed position
        proposals = apply_regression_pred_to_anchors_or_proposals(
            box_transform_pred.detach().reshape(-1, 1, 4),
            anchors)
        proposals = proposals.reshape(proposals.size(0), 4)

        # filter_proposals via NMS (non-maximum suppression) etc
        proposals, scores = self.filter_proposals(proposals, cls_scores.detach(), image.shape)
        rpn_output = {
            'proposals': proposals,
            'scores': scores
        }

        # If not training, no need to do anything
        # If training, assign gt box and label for each anchor
        if not self.training or target is None:
            return rpn_output
        else:
            labels_for_anchors, matched_gt_boxes_for_anchors = self.assign_targets_to_anchors(
                anchors,
                target['bboxes'][0])
            
            # The model predicts "(tx, ty, tw, th)" rather than the coordinates themselves (x, y, w, h)
            # so gt is regression_targets
            regression_targets = boxes_to_transformation_targets(matched_gt_boxes_for_anchors, anchors)
            
            # Sampling positive and negative anchors {fg:1, bg:0, to_be_ignored:-1}
            # Most of the anchors are background, negative.
            # Using all of them for training can bias the model to predict only the background.
            sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_negative(
                labels_for_anchors,
                positive_count=self.rpn_pos_count,
                total_count=self.rpn_batch_size)
            
            sampled_idxs = torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)[0]
            

            # Calculate only for the positive object anchors (because the position of the background does not matter)
            localization_loss = (
                    F.smooth_l1_loss(
                        box_transform_pred[sampled_pos_idx_mask],
                        regression_targets[sampled_pos_idx_mask],
                        beta=1 / 9,
                        reduction="sum",
                    )
                    / (sampled_idxs.numel())
            ) 

            # Calculate for both positive and negative anchors sampled
            cls_loss = F.binary_cross_entropy_with_logits(cls_scores[sampled_idxs].flatten(),
                                                                            labels_for_anchors[sampled_idxs].flatten())

            rpn_output['rpn_classification_loss'] = cls_loss
            rpn_output['rpn_localization_loss'] = localization_loss
            return rpn_output
