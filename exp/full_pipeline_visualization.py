import torch
import torchvision
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch.nn.functional as F

# Add root directory to path to import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.rpn import RegionProposalNetwork
from model.roi_head import ROIHead
from model.utils import apply_regression_pred_to_anchors_or_proposals, boxes_to_transformation_targets, get_iou, clamp_boxes_to_image_boundary
from config.voc_config import FasterRCNNConfig
from dataset.voc_loader import VOCDataset

def visualize_faster_rcnn_pipeline():
    """
    Visualizes the entire Faster R-CNN pipeline in a single plot with multiple subplots.
    """
    print("Visualizing Faster R-CNN Pipeline")
    print("---")

    os.makedirs('exp', exist_ok=True)

    print("Loading configuration, models, and dataset...")
    config = FasterRCNNConfig()
    config.scales = [256] # Reduce the number of scales to reduce anchors
    # RPN and ROIHead in eval mode for inference-like flow, but RPN in train mode for target assignment
    rpn_eval = RegionProposalNetwork(config).eval()
    rpn_train = RegionProposalNetwork(config).train() # For assign_targets_to_anchors
    roi_head_eval = ROIHead(config).eval()
    roi_head_eval.low_score_threshold = 0.9 # Set threshold to 0.8
    roi_head_train = ROIHead(config).train() # For assign_target_to_proposals

    # Load the first image and its ground truth targets from the test dataset
    test_dataset = VOCDataset(split='test', im_dir=config.im_test_path, ann_dir=config.ann_test_path)
    image_pil, targets, image_path = test_dataset.get_raw(0)
    gt_boxes = targets['bboxes']
    gt_labels = targets['labels']

    # Resize image and GT boxes to match model input
    w, h = image_pil.size
    scale = min(config.max_im_size / max(h, w), config.min_im_size / min(h, w))
    new_w, new_h = int(w * scale), int(h * scale)
    gt_boxes_resized = gt_boxes.float() * scale
    image_resized = image_pil.resize((new_w, new_h))

    image_tensor = torch.tensor(np.array(image_resized)).permute(2, 0, 1).float().unsqueeze(0)
    image_h, image_w = image_tensor.shape[-2:]

    # Create dummy feature map (mimic backbone output)
    stride = 16
    feat_h, feat_w = image_h // stride, image_w // stride
    feat_tensor = torch.randn(1, config.backbone_out_channels, feat_h, feat_w)

    print(f"Using first image from test set: {os.path.basename(image_path)}")
    print(f"Resized image dimensions: ({image_h}, {image_w})")
    print(f"Ground Truth Boxes: {gt_boxes_resized.shape[0]}")

    # --- Setup for subplots ---
    fig, axes = plt.subplots(3, 3, figsize=(27, 18)) # Changed to 3x3 grid
    axes = axes.flatten()
    plot_idx = 0

    # --- 1. Original Image & GT Boxes ---
    ax = axes[plot_idx]
    ax.imshow(image_resized)
    for i in range(gt_boxes_resized.shape[0]):
        x1, y1, x2, y2 = gt_boxes_resized[i].numpy()
        w_gt, h_gt = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w_gt, h_gt, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
    ax.set_title("1. Original Image & GT Boxes")
    ax.set_xlim(0, image_w)
    ax.set_ylim(image_h, 0)
    plot_idx += 1

    # --- 2. RPN: Generated Anchors (Sampled) ---
    anchors = rpn_eval.generate_anchors(image_tensor, feat_tensor)
    num_anchors_to_vis = 300
    random_indices = np.random.choice(anchors.shape[0], size=num_anchors_to_vis, replace=False)
    sampled_anchors = anchors[random_indices]

    ax = axes[plot_idx]
    ax.imshow(image_resized)
    for i in range(sampled_anchors.shape[0]):
        x1, y1, x2, y2 = sampled_anchors[i].numpy()
        w_a, h_a = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w_a, h_a, linewidth=0.5, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
    ax.set_title(f"2. RPN: Generated Anchors ({num_anchors_to_vis} Sampled)")
    # Set plot limits to see anchors outside the image
    min_x = anchors[:, 0].min().item()
    min_y = anchors[:, 1].min().item()
    max_x = anchors[:, 2].max().item()
    max_y = anchors[:, 3].max().item()
    padding = 100
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(max_y + padding, min_y - padding)
    plot_idx += 1

    # --- 3. RPN: Proposals (after Regression) ---
    num_anchors = anchors.shape[0]
    box_transform_pred_rpn_dummy = torch.zeros(num_anchors, 1, 4)
    dummy_transformation = torch.tensor([0.2, 0.1, 0.3, 0.4]) # Example transformation
    box_transform_pred_rpn_dummy[:, 0, :] = dummy_transformation
    proposals_after_regression = apply_regression_pred_to_anchors_or_proposals(box_transform_pred_rpn_dummy, anchors).squeeze(1)

    num_to_vis_reg = 300
    random_indices_reg = np.random.choice(anchors.shape[0], size=num_to_vis_reg, replace=False)
    sampled_anchors_reg = anchors[random_indices_reg]
    sampled_proposals_reg = proposals_after_regression[random_indices_reg]

    ax = axes[plot_idx]
    ax.imshow(image_resized)
    for i in range(sampled_anchors_reg.shape[0]):
        x1, y1, x2, y2 = sampled_anchors_reg[i].numpy()
        w_a, h_a = x2 - x1, y2 - y1
        rect_anchor = patches.Rectangle((x1, y1), w_a, h_a, linewidth=0.5, edgecolor='g', facecolor='none')
        ax.add_patch(rect_anchor)

        x1_p, y1_p, x2_p, y2_p = sampled_proposals_reg[i].numpy()
        w_p, h_p = x2_p - x1_p, y2_p - y1_p
        rect_proposal = patches.Rectangle((x1_p, y1_p), w_p, h_p, linewidth=0.5, edgecolor='b', facecolor='none')
        ax.add_patch(rect_proposal)
    ax.set_title(f"3. RPN: Proposals (after Regression) ({num_to_vis_reg} Sampled)")
    all_boxes_reg = torch.cat([anchors, proposals_after_regression], dim=0)
    min_x_reg, min_y_reg = all_boxes_reg[:, 0].min().item(), all_boxes_reg[:, 1].min().item()
    max_x_reg, max_y_reg = all_boxes_reg[:, 2].max().item(), all_boxes_reg[:, 3].max().item()
    ax.set_xlim(min_x_reg - padding, max_x_reg + padding)
    ax.set_ylim(max_y_reg + padding, min_y_reg - padding)
    plot_idx += 1

    # --- RPN Forward Pass ---
    # Call the RPN forward pass to get actual rpn_output
    rpn_output = rpn_eval(image_tensor, feat_tensor)
    # Use the proposals from the actual RPN output
    filtered_proposals_rpn = rpn_output['proposals']

    # --- 4. RPN Output (Filtered Proposals) --- (Previously "RPN: Filtered Proposals (after NMS)")
    ax = axes[plot_idx]
    ax.imshow(image_resized)
    # Draw final proposals *after* NMS (blue)
    for i in range(filtered_proposals_rpn.shape[0]):
        x1, y1, x2, y2 = filtered_proposals_rpn[i].numpy()
        w_f, h_f = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w_f, h_f, linewidth=1.5, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
    ax.set_title(f"4. RPN Output (Filtered Proposals: {filtered_proposals_rpn.shape[0]} Final)")
    # Adjust limits based on filtered proposals
    if filtered_proposals_rpn.shape[0] > 0:
        min_x_filter_rpn, min_y_filter_rpn = filtered_proposals_rpn[:, 0].min().item(), filtered_proposals_rpn[:, 1].min().item()
        max_x_filter_rpn, max_y_filter_rpn = filtered_proposals_rpn[:, 2].max().item(), filtered_proposals_rpn[:, 3].max().item()
        ax.set_xlim(min_x_filter_rpn - padding, max_x_filter_rpn + padding)
        ax.set_ylim(max_y_filter_rpn + padding, min_y_filter_rpn - padding)
    else:
        ax.set_xlim(0, image_w)
        ax.set_ylim(image_h, 0)
    plot_idx += 1

    # --- 5. ROIHead: Assigned Proposals (for training) ---
    # Use filtered_proposals_rpn as input to ROIHead
    proposals_for_roi_assign = filtered_proposals_rpn
    proposals_with_gt_roi = torch.cat([proposals_for_roi_assign, gt_boxes_resized], dim=0)
    labels_roi, _ = roi_head_train.assign_target_to_proposals(proposals_with_gt_roi, gt_boxes_resized, gt_labels)

    positive_proposals_roi = proposals_with_gt_roi[labels_roi > 0]
    negative_proposals_roi = proposals_with_gt_roi[labels_roi == 0]
    ignore_proposals_roi = proposals_with_gt_roi[labels_roi == -1]

    ax = axes[plot_idx]
    ax.imshow(image_resized)
    # Draw Ground Truth boxes (White)
    for i in range(gt_boxes_resized.shape[0]):
        x1, y1, x2, y2 = gt_boxes_resized[i].numpy()
        w_gt, h_gt = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w_gt, h_gt, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
    # Draw a sample of Negative proposals (Red)
    num_to_sample_roi = 200
    if negative_proposals_roi.shape[0] > 0:
        indices = np.random.choice(negative_proposals_roi.shape[0], size=min(num_to_sample_roi, negative_proposals_roi.shape[0]), replace=False)
        for i in indices:
            x1, y1, x2, y2 = negative_proposals_roi[i].numpy()
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=0.5, edgecolor='r', facecolor='none', alpha=0.7)
            ax.add_patch(rect)
    # Draw a sample of Ignore proposals (Yellow)
    if ignore_proposals_roi.shape[0] > 0:
        indices = np.random.choice(ignore_proposals_roi.shape[0], size=min(num_to_sample_roi, ignore_proposals_roi.shape[0]), replace=False)
        for i in indices:
            x1, y1, x2, y2 = ignore_proposals_roi[i].numpy()
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=0.5, edgecolor='y', facecolor='none', alpha=0.7)
            ax.add_patch(rect)
    # Draw ALL Positive proposals (Green)
    for i in range(positive_proposals_roi.shape[0]):
        x1, y1, x2, y2 = positive_proposals_roi[i].numpy()
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=1.5, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
    ax.set_title(f"5. ROIHead: Assigned Proposals (Pos: {positive_proposals_roi.shape[0]}) ")
    ax.set_xlim(0, image_w)
    ax.set_ylim(image_h, 0)
    plot_idx += 1

    # --- ROIHead Forward Pass ---
    # Call the ROIHead forward pass to get actual frcnn_output
    frcnn_output = roi_head_eval(feat_tensor, rpn_output['proposals'], image_tensor.shape[-2:], target=None)
    # Use the detections from the actual FRCNN output
    filtered_boxes_roi = frcnn_output['boxes']
    filtered_labels_roi = frcnn_output['labels']
    filtered_scores_roi = frcnn_output['scores']

    # --- 6. Faster R-CNN Final Detections --- (Previously "ROIHead: Final Detections")
    ax = axes[plot_idx]
    ax.imshow(image_resized)
    # Draw Ground Truth boxes (White)
    for i in range(gt_boxes_resized.shape[0]):
        x1, y1, x2, y2 = gt_boxes_resized[i].numpy()
        w_gt, h_gt = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w_gt, h_gt, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
    # Draw final filtered detections (Red, opaque)
    for i in range(filtered_boxes_roi.shape[0]):
        x1, y1, x2, y2 = filtered_boxes_roi[i].numpy()
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    ax.set_title(f"6. Faster R-CNN Final Detections ({filtered_boxes_roi.shape[0]} Final)")
    all_boxes_filter_roi = torch.cat([gt_boxes_resized, filtered_boxes_roi], dim=0) # Only GT and final detections
    if all_boxes_filter_roi.shape[0] > 0:
        min_x_filter_roi, min_y_filter_roi = all_boxes_filter_roi[:, 0].min().item(), all_boxes_filter_roi[:, 1].min().item()
        max_x_filter_roi, max_y_filter_roi = all_boxes_filter_roi[:, 2].max().item(), all_boxes_filter_roi[:, 3].max().item()
        ax.set_xlim(min_x_filter_roi - padding, max_x_filter_roi + padding)
        ax.set_ylim(max_y_filter_roi + padding, min_y_filter_roi - padding)
    else:
        ax.set_xlim(0, image_w)
        ax.set_ylim(image_h, 0)
    plot_idx += 1

    # --- 7. ROI Pooled Features (Sampled) ---
    ax = axes[plot_idx]
    ax.set_title("7. ROI Pooled Features (Sampled)")
    ax.axis('off') # Turn off axis for feature map visualization

    if rpn_output['proposals'].shape[0] > 0:
        # Simulate ROI pooling
        # spatial_scale = 1/stride is correct for VGG-like backbones where stride is 16
        proposal_roi_pool_feats = torchvision.ops.roi_pool(feat_tensor, [rpn_output['proposals']],
                                                           output_size=config.roi_pool_size,
                                                           spatial_scale=1/stride)

        num_pooled_to_vis = min(9, proposal_roi_pool_feats.shape[0]) # Visualize up to 9 pooled features
        if num_pooled_to_vis > 0:
            # Create a sub-grid within this subplot for the pooled features
            gs = ax.get_subplotspec().subgridspec(int(np.ceil(np.sqrt(num_pooled_to_vis))), int(np.ceil(np.sqrt(num_pooled_to_vis))))
            gs_axes = [fig.add_subplot(gs[i]) for i in range(num_pooled_to_vis)]

            for i in range(num_pooled_to_vis):
                pooled_feat = proposal_roi_pool_feats[i] # (C, H_pool, W_pool)
                # Average across channels for a single grayscale image
                pooled_feat_avg = pooled_feat.mean(dim=0).cpu().numpy()
                gs_axes[i].imshow(pooled_feat_avg, cmap='viridis')
                gs_axes[i].axis('off')
                gs_axes[i].set_title(f"Prop {i+1}", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No proposals to pool", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    plot_idx += 1

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    visualize_faster_rcnn_pipeline()
