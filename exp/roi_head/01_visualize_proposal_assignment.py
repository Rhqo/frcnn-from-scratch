import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Add root directory to path to import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from model.utils import apply_regression_pred_to_anchors_or_proposals

from model.rpn import RegionProposalNetwork
from model.roi_head import ROIHead
from config.voc_config import FasterRCNNConfig
from dataset.voc_loader import VOCDataset

def visualize_proposal_assignment():
    """
    Visualizes the assignment of targets to proposals within the ROIHead.
    Shows positive, negative, and ignored proposals based on ground truth.
    """
    print("Step 1: Visualizing assign_targets_to_proposals() for ROIHead")
    print("---")

    os.makedirs('exp/roi_head', exist_ok=True)

    print("Loading configuration, models, and dataset...")
    config = FasterRCNNConfig()
    # RPN in eval mode to get proposals, ROIHead in train mode for target assignment
    rpn = RegionProposalNetwork(config).eval()
    roi_head = ROIHead(config).train()

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

    # 1. Generate proposals from RPN (using dummy feature map)
    # We need to simulate the RPN output for ROIHead input
    # For simplicity, let's generate some random proposals around GT boxes
    # In a real scenario, these would come from rpn.forward()
    
    # Let's use the RPN's generate_anchors and apply_regression_pred_to_anchors_or_proposals
    # to get a more realistic set of proposals, then filter them.
    anchors = rpn.generate_anchors(image_tensor, feat_tensor)
    num_anchors = anchors.shape[0]
    box_transform_pred_rpn = torch.randn(num_anchors, 1, 4) * 0.1 # Small random transformations
    initial_proposals = apply_regression_pred_to_anchors_or_proposals(box_transform_pred_rpn, anchors).squeeze(1)
    
    # Simulate objectness scores for filtering
    cls_scores_rpn = torch.randn(num_anchors, 1) * 2 - 1
    proposal_centers_x = (initial_proposals[:, 0] + initial_proposals[:, 2]) / 2
    proposal_centers_y = (initial_proposals[:, 1] + initial_proposals[:, 3]) / 2
    center_x, center_y = image_w / 2, image_h / 2
    distances = torch.sqrt((proposal_centers_x - center_x)**2 + (proposal_centers_y - center_y)**2)
    cls_scores_rpn[distances < 150] += 2.5

    # Filter proposals using RPN's filter_proposals logic
    # Detach to simulate fixed proposals from RPN for ROIHead
    proposals, _ = rpn.filter_proposals(initial_proposals.detach(), cls_scores_rpn.detach(), image_tensor.shape)
    print(f"Generated {proposals.shape[0]} proposals from RPN simulation.")

    # 2. Assign targets to proposals
    # In training, ROIHead also concatenates GT boxes to proposals for assignment
    proposals_with_gt = torch.cat([proposals, gt_boxes_resized], dim=0)
    print(f"Proposals for assignment (including GT): {proposals_with_gt.shape[0]}")

    labels, matched_gt_boxes = roi_head.assign_target_to_proposals(proposals_with_gt, gt_boxes_resized, gt_labels)

    # 3. Identify the different types of proposals
    positive_proposals = proposals_with_gt[labels > 0] # Labels > 0 are class IDs
    negative_proposals = proposals_with_gt[labels == 0]
    ignore_proposals = proposals_with_gt[labels == -1]

    print(f"Proposal stats: {positive_proposals.shape[0]} positive, {negative_proposals.shape[0]} negative, {ignore_proposals.shape[0]} ignore")

    # 4. Visualize with Matplotlib
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image_resized)

    num_to_sample = 200 # Sample size for negative/ignore proposals

    # Draw Ground Truth boxes (White)
    for i in range(gt_boxes_resized.shape[0]):
        x1, y1, x2, y2 = gt_boxes_resized[i].numpy()
        w_gt, h_gt = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w_gt, h_gt, linewidth=4, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    # Draw a sample of Negative proposals (Red)
    if negative_proposals.shape[0] > 0:
        indices = np.random.choice(negative_proposals.shape[0], size=min(num_to_sample, negative_proposals.shape[0]), replace=False)
        for i in indices:
            x1, y1, x2, y2 = negative_proposals[i].numpy()
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=0.5, edgecolor='r', facecolor='none', alpha=0.7)
            ax.add_patch(rect)

    # Draw a sample of Ignore proposals (Yellow)
    if ignore_proposals.shape[0] > 0:
        indices = np.random.choice(ignore_proposals.shape[0], size=min(num_to_sample, ignore_proposals.shape[0]), replace=False)
        for i in indices:
            x1, y1, x2, y2 = ignore_proposals[i].numpy()
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='y', facecolor='none', alpha=0.7)
            ax.add_patch(rect)

    # Draw ALL Positive proposals (Green)
    for i in range(positive_proposals.shape[0]):
        x1, y1, x2, y2 = positive_proposals[i].numpy()
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    # Set plot limits to ensure all boxes are visible
    all_boxes = torch.cat([proposals_with_gt, gt_boxes_resized], dim=0)
    min_x, min_y = all_boxes[:, 0].min().item(), all_boxes[:, 1].min().item()
    max_x, max_y = all_boxes[:, 2].max().item(), all_boxes[:, 3].max().item()
    padding = 100
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(max_y + padding, min_y - padding)

    handles = [
        patches.Patch(color='blue', label='Ground Truth'),
        patches.Patch(color='green', label=f'Positive Proposals ({positive_proposals.shape[0]})'),
        patches.Patch(color='red', alpha=0.7, label=f'Negative Proposals ({min(num_to_sample, negative_proposals.shape[0])} sampled)'),
        patches.Patch(color='yellow', alpha=0.7, label=f'Ignore Proposals ({min(num_to_sample, ignore_proposals.shape[0])} sampled)')
    ]
    plt.legend(handles=handles)
    plt.title("ROIHead Proposal Assignment Visualization")
    plt.show()

if __name__ == '__main__':
    visualize_proposal_assignment()
