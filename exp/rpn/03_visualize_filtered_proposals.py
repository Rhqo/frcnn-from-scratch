import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Add root directory to path to import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from model.rpn import RegionProposalNetwork
from model.utils import apply_regression_pred_to_anchors_or_proposals
from config.voc_config import FasterRCNNConfig
from dataset.voc_loader import VOCDataset

def visualize_filtered_proposals():
    """
    Visualizes proposal filtering using Matplotlib to show proposals outside image boundaries.
    """
    print("Step 3: Visualizing filter_proposals() with Matplotlib")
    print("---")

    os.makedirs('exp/rpn', exist_ok=True)

    print("Loading configuration, model, and dataset...")
    config = FasterRCNNConfig()
    rpn = RegionProposalNetwork(config).eval()

    # Load the first image from the test dataset
    test_dataset = VOCDataset(split='test', im_dir=config.im_test_path, ann_dir=config.ann_test_path)
    image_tensor, _, image_path = test_dataset[0]
    image_tensor = image_tensor.unsqueeze(0)

    image = Image.open(image_path).convert("RGB")
    image_h, image_w = image_tensor.shape[-2:]
    image_resized = image.resize((image_w, image_h))

    # Create dummy feature map
    stride = 16
    feat_h, feat_w = image_h // stride, image_w // stride
    feat_tensor = torch.randn(1, config.backbone_out_channels, feat_h, feat_w)

    # 1. Generate anchors and initial proposals
    anchors = rpn.generate_anchors(image_tensor, feat_tensor)
    num_anchors = anchors.shape[0]
    box_transform_pred = torch.randn(num_anchors, 1, 4) * 0.1
    proposals = apply_regression_pred_to_anchors_or_proposals(box_transform_pred, anchors).squeeze(1)
    cls_scores = torch.randn(num_anchors, 1) * 2 - 1
    
    proposal_centers_x = (proposals[:, 0] + proposals[:, 2]) / 2
    proposal_centers_y = (proposals[:, 1] + proposals[:, 3]) / 2
    center_x, center_y = image_w / 2, image_h / 2
    distances = torch.sqrt((proposal_centers_x - center_x)**2 + (proposal_centers_y - center_y)**2)
    cls_scores[distances < 150] += 2.5

    # 2. Get top proposals before NMS for visualization
    pre_nms_topk = config.rpn_test_prenms_topk
    scores_for_vis = torch.sigmoid(cls_scores.reshape(-1))
    _, top_indices_before = scores_for_vis.topk(min(pre_nms_topk, len(scores_for_vis)))
    proposals_before_filter = proposals[top_indices_before]

    # 3. Filter proposals
    filtered_proposals, _ = rpn.filter_proposals(proposals.detach(), cls_scores.detach(), image_tensor.shape)
    print(f"After filtering, {filtered_proposals.shape[0]} proposals remain.")

    # 4. Visualize with Matplotlib
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image_resized)

    # Draw final proposals *after* NMS (blue)
    for i in range(filtered_proposals.shape[0]):
        x1, y1, x2, y2 = filtered_proposals[i].numpy()
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=0.4, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    # Set plot limits to image boundaries as proposals are clipped
    ax.set_xlim(0, image_w)
    ax.set_ylim(image_h, 0)

    handles = [
        patches.Patch(color='blue', label=f'Filtered Proposals: {filtered_proposals.shape[0]}')
    ]
    plt.legend(handles=handles)
    plt.title(f"Proposal Filtering Visualization (Before NMS: {proposals_before_filter.shape[0]}, After NMS: {filtered_proposals.shape[0]})")
    plt.show()

if __name__ == '__main__':
    visualize_filtered_proposals()
