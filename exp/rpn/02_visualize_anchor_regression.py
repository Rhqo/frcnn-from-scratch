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

def visualize_anchor_regression():
    """
    Visualizes anchor regression using Matplotlib to show anchors outside image boundaries.
    """
    print("Step 2: Visualizing apply_regression_pred_to_anchors_or_proposals() with Matplotlib")
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

    # 1. Generate anchors
    anchors = rpn.generate_anchors(image_tensor, feat_tensor)

    # 2. Create dummy regression predictions for the first 9 anchors
    num_anchors_to_regress = 9
    # Ensure we don't try to regress more anchors than available
    if anchors.shape[0] < num_anchors_to_regress:
        num_anchors_to_regress = anchors.shape[0]

    box_transform_pred = torch.zeros(num_anchors_to_regress, 1, 4)
    dummy_transformation = torch.tensor([0.2, 0.1, 0.3, 0.4])
    box_transform_pred[:, 0, :] = dummy_transformation

    print(f"Applying a dummy transformation {dummy_transformation.numpy()} to the first {num_anchors_to_regress} anchors.")

    # 3. Apply regression to the first 9 anchors to get proposals
    proposals = apply_regression_pred_to_anchors_or_proposals(box_transform_pred, anchors[:num_anchors_to_regress])
    proposals = proposals.squeeze(1)

    # 4. Visualize with Matplotlib
    # Use the first num_anchors_to_regress for visualization
    sampled_anchors = anchors[:num_anchors_to_regress]
    sampled_proposals = proposals

    print(f"Visualizing the first {num_anchors_to_regress} original anchors (green) and regressed proposals (blue).")

    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image_resized)

    for i in range(num_anchors_to_regress): # Iterate only over the regressed anchors
        # Original Anchor
        x1, y1, x2, y2 = sampled_anchors[i].numpy()
        w, h = x2 - x1, y2 - y1
        rect_anchor = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='g', facecolor='none', label='Original Anchors')
        ax.add_patch(rect_anchor)

        # Regressed Proposal
        x1_p, y1_p, x2_p, y2_p = sampled_proposals[i].numpy()
        w_p, h_p = x2_p - x1_p, y2_p - y1_p
        rect_proposal = patches.Rectangle((x1_p, y1_p), w_p, h_p, linewidth=1, edgecolor='b', facecolor='none', label='Regressed Proposals')
        ax.add_patch(rect_proposal)

    # Set plot limits based on the visualized anchors and proposals
    all_boxes = torch.cat([sampled_anchors, sampled_proposals], dim=0)
    min_x, min_y = all_boxes[:, 0].min().item(), all_boxes[:, 1].min().item()
    max_x, max_y = all_boxes[:, 2].max().item(), all_boxes[:, 3].max().item()
    padding = 200
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(max_y + padding, min_y - padding)

    # Create custom legend
    handles = [
        patches.Patch(color='green', label=f'Original Anchors ({num_anchors_to_regress})'),
        patches.Patch(color='blue', label=f'Regressed Proposals ({num_anchors_to_regress})')
    ]
    plt.legend(handles=handles)
    plt.title(f"Anchor Regression Visualization (First {num_anchors_to_regress} Anchors)")
    plt.show()

if __name__ == '__main__':
    visualize_anchor_regression()
