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
from config.voc_config import FasterRCNNConfig
from dataset.voc_loader import VOCDataset

def visualize_anchors():
    """
    Generates and visualizes anchors using Matplotlib to show anchors outside image boundaries.
    """
    print("Step 1: Visualizing generate_anchors() with Matplotlib")
    print("---")

    os.makedirs('exp/rpn', exist_ok=True)

    print("Loading configuration, model, and dataset...")
    config = FasterRCNNConfig()
    rpn = RegionProposalNetwork(config).eval()
    
    # Load the first image from the test dataset
    test_dataset = VOCDataset(split='test', im_dir=config.im_test_path, ann_dir=config.ann_test_path)
    image_tensor, _, image_path = test_dataset[0]
    image_tensor = image_tensor.unsqueeze(0) # Add batch dimension

    # Load image with PIL for visualization
    image = Image.open(image_path).convert("RGB")
    image_h_orig, image_w_orig = image.height, image.width
    
    # Resize image to match tensor dimensions for display
    image_h, image_w = image_tensor.shape[-2:]
    image_resized = image.resize((image_w, image_h))

    # Create a dummy feature map based on the image tensor size
    stride = 16
    feat_h, feat_w = image_h // stride, image_w // stride
    feat_tensor = torch.randn(1, config.backbone_out_channels, feat_h, feat_w)

    print(f"Using first image from test set: {os.path.basename(image_path)}")
    print(f"Image tensor dimensions (N, C, H, W): {image_tensor.shape}")

    # 1. Generate anchors
    print("Calling rpn.generate_anchors(image, feat)...")
    anchors = rpn.generate_anchors(image_tensor, feat_tensor)
    print(f"Total anchors generated: {anchors.shape[0]}")

    # 2. Visualize with Matplotlib
    num_anchors_to_vis = 300
    random_indices = np.random.choice(anchors.shape[0], size=num_anchors_to_vis, replace=False)
    sampled_anchors = anchors[random_indices]

    print(f"Visualizing a random sample of {num_anchors_to_vis} anchors (green).")

    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image_resized)

    for i in range(sampled_anchors.shape[0]):
        x1, y1, x2, y2 = sampled_anchors[i].numpy()
        w, h = x2 - x1, y2 - y1
        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='g', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

    # Set plot limits to see anchors outside the image
    # Find min/max coordinates of all anchors to define the canvas size
    min_x = anchors[:, 0].min().item()
    min_y = anchors[:, 1].min().item()
    max_x = anchors[:, 2].max().item()
    max_y = anchors[:, 3].max().item()

    # Give a little padding
    padding = 100
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(max_y + padding, min_y - padding) # Flipped for image coordinates

    plt.title(f"Visualizing {num_anchors_to_vis} Sampled Anchors")
    plt.show()

if __name__ == '__main__':
    visualize_anchors()
