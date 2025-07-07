import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Add root directory to path to import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from model.utils import clamp_boxes_to_image_boundary
from config.voc_config import FasterRCNNConfig
from dataset.voc_loader import VOCDataset

def visualize_clamp_boxes():
    """
    Visualizes the effect of clamping bounding boxes to image boundaries.
    """
    print("Step 2: Visualizing clamp_boxes_to_image_boundary()")
    print("---")

    os.makedirs('exp/utils', exist_ok=True)

    print("Loading configuration and dataset...")
    config = FasterRCNNConfig()

    # Load the first image from the test dataset
    test_dataset = VOCDataset(split='test', im_dir=config.im_test_path, ann_dir=config.ann_test_path)
    image_pil, _, image_path = test_dataset.get_raw(0)

    # Resize image to match model input dimensions
    w, h = image_pil.size
    scale = min(config.max_im_size / max(h, w), config.min_im_size / min(h, w))
    new_w, new_h = int(w * scale), int(h * scale)
    image_resized = image_pil.resize((new_w, new_h))

    image_h, image_w = new_h, new_w

    print(f"Using first image from test set: {os.path.basename(image_path)}")
    print(f"Resized image dimensions: ({image_h}, {image_w})")

    # 1. Create some boxes that go outside the image boundaries
    boxes_before_clamp = torch.tensor([
        [-50, -50, 100, 100],  # Top-left out
        [image_w - 50, image_h - 50, image_w + 50, image_h + 50], # Bottom-right out
        [image_w / 2 - 100, -50, image_w / 2 + 100, 50], # Top out
        [-50, image_h / 2 - 100, 50, image_h / 2 + 100], # Left out
        [10, 10, image_w - 10, image_h - 10] # Fully inside
    ], dtype=torch.float32)

    print(f"Boxes before clamping: {boxes_before_clamp.shape[0]}")

    # 2. Clamp the boxes
    image_shape_tensor = torch.tensor([1, 3, image_h, image_w]) # Dummy shape for function
    boxes_after_clamp = clamp_boxes_to_image_boundary(boxes_before_clamp, image_shape_tensor)

    print(f"Boxes after clamping: {boxes_after_clamp.shape[0]}")

    # 3. Visualize with Matplotlib
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image_resized)

    # Draw original boxes (Red, dashed)
    for i, box in enumerate(boxes_before_clamp):
        x1, y1, x2, y2 = box.numpy()
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=1.5, edgecolor='r', facecolor='none', linestyle='--', label='Before Clamp')
        ax.add_patch(rect)
        ax.text(x1, y1 - 10, f'Original {i}', color='red', fontsize=9)

    # Draw clamped boxes (Green, solid)
    for i, box in enumerate(boxes_after_clamp):
        x1, y1, x2, y2 = box.numpy()
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='g', facecolor='none', label='After Clamp')
        ax.add_patch(rect)
        ax.text(x1, y1 + h + 5, f'Clamped {i}', color='green', fontsize=9)

    # Set plot limits to show the full extent of original boxes
    min_x = min(boxes_before_clamp[:, 0].min().item(), boxes_after_clamp[:, 0].min().item())
    min_y = min(boxes_before_clamp[:, 1].min().item(), boxes_after_clamp[:, 1].min().item())
    max_x = max(boxes_before_clamp[:, 2].max().item(), boxes_after_clamp[:, 2].max().item())
    max_y = max(boxes_before_clamp[:, 3].max().item(), boxes_after_clamp[:, 3].max().item())
    padding = 50
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(max_y + padding, min_y - padding)

    handles = [
        patches.Patch(color='red', linestyle='--', label='Boxes Before Clamping'),
        patches.Patch(color='green', label='Boxes After Clamping')
    ]
    plt.legend(handles=handles)
    plt.title("Clamp Boxes to Image Boundary Visualization")
    plt.show()

if __name__ == '__main__':
    visualize_clamp_boxes()
