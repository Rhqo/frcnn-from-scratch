import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Add root directory to path to import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from model.utils import get_iou
from config.voc_config import FasterRCNNConfig
from dataset.voc_loader import VOCDataset

def visualize_iou():
    """
    Visualizes the Intersection over Union (IoU) calculation between two sets of bounding boxes.
    """
    print("Step 1: Visualizing get_iou()")
    print("---")

    os.makedirs('exp/utils', exist_ok=True)

    print("Loading configuration and dataset...")
    config = FasterRCNNConfig()

    # Load the first image and its ground truth targets from the test dataset
    test_dataset = VOCDataset(split='test', im_dir=config.im_test_path, ann_dir=config.ann_test_path)
    image_pil, targets, image_path = test_dataset.get_raw(0)
    gt_boxes = targets['bboxes']

    # Resize image and GT boxes to match model input
    w, h = image_pil.size
    scale = min(config.max_im_size / max(h, w), config.min_im_size / min(h, w))
    new_w, new_h = int(w * scale), int(h * scale)
    gt_boxes_resized = gt_boxes.float() * scale
    image_resized = image_pil.resize((new_w, new_h))

    image_h, image_w = new_h, new_w

    print(f"Using first image from test set: {os.path.basename(image_path)}")
    print(f"Resized image dimensions: ({image_h}, {image_w})")

    # 1. Define two sets of boxes
    # Box Set 1: Ground Truth boxes (or a subset of them)
    boxes1 = gt_boxes_resized[:min(5, gt_boxes_resized.shape[0])].clone()

    # Box Set 2: Some arbitrary proposals for demonstration
    # Let's create some proposals that overlap with GT and some that don't
    boxes2 = torch.tensor([
        [50, 50, 150, 150],  # Overlaps with first GT
        [100, 100, 200, 200], # Overlaps with first GT
        [300, 300, 400, 400], # No overlap
        [gt_boxes_resized[0,0]+10, gt_boxes_resized[0,1]+10, gt_boxes_resized[0,2]-10, gt_boxes_resized[0,3]-10], # High overlap
        [gt_boxes_resized[0,0]-50, gt_boxes_resized[0,1]-50, gt_boxes_resized[0,2]+50, gt_boxes_resized[0,3]+50], # Contains GT
    ], dtype=torch.float32)
    # Adjust boxes2 to be within image bounds if they are not already
    boxes2[:, 0] = torch.clamp(boxes2[:, 0], 0, image_w)
    boxes2[:, 1] = torch.clamp(boxes2[:, 1], 0, image_h)
    boxes2[:, 2] = torch.clamp(boxes2[:, 2], 0, image_w)
    boxes2[:, 3] = torch.clamp(boxes2[:, 3], 0, image_h)

    print(f"Box Set 1 (GT): {boxes1.shape[0]} boxes")
    print(f"Box Set 2 (Proposals): {boxes2.shape[0]} boxes")

    # 2. Calculate IoU matrix
    iou_matrix = get_iou(boxes1, boxes2)
    print("IoU Matrix:")
    print(iou_matrix.numpy())

    # 3. Visualize with Matplotlib
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image_resized)

    # Draw Box Set 1 (GT) in White
    for i, box in enumerate(boxes1):
        x1, y1, x2, y2 = box.numpy()
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='w', facecolor='none', label='GT Boxes')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f'GT {i}', color='white', fontsize=10)

    # Draw Box Set 2 (Proposals) in Cyan
    for i, box in enumerate(boxes2):
        x1, y1, x2, y2 = box.numpy()
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=1.5, edgecolor='cyan', facecolor='none', linestyle='--', label='Proposals')
        ax.add_patch(rect)
        ax.text(x1, y1 + h + 10, f'Prop {i}', color='cyan', fontsize=10)

    # Display IoU values between overlapping boxes
    for i in range(iou_matrix.shape[0]):
        for j in range(iou_matrix.shape[1]):
            iou_val = iou_matrix[i, j].item()
            if iou_val > 0.01: # Only show for significant overlaps
                # Calculate center of intersection for text placement
                box1 = boxes1[i]
                box2 = boxes2[j]
                
                x_left = max(box1[0], box2[0])
                y_top = max(box1[1], box2[1])
                x_right = min(box1[2], box2[2])
                y_bottom = min(box1[3], box2[3])
                
                if x_right > x_left and y_bottom > y_top:
                    center_x = (x_left + x_right) / 2
                    center_y = (y_top + y_bottom) / 2
                    ax.text(center_x, center_y, f'{iou_val:.2f}', color='magenta', fontsize=9, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    ax.set_xlim(0, image_w)
    ax.set_ylim(image_h, 0)

    handles = [
        patches.Patch(color='white', label='Ground Truth Boxes'),
        patches.Patch(color='cyan', linestyle='--', label='Proposal Boxes')
    ]
    plt.legend(handles=handles)
    plt.title("IoU Visualization")
    plt.show()

if __name__ == '__main__':
    visualize_iou()
