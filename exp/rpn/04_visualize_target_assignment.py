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

def visualize_target_assignment():
    """
    Visualizes anchor target assignment using Matplotlib.
    """
    print("Step 4: Visualizing assign_targets_to_anchors() with Matplotlib")
    print("---")

    os.makedirs('exp/rpn', exist_ok=True)

    print("Loading configuration, model, and dataset...")
    config = FasterRCNNConfig()
    rpn = RegionProposalNetwork(config).train()

    # Load the first image and its ground truth targets
    test_dataset = VOCDataset(split='test', im_dir=config.im_test_path, ann_dir=config.ann_test_path)
    image_pil, targets, image_path = test_dataset.get_raw(0)
    gt_boxes = targets['bboxes']

    # Resize image and GT boxes
    w, h = image_pil.size
    scale = min(config.max_im_size / max(h, w), config.min_im_size / min(h, w))
    new_w, new_h = int(w * scale), int(h * scale)
    gt_boxes_resized = gt_boxes.float() * scale
    image_resized = image_pil.resize((new_w, new_h))

    image_tensor = torch.tensor(np.array(image_resized)).permute(2, 0, 1).float().unsqueeze(0)
    image_h, image_w = image_tensor.shape[-2:]

    # Create dummy feature map
    stride = 16
    feat_h, feat_w = image_h // stride, image_w // stride
    feat_tensor = torch.randn(1, config.backbone_out_channels, feat_h, feat_w)

    # 1. Generate anchors and assign targets
    anchors = rpn.generate_anchors(image_tensor, feat_tensor)
    labels, _ = rpn.assign_targets_to_anchors(anchors, gt_boxes_resized)

    # 2. Identify anchor types
    positive_anchors = anchors[labels == 1]
    negative_anchors = anchors[labels == 0]
    ignore_anchors = anchors[labels == -1]
    print(f"Anchor stats: {positive_anchors.shape[0]} positive, {negative_anchors.shape[0]} negative, {ignore_anchors.shape[0]} ignore")

    # 3. Visualize with Matplotlib
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image_resized)

    num_to_vis = 200 # Sample size for negative/ignore anchors

    # Draw Ground Truth boxes (White)
    for i in range(gt_boxes_resized.shape[0]):
        x1, y1, x2, y2 = gt_boxes_resized[i].numpy()
        w_gt, h_gt = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w_gt, h_gt, linewidth=4, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    # Draw a sample of Negative anchors (Red)
    if negative_anchors.shape[0] > 0:
        indices = np.random.choice(negative_anchors.shape[0], size=min(num_to_vis, negative_anchors.shape[0]), replace=False)
        for i in indices:
            x1, y1, x2, y2 = negative_anchors[i].numpy()
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=0.5, edgecolor='r', facecolor='none', alpha=0.7)
            ax.add_patch(rect)

    # Draw a sample of Ignore anchors (Yellow)
    if ignore_anchors.shape[0] > 0:
        indices = np.random.choice(ignore_anchors.shape[0], size=min(num_to_vis, ignore_anchors.shape[0]), replace=False)
        for i in indices:
            x1, y1, x2, y2 = ignore_anchors[i].numpy()
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=0.5, edgecolor='y', facecolor='none', alpha=0.7)
            ax.add_patch(rect)

    # Draw ALL Positive anchors (Green)
    for i in range(positive_anchors.shape[0]):
        x1, y1, x2, y2 = positive_anchors[i].numpy()
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    # Set plot limits
    ax.set_xlim(0, image_w)
    ax.set_ylim(image_h, 0)

    handles = [
        patches.Patch(color='blue', label='Ground Truth'),
        patches.Patch(color='green', label=f'Positive ({positive_anchors.shape[0]})'),
        patches.Patch(color='red', alpha=0.7, label=f'Negative ({min(num_to_vis, negative_anchors.shape[0])} sampled)'),
        patches.Patch(color='yellow', alpha=0.7, label=f'Ignore ({min(num_to_vis, ignore_anchors.shape[0])} sampled)')
    ]
    plt.legend(handles=handles)
    plt.title("Target Assignment Visualization")
    plt.show()

# Helper method for VOCDataset
def get_raw(self, index):
    im_info = self.images_info[index]
    im = Image.open(im_info['filename'])
    targets = {}
    targets['bboxes'] = torch.as_tensor([detection['bbox'] for detection in im_info['detections']])
    targets['labels'] = torch.as_tensor([detection['label'] for detection in im_info['detections']])
    return im, targets, im_info['filename']

VOCDataset.get_raw = get_raw

if __name__ == '__main__':
    visualize_target_assignment()
