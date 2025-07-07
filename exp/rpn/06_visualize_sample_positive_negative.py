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
from model.utils import sample_positive_negative

def visualize_sample_positive_negative():
    """
    Visualizes the sampling of positive and negative anchors using Matplotlib.
    """
    print("Step 6: Visualizing sample_positive_negative() with Matplotlib")
    print("---")

    os.makedirs('exp/rpn', exist_ok=True)

    print("Loading configuration, model, and dataset...")
    config = FasterRCNNConfig()
    rpn = RegionProposalNetwork(config).train() # Use train mode for target assignment and sampling

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

    # 2. Sample positive and negative anchors
    positive_count = rpn.rpn_pos_count
    total_count = rpn.rpn_batch_size
    sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_negative(
        labels,
        positive_count=positive_count,
        total_count=total_count
    )

    sampled_positive_anchors = anchors[sampled_pos_idx_mask]
    sampled_negative_anchors = anchors[sampled_neg_idx_mask]
    
    # Anchors that were ignored during target assignment (label == -1)
    ignored_anchors_from_assignment = anchors[labels == -1]

    print(f"Total anchors: {anchors.shape[0]}")
    print(f"Assigned positive anchors: {torch.where(labels == 1)[0].shape[0]}")
    print(f"Assigned negative anchors: {torch.where(labels == 0)[0].shape[0]}")
    print(f"Assigned ignored anchors: {torch.where(labels == -1)[0].shape[0]}")
    print(f"Sampled positive anchors: {sampled_positive_anchors.shape[0]}")
    print(f"Sampled negative anchors: {sampled_negative_anchors.shape[0]}")

    # 3. Visualize with Matplotlib
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image_resized)

    # Draw Ground Truth boxes (Blue)
    for i in range(gt_boxes_resized.shape[0]):
        x1, y1, x2, y2 = gt_boxes_resized[i].numpy()
        w_gt, h_gt = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w_gt, h_gt, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    # Draw Sampled Positive anchors (Green)
    for i in range(sampled_positive_anchors.shape[0]):
        x1, y1, x2, y2 = sampled_positive_anchors[i].numpy()
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    # Draw Sampled Negative anchors (Red)
    for i in range(sampled_negative_anchors.shape[0]):
        x1, y1, x2, y2 = sampled_negative_anchors[i].numpy()
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=0.5, edgecolor='r', facecolor='none', alpha=0.7)
        ax.add_patch(rect)

    # Set plot limits
    ax.set_xlim(0, image_w)
    ax.set_ylim(image_h, 0)

    handles = [
        patches.Patch(color='blue', label='Ground Truth'),
        patches.Patch(color='green', label=f'Sampled Positive ({sampled_positive_anchors.shape[0]})'),
        patches.Patch(color='red', alpha=0.7, label=f'Sampled Negative ({sampled_negative_anchors.shape[0]})')
    ]
    plt.legend(handles=handles)
    plt.title("Sampled Positive and Negative Anchors")
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
    visualize_sample_positive_negative()