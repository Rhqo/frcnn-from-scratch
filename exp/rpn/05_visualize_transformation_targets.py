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
from model.utils import boxes_to_transformation_targets, apply_regression_pred_to_anchors_or_proposals
from config.voc_config import FasterRCNNConfig
from dataset.voc_loader import VOCDataset

def visualize_transformation_targets():
    """
    Visualizes transformation target calculation and application using Matplotlib.
    """
    print("Step 5: Visualizing boxes_to_transformation_targets() with Matplotlib")
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

    # 1. Find positive anchors
    anchors = rpn.generate_anchors(image_tensor, feat_tensor)
    labels, matched_gt_boxes = rpn.assign_targets_to_anchors(anchors, gt_boxes_resized)
    positive_anchor_indices = torch.where(labels == 1)[0]
    positive_anchors = anchors[positive_anchor_indices]
    gt_boxes_for_pos_anchors = matched_gt_boxes[positive_anchor_indices]

    # 2. Calculate transformation targets
    regression_targets = boxes_to_transformation_targets(gt_boxes_for_pos_anchors, positive_anchors)

    # 3. Verify by applying the targets back to the anchors
    reconstructed_boxes = apply_regression_pred_to_anchors_or_proposals(regression_targets.unsqueeze(1), positive_anchors)
    reconstructed_boxes = reconstructed_boxes.squeeze(1)

    # 4. Visualize with Matplotlib
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image_resized)

    # Draw Ground Truth boxes (Blue)
    for i in range(gt_boxes_resized.shape[0]):
        x1, y1, x2, y2 = gt_boxes_resized[i].numpy()
        w_gt, h_gt = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w_gt, h_gt, linewidth=3, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    # Draw Positive Anchors (Green)
    for i in range(positive_anchors.shape[0]):
        x1, y1, x2, y2 = positive_anchors[i].numpy()
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=0.5, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    # Draw Reconstructed boxes (Yellow)
    for i in range(reconstructed_boxes.shape[0]):
        x1, y1, x2, y2 = reconstructed_boxes[i].detach().numpy()
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='y', facecolor='none')
        ax.add_patch(rect)

        # Draw a line from the center of the positive anchor to the center of the reconstructed box
        anchor_center_x = (positive_anchors[i, 0] + positive_anchors[i, 2]) / 2
        anchor_center_y = (positive_anchors[i, 1] + positive_anchors[i, 3]) / 2
        reconstructed_center_x = (reconstructed_boxes[i, 0] + reconstructed_boxes[i, 2]) / 2
        reconstructed_center_y = (reconstructed_boxes[i, 1] + reconstructed_boxes[i, 3]) / 2
        ax.plot([anchor_center_x, reconstructed_center_x], [anchor_center_y, reconstructed_center_y], 'r--', linewidth=0.8)

    ax.set_xlim(0, image_w)
    ax.set_ylim(image_h, 0)

    handles = [
        patches.Patch(color='blue', label='Ground Truth'),
        patches.Patch(color='green', label=f'Positive Anchors ({positive_anchors.shape[0]})'),
        patches.Patch(color='yellow', label='Reconstructed from Targets')
    ]
    plt.legend(handles=handles)
    plt.title("Transformation Targets Verification")
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
    visualize_transformation_targets()
