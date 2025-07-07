import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Add root directory to path to import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from model.rpn import RegionProposalNetwork
from model.roi_head import ROIHead
from model.utils import apply_regression_pred_to_anchors_or_proposals
from config.voc_config import FasterRCNNConfig
from dataset.voc_loader import VOCDataset

def visualize_filtered_detections():
    """
    Visualizes the filter_predictions process within the ROIHead.
    Shows predicted boxes before and after filtering (low score, small size, NMS, topK).
    """
    print("Step 2: Visualizing filter_predictions() for ROIHead")
    print("---")

    os.makedirs('exp/roi_head', exist_ok=True)

    print("Loading configuration, models, and dataset...")
    config = FasterRCNNConfig()
    rpn = RegionProposalNetwork(config).eval() # For generating proposals
    roi_head = ROIHead(config).eval() # In eval mode for filter_predictions

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

    # 1. Simulate proposals (from RPN output)
    anchors = rpn.generate_anchors(image_tensor, feat_tensor)
    num_anchors = anchors.shape[0]
    box_transform_pred_rpn = torch.randn(num_anchors, 1, 4) * 0.1
    initial_proposals = apply_regression_pred_to_anchors_or_proposals(box_transform_pred_rpn, anchors).squeeze(1)
    cls_scores_rpn = torch.randn(num_anchors, 1) * 2 - 1
    proposals, _ = rpn.filter_proposals(initial_proposals.detach(), cls_scores_rpn.detach(), image_tensor.shape)
    print(f"Simulated {proposals.shape[0]} proposals from RPN.")

    # 2. Simulate ROIHead's raw predictions (before filter_predictions)
    # These would come from the cls_layer and bbox_reg_layer of ROIHead
    num_proposals = proposals.shape[0]
    num_classes = config.num_classes

    # Create dummy box transformation predictions for each class
    # (num_proposals, num_classes, 4)
    dummy_box_transform_pred = torch.randn(num_proposals, num_classes, 4) * 0.1
    # Apply these transformations to the proposals to get raw predicted boxes
    raw_pred_boxes = apply_regression_pred_to_anchors_or_proposals(dummy_box_transform_pred, proposals)

    # Create dummy classification scores (logits) for each class
    # (num_proposals, num_classes)
    dummy_cls_scores = torch.randn(num_proposals, num_classes) * 2 - 1
    # Boost scores for some classes/proposals to simulate detections
    # For simplicity, let's boost scores for the first few proposals for a random class
    if num_proposals > 0:
        for i in range(min(50, num_proposals)):
            random_class_idx = np.random.randint(1, num_classes) # Avoid background class (0)
            dummy_cls_scores[i, random_class_idx] += 5.0 # High score for a specific class
            # Also make some boxes overlap to test NMS
            if i > 0 and i % 5 == 0:
                raw_pred_boxes[i, random_class_idx, :] = raw_pred_boxes[i-1, random_class_idx, :] + torch.randn(4) * 5 # Slightly shifted overlap

    # Convert logits to probabilities
    raw_pred_scores = F.softmax(dummy_cls_scores, dim=-1)

    # Create raw predicted labels (each proposal has a prediction for each class)
    raw_pred_labels = torch.arange(num_classes).view(1, -1).expand(num_proposals, -1)

    # Reshape for filter_predictions input: flatten (num_proposals * num_classes) predictions
    # Exclude background class (index 0)
    pred_boxes_before_filter = raw_pred_boxes[:, 1:].reshape(-1, 4)
    pred_scores_before_filter = raw_pred_scores[:, 1:].reshape(-1)
    pred_labels_before_filter = raw_pred_labels[:, 1:].reshape(-1)

    print(f"Simulated {pred_boxes_before_filter.shape[0]} raw predictions before filtering.")

    # 3. Apply filter_predictions
    print("Calling roi_head.filter_predictions()...")
    filtered_boxes, filtered_labels, filtered_scores = roi_head.filter_predictions(
        pred_boxes_before_filter.detach(), 
        pred_labels_before_filter.detach(), 
        pred_scores_before_filter.detach()
    )
    print(f"After filtering, {filtered_boxes.shape[0]} final detections remain.")

    # 4. Visualize with Matplotlib
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image_resized)

    # Draw Ground Truth boxes (White)
    for i in range(gt_boxes_resized.shape[0]):
        x1, y1, x2, y2 = gt_boxes_resized[i].numpy()
        w_gt, h_gt = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w_gt, h_gt, linewidth=2, edgecolor='w', facecolor='none')
        ax.add_patch(rect)

    # Draw predictions *before* filtering (Yellow, semi-transparent)
    # Sample a subset for clarity if there are too many
    num_before_vis = min(1000, pred_boxes_before_filter.shape[0])
    if num_before_vis > 0:
        indices = np.random.choice(pred_boxes_before_filter.shape[0], size=num_before_vis, replace=False)
        for i in indices:
            x1, y1, x2, y2 = pred_boxes_before_filter[i].numpy()
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=0.5, edgecolor='y', facecolor='none', alpha=0.3)
            ax.add_patch(rect)

    # Draw final filtered detections (Blue, opaque)
    for i in range(filtered_boxes.shape[0]):
        x1, y1, x2, y2 = filtered_boxes[i].numpy()
        w, h = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        # Optionally add score/label
        # ax.text(x1, y1 - 5, f'{filtered_labels[i].item()}: {filtered_scores[i].item():.2f}', color='blue', fontsize=8)

    # Set plot limits
    all_boxes = torch.cat([gt_boxes_resized, pred_boxes_before_filter, filtered_boxes], dim=0)
    min_x, min_y = all_boxes[:, 0].min().item(), all_boxes[:, 1].min().item()
    max_x, max_y = all_boxes[:, 2].max().item(), all_boxes[:, 3].max().item()
    padding = 100
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(max_y + padding, min_y - padding)

    handles = [
        patches.Patch(color='white', label='Ground Truth'),
        patches.Patch(color='yellow', alpha=0.3, label=f'Raw Predictions ({num_before_vis} sampled)'),
        patches.Patch(color='blue', label=f'Filtered Detections ({filtered_boxes.shape[0]})')
    ]
    plt.legend(handles=handles)
    plt.title("ROIHead Filtered Detections Visualization")
    plt.show()

if __name__ == '__main__':
    visualize_filtered_detections()
