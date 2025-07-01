

import torch
import sys
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Add the project root to the Python path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from frcnn.models.fast_rcnn_head import FastRCNNHead
from frcnn.models.vgg16 import get_vgg16_base_net
from frcnn.models.rpn import RPN
from frcnn.models.proposal_layer import ProposalLayer
from frcnn.models.roi_pooling import RoIPooling
from frcnn.dataset import VOCDataset, VOC_CLASSES


def main():
    """
    Tests the FastRCNNHead.
    """
    print("--- Testing Fast R-CNN Head ---")

    # Path to the Pascal VOC dataset root (VOCdevkit)
    dataset_root = os.path.join(module_path, 'data', 'VOCdevkit')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    voc_dataset = VOCDataset(root_dir=dataset_root, split='trainval', transform=transform)

    # Fetch a sample from the dataset
    image_tensor, target = voc_dataset[5]  # Get the first sample
    dummy_image = image_tensor.unsqueeze(0) # Add batch dimension
    img_size = image_tensor.shape[1:] # Get H, W from the image tensor
    print(f"Input image shape: {dummy_image.shape}")

    # Get the VGG16 base network
    base_net = get_vgg16_base_net()

    # Get the feature map from the base network
    feature_map = base_net(dummy_image)
    print(f"Feature map shape: {feature_map.shape}")

    # Instantiate the RPN
    rpn = RPN(in_channels=512, mid_channels=512, n_anchor=9)

    # Pass the feature map through the RPN
    rpn_cls_scores, rpn_bbox_preds = rpn(feature_map)

    # Instantiate the ProposalLayer
    proposal_layer = ProposalLayer()

    # Get the proposals (these will be the ROIs for RoI Pooling)
    rois = proposal_layer(rpn_cls_scores, rpn_bbox_preds, img_size)
    print(f"\nGenerated ROIs shape: {rois.shape}")

    # Instantiate the RoIPooling layer
    roi_output_size = (7, 7)
    spatial_scale = 1.0 / 16.0  # 1/16th of the input image size
    roi_pooling = RoIPooling(roi_output_size, spatial_scale)

    # Get the RoI-pooled features
    pooled_features = roi_pooling(feature_map, rois)
    print(f"Input RoI-pooled features shape: {pooled_features.shape}")

    # Define parameters for the head
    in_channels = 512  # From VGG16 feature map
    num_classes = 21   # 20 Pascal VOC classes + 1 background class

    # Instantiate the FastRCNNHead
    fast_rcnn_head = FastRCNNHead(in_channels, num_classes, roi_output_size)
    print("\nSuccessfully loaded Fast R-CNN Head.")

    # Pass the pooled features through the head
    cls_scores, bbox_preds = fast_rcnn_head(pooled_features)

    # Print the shapes of the output tensors
    print(f"\nClassification scores shape: {cls_scores.shape}")
    print(f"Bounding box predictions shape: {bbox_preds.shape}")

    # --- Visualization (Text-based for now) ---
    print("\n--- Top Predicted Classes and Scores for a few ROIs ---")
    # Take a few random ROIs for inspection
    num_rois_to_inspect = min(5, cls_scores.shape[0])
    random_indices = torch.randperm(cls_scores.shape[0])[:num_rois_to_inspect]

    for i, idx in enumerate(random_indices):
        scores = torch.softmax(cls_scores[idx], dim=0)
        top_score, top_label_idx = torch.max(scores, dim=0)
        class_name = VOC_CLASSES[top_label_idx.item()]
        print(f"RoI {i+1} (Original Index: {idx.item()}): Class='{class_name}', Score={top_score.item():.4f}")

    print("--- Test Complete ---")


if __name__ == '__main__':
    main()

