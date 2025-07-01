

import torch
import sys
import os

# Add the project root to the Python path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from frcnn.models.fast_rcnn_head import FastRCNNHead


def main():
    """
    Tests the FastRCNNHead.
    """
    print("--- Testing Fast R-CNN Head ---")

    # Define parameters for the head
    in_channels = 512  # From VGG16 feature map
    num_classes = 21   # 20 Pascal VOC classes + 1 background class
    roi_output_size = (7, 7)

    # Create dummy RoI-pooled features
    # Shape: (n_rois, in_channels, roi_output_height, roi_output_width)
    n_rois = 128  # Example number of proposals
    dummy_pooled_features = torch.randn(n_rois, in_channels, roi_output_size[0], roi_output_size[1])
    print(f"Input RoI-pooled features shape: {dummy_pooled_features.shape}")

    # Instantiate the FastRCNNHead
    fast_rcnn_head = FastRCNNHead(in_channels, num_classes, roi_output_size)
    print("\nSuccessfully loaded Fast R-CNN Head.")

    # Pass the dummy pooled features through the head
    cls_scores, bbox_preds = fast_rcnn_head(dummy_pooled_features)

    # Print the shapes of the output tensors
    # cls_scores should be (n_rois, num_classes)
    # bbox_preds should be (n_rois, num_classes * 4)
    print(f"\nClassification scores shape: {cls_scores.shape}")
    print(f"Bounding box predictions shape: {bbox_preds.shape}")
    print("--- Test Complete ---")


if __name__ == '__main__':
    main()

