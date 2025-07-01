import torch
import sys
import os

# Add the project root to the Python path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from frcnn.models.faster_rcnn import FasterRCNN


def main():
    """
    Tests the full FasterRCNN model forward pass.
    """
    print("--- Testing Full Faster R-CNN Model ---")

    # Create a dummy input image
    dummy_image = torch.randn(1, 3, 600, 800)
    img_size = (600, 800)
    print(f"Input image shape: {dummy_image.shape}")

    # Instantiate the FasterRCNN model
    # num_classes = 21 (20 VOC classes + background)
    model = FasterRCNN(num_classes=21)
    print("\nSuccessfully loaded Faster R-CNN model.")

    # Perform a forward pass
    model.eval() # Set to evaluation mode for consistent behavior
    rpn_cls_scores, rpn_bbox_preds, roi_cls_scores, roi_bbox_preds = model(dummy_image, img_size)

    # Print the shapes of the outputs
    print(f"\nRPN classification scores shape: {rpn_cls_scores.shape}")
    print(f"RPN bounding box predictions shape: {rpn_bbox_preds.shape}")
    print(f"RoI classification scores shape: {roi_cls_scores.shape}")
    print(f"RoI bounding box predictions shape: {roi_bbox_preds.shape}")
    print("--- Test Complete ---")


if __name__ == '__main__':
    main()
