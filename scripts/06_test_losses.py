
import torch
import sys
import os

# Add the project root to the Python path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from frcnn.losses import smooth_l1_loss, cross_entropy_loss


def main():
    """
    Tests the loss functions.
    """
    print("--- Testing Loss Functions ---")

    # Test Smooth L1 Loss
    bbox_preds = torch.randn(10, 4)
    bbox_targets = torch.randn(10, 4)
    bbox_inside_weights = torch.ones(10, 4)
    bbox_outside_weights = torch.ones(10, 4)
    loss_l1 = smooth_l1_loss(bbox_preds, bbox_targets, bbox_inside_weights, bbox_outside_weights)
    print(f"Smooth L1 Loss: {loss_l1.item():.4f}")

    # Test Cross-Entropy Loss
    cls_scores = torch.randn(10, 2) # 10 samples, 2 classes (foreground/background)
    cls_targets = torch.randint(0, 2, (10,)) # Random labels 0 or 1
    loss_ce = cross_entropy_loss(cls_scores, cls_targets)
    print(f"Cross-Entropy Loss: {loss_ce.item():.4f}")

    print("--- Test Complete ---")


if __name__ == '__main__':
    main()
