
import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_l1_loss(bbox_preds, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0):
    """
    Smooth L1 loss for bounding box regression.

    Args:
        bbox_preds (torch.Tensor): Predicted bounding box regression values.
        bbox_targets (torch.Tensor): Ground truth bounding box regression targets.
        bbox_inside_weights (torch.Tensor): Weights for the inside of the bounding box.
        bbox_outside_weights (torch.Tensor): Weights for the outside of the bounding box.
        sigma (float): Parameter for the Smooth L1 loss.

    Returns:
        torch.Tensor: The computed Smooth L1 loss.
    """
    sigma_2 = sigma ** 2
    box_diff = bbox_preds - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < (1.0 / sigma_2)).float()

    loss_box = (
        (in_box_diff ** 2) * 0.5 * sigma_2 * smoothL1_sign
        + (abs_in_box_diff - (0.5 / sigma_2)) * (1.0 - smoothL1_sign)
    )
    loss_box = loss_box * bbox_outside_weights

    return loss_box.sum()


def cross_entropy_loss(cls_scores, cls_targets):
    """
    Cross-entropy loss for classification.

    Args:
        cls_scores (torch.Tensor): Predicted classification scores.
        cls_targets (torch.Tensor): Ground truth class labels.

    Returns:
        torch.Tensor: The computed cross-entropy loss.
    """
    return F.cross_entropy(cls_scores, cls_targets, ignore_index=-1)


if __name__ == '__main__':
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
