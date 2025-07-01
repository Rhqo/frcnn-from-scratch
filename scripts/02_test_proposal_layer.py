
import torch
import sys
import os

# Add the project root to the Python path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from frcnn.models.vgg16 import get_vgg16_base_net
from frcnn.models.rpn import RPN
from frcnn.models.proposal_layer import ProposalLayer


def main():
    """
    Tests the ProposalLayer.
    """
    print("--- Testing Proposal Layer ---")

    # Get the VGG16 base network
    base_net = get_vgg16_base_net()

    # Create a dummy input image
    dummy_image = torch.randn(1, 3, 600, 800)
    img_size = (600, 800)
    print(f"Input image shape: {dummy_image.shape}")

    # Get the feature map from the base network
    feature_map = base_net(dummy_image)
    print(f"Feature map shape: {feature_map.shape}")

    # Instantiate the RPN
    rpn = RPN(in_channels=512, mid_channels=512, n_anchor=9)
    print("\nSuccessfully loaded RPN.")

    # Pass the feature map through the RPN
    rpn_cls_scores, rpn_bbox_preds = rpn(feature_map)
    print(f"\nRPN classification scores shape: {rpn_cls_scores.shape}")
    print(f"RPN bounding box predictions shape: {rpn_bbox_preds.shape}")

    # Instantiate the ProposalLayer
    proposal_layer = ProposalLayer()
    print("\nSuccessfully loaded Proposal Layer.")

    # Get the proposals
    proposals = proposal_layer(rpn_cls_scores, rpn_bbox_preds, img_size)

    # Print the shape of the proposals
    # The shape should be (n_proposals, 5), where the 5 columns are
    # (batch_index, x1, y1, x2, y2)
    print(f"\nProposals shape: {proposals.shape}")
    print("--- Test Complete ---")


if __name__ == '__main__':
    main()
