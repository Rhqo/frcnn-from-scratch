
import torch
import sys
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes

# Add the project root to the Python path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from frcnn.models.vgg16 import get_vgg16_base_net
from frcnn.models.rpn import RPN
from frcnn.models.proposal_layer import ProposalLayer
from frcnn.dataset import VOCDataset


def main():
    """
    Tests the ProposalLayer.
    """
    print("--- Testing Proposal Layer ---")

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

    # --- Visualization ---
    # Convert image tensor to uint8 for drawing bounding boxes
    image_uint8 = (image_tensor * 255).to(torch.uint8)

    # Extract bounding boxes from proposals (skip batch_index)
    # proposals are (batch_index, x1, y1, x2, y2)
    # We need (x1, y1, x2, y2)
    boxes_to_draw = proposals[:, 1:]

    # Draw bounding boxes
    # Labels are optional, but we can add dummy labels if needed
    labels = [f"Proposal {i+1}" for i in range(boxes_to_draw.shape[0])]

    drawn_image = draw_bounding_boxes(
        image=image_uint8,
        boxes=boxes_to_draw,
        labels=labels,
        colors="green",
        width=2
    )

    # Convert to numpy array for matplotlib and permute dimensions (C, H, W) -> (H, W, C)
    drawn_image_np = drawn_image.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(drawn_image_np)
    plt.title('Image with Region Proposals')
    plt.axis('off')
    plt.show()

    print("--- Test Complete ---")


if __name__ == '__main__':
    main()
