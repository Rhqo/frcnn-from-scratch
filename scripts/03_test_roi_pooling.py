

import torch
import sys
import os
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes

# Add the project root to the Python path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from frcnn.models.roi_pooling import RoIPooling
from frcnn.models.vgg16 import get_vgg16_base_net
from frcnn.models.rpn import RPN
from frcnn.models.proposal_layer import ProposalLayer
from frcnn.dataset import VOCDataset


def main():
    """
    Tests the RoIPooling layer.
    """
    print("--- Testing RoI Pooling Layer ---")

    # Path to the Pascal VOC dataset root (VOCdevkit)
    dataset_root = os.path.join(module_path, 'data', 'VOCdevkit')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    voc_dataset = VOCDataset(root_dir=dataset_root, split='trainval', transform=transform)

    # Fetch a sample from the dataset
    image_tensor, target = voc_dataset[5]  # Get the first sample
    original_image = transforms.ToPILImage()(image_tensor) # For visualization
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
    print(f"\nInput proposals (ROIs) shape: {rois.shape}")


    # Instantiate the RoIPooling layer
    output_size = (7, 7)
    spatial_scale = 1.0 / 16.0  # 1/16th of the input image size
    roi_pooling = RoIPooling(output_size, spatial_scale)
    print(f"\nSuccessfully loaded RoI Pooling Layer with output size {output_size}.")

    # Get the RoI-pooled features
    pooled_features = roi_pooling(feature_map, rois)

    # Print the shape of the output
    # The shape should be (n_rois, channels, output_height, output_width)
    print(f"\nPooled features shape: {pooled_features.shape}")

    # --- Visualization ---
    # Convert image tensor to uint8 for drawing bounding boxes
    image_uint8 = (image_tensor * 255).to(torch.uint8)

    # Extract bounding boxes from proposals (skip batch_index)
    # proposals are (batch_index, x1, y1, x2, y2)
    # We need (x1, y1, x2, y2)
    boxes_to_draw = rois[:, 1:]

    # Draw bounding boxes
    labels = [f"RoI {i+1}" for i in range(boxes_to_draw.shape[0])]

    drawn_image = draw_bounding_boxes(
        image=image_uint8,
        boxes=boxes_to_draw,
        labels=labels,
        colors="blue",
        width=2
    )

    # Convert to numpy array for matplotlib and permute dimensions (C, H, W) -> (H, W, C)
    drawn_image_np = drawn_image.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(drawn_image_np)
    plt.title('Image with RoIs (Proposals) for RoI Pooling')
    plt.axis('off')
    plt.show()

    print("--- Test Complete ---")


if __name__ == '__main__':
    main()

