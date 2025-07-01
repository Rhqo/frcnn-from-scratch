

import torch
import sys
import os
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm # Import colormap
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights # Import pre-trained model
from torchvision.models.detection.image_list import ImageList # Import ImageList

# Add the project root to the Python path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from frcnn.models.roi_pooling import RoIPooling

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

    # Load pre-trained Faster R-CNN model
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    pretrained_model = fasterrcnn_resnet50_fpn(weights=weights)
    pretrained_model.eval() # Set to evaluation mode
    print("Successfully loaded pre-trained Faster R-CNN ResNet50 FPN model.")

    # Use backbone and RPN from pre-trained model
    base_net = pretrained_model.backbone
    rpn = pretrained_model.rpn

    # Create ImageList for torchvision RPN
    image_list = ImageList(dummy_image, [(img_size[0], img_size[1])])

    # Get the feature map from the base network
    feature_maps = base_net(dummy_image) # This will be an OrderedDict
    print(f"Feature map keys: {feature_maps.keys()}")

    # Pass the feature map through the RPN to get proposals
    # Note: torchvision RPN directly returns proposals and losses
    rois_list, _ = rpn(image_list, feature_maps)
    rois = rois_list[0] # Take the first element from the list (batch_size=1)
    print(f"Input proposals (ROIs) shape: {rois.shape}")


    # Instantiate the RoIPooling layer
    output_size = (7, 7)
    spatial_scale = 1.0 / 16.0  # 1/16th of the input image size
    roi_pooling = RoIPooling(output_size, spatial_scale)
    print(f"Successfully loaded RoI Pooling Layer with output size {output_size}.")

    # Add batch index to rois for custom RoIPooling layer
    # torchvision rois are (x1, y1, x2, y2), custom RoIPooling expects (batch_index, x1, y1, x2, y2)
    batch_indices = torch.zeros((rois.shape[0], 1), dtype=rois.dtype, device=rois.device)
    rois_with_batch_index = torch.cat([batch_indices, rois], dim=1)

    # Get the RoI-pooled features
    # We need to pass the correct feature map to RoIPooling. For FPN, it's usually '0' or 'pool'
    pooled_features = roi_pooling(feature_maps['0'], rois_with_batch_index)

    # Print the shape of the output
    # The shape should be (n_rois, channels, output_height, output_width)
    print(f'Pooled features shape: {pooled_features.shape}')

    # --- Visualization ---
    # Convert image tensor to uint8 for drawing bounding boxes
    image_uint8 = (image_tensor * 255).to(torch.uint8)

    # Extract bounding boxes from proposals (skip batch_index)
    # proposals are (batch_index, x1, y1, x2, y2)
    # We need (x1, y1, x2, y2)
    boxes_to_draw = rois

    num_rois = boxes_to_draw.shape[0]
    print(f'Number of RoIs visualized: {num_rois}')

    # Generate unique colors for each bounding box
    colors = [cm.viridis(i / num_rois) for i in range(num_rois)]
    # Convert RGBA to RGB for torchvision.utils.draw_bounding_boxes
    colors_rgb = [tuple(int(c * 255) for c in color[:3]) for color in colors]

    # Draw bounding boxes
    labels = [f"RoI {i+1}" for i in range(num_rois)]

    drawn_image = draw_bounding_boxes(
        image=image_uint8,
        boxes=boxes_to_draw,
        labels=labels,
        colors=colors_rgb,
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

