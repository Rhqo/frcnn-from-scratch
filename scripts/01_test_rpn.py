import torch
import torch.nn as nn
import numpy as np
import sys
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Add the project root to the Python path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from frcnn.models.resnet50 import get_resnet50_base_net
from frcnn.dataset import VOCDataset


class RPN(nn.Module):
    """
    Region Proposal Network.

    This network takes a feature map from a backbone network (e.g., VGG16) and
    outputs a set of rectangular object proposals, each with an objectness score.

    Args:
        in_channels (int): Number of channels in the input feature map.
        mid_channels (int): Number of channels in the intermediate convolutional layer.
        n_anchor (int): Number of anchors to be generated for each feature map pixel.
    """

    def __init__(self, in_channels=512, mid_channels=512, n_anchor=9):
        super(RPN, self).__init__()

        # A 3x3 convolutional layer to process the input feature map.
        # Padding is 1 to preserve the spatial dimensions.
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)

        # A 1x1 convolutional layer for classification.
        # It outputs 2 scores for each of the `n_anchor` anchors: (background, foreground).
        self.cls_layer = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)

        # A 1x1 convolutional layer for bounding box regression.
        # It outputs 4 regression parameters (dx, dy, dw, dh) for each of the `n_anchor` anchors.
        self.reg_layer = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

        # Initialize the weights of the convolutional layers
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for the RPN layers."""
        # Normal initialization for the first convolutional layer
        nn.init.normal_(self.conv1.weight, std=0.01)
        nn.init.constant_(self.conv1.bias, 0)
        # Normal initialization for the classification layer
        nn.init.normal_(self.cls_layer.weight, std=0.01)
        nn.init.constant_(self.cls_layer.bias, 0)
        # Normal initialization for the regression layer
        nn.init.normal_(self.reg_layer.weight, std=0.01)
        nn.init.constant_(self.reg_layer.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the RPN.

        Args:
            x (torch.Tensor): The feature map from the backbone network.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - rpn_cls_scores (torch.Tensor): Classification scores for each anchor.
                - rpn_bbox_preds (torch.Tensor): Bounding box regression predictions for each anchor.
        """
        # Pass the feature map through the first convolutional layer and a ReLU activation
        x = nn.functional.relu(self.conv1(x))

        # Get the classification scores and bounding box predictions
        rpn_cls_scores = self.cls_layer(x)
        rpn_bbox_preds = self.reg_layer(x)

        return rpn_cls_scores, rpn_bbox_preds


if __name__ == '__main__':
    print("--- Testing RPN ---")

    # Path to the Pascal VOC dataset root (VOCdevkit)
    dataset_root = os.path.join(module_path, 'data', 'VOCdevkit')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    voc_dataset = VOCDataset(root_dir=dataset_root, split='trainval', transform=transform)

    # Fetch a sample from the dataset
    image_tensor, target = voc_dataset[5]  # Get the first sample
    dummy_image = image_tensor.unsqueeze(0) # Add batch dimension
    print(f"Input image shape: {dummy_image.shape}")

    # Get the VGG16 base network
    base_net = get_resnet50_base_net()

    # Get the feature map from the base network
    feature_map = base_net(dummy_image)
    print(f"Feature map shape: {feature_map.shape}")

    # Instantiate the RPN
    rpn = RPN(in_channels=2048, mid_channels=512, n_anchor=9)
    print("\nSuccessfully loaded RPN.")

    # Pass the feature map through the RPN
    rpn_cls_scores, rpn_bbox_preds = rpn(feature_map)

    # Print the shapes of the output tensors
    print(f"\nRPN classification scores shape: {rpn_cls_scores.shape}")
    print(f"RPN bounding box predictions shape: {rpn_bbox_preds.shape}")

    # --- Visualization ---
    # Reshape RPN classification scores to (batch_size, height, width, n_anchor * 2)
    # Then take the foreground scores ([:, 1::2]) and reshape to (batch_size, height, width, n_anchor)
    # For simplicity, we'll average across anchors or take the max for visualization
    
    # Assuming batch_size = 1
    rpn_cls_scores_reshaped = rpn_cls_scores.permute(0, 2, 3, 1).contiguous()
    # Get foreground scores (class 1)
    foreground_scores = rpn_cls_scores_reshaped[0, :, :, 1::2] # Shape: (H, W, n_anchor)
    
    # Take the maximum score across anchors for each spatial location
    # This gives a single heatmap for objectness
    objectness_heatmap = foreground_scores.max(dim=-1)[0] # Shape: (H, W)

    # Normalize heatmap for display
    objectness_heatmap_np = objectness_heatmap.detach().cpu().numpy()
    objectness_heatmap_np = (objectness_heatmap_np - objectness_heatmap_np.min()) / \
                            (objectness_heatmap_np.max() - objectness_heatmap_np.min() + 1e-6)

    # Convert image tensor to numpy for display
    image_np = dummy_image.squeeze(0).permute(1, 2, 0).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Display heatmap
    im = axes[1].imshow(objectness_heatmap_np, cmap='viridis', alpha=0.8)
    axes[1].set_title('RPN Objectness Heatmap')
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    print("--- Test Complete ---")