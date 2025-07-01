
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_resnet50_base_net():
    """
    Returns the base ResNet50 network (feature extractor) from a pre-trained model.

    The base network consists of all convolutional layers from ResNet50, up to the
    average pooling layer, effectively serving as a feature extractor.
    The fully connected layer (classifier) of the original ResNet50 is removed.

    Returns:
        torch.nn.Sequential: The ResNet50 base network.
    """
    # Load a pre-trained ResNet50 model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # The feature extractor is everything up to the final classification layer
    # For ResNet, this typically means all layers before `fc`
    # We want to remove the average pooling and the fully connected layer
    # The feature map from ResNet50's last convolutional block (layer4) has 2048 channels.
    base_net = nn.Sequential(*list(model.children())[:-2])

    # Freeze the parameters of the base network
    for param in base_net.parameters():
        param.requires_grad = False

    return base_net


if __name__ == "__main__":
    print("--- Testing ResNet50 Base Network ---")
    resnet50_base = get_resnet50_base_net()
    print("\nSuccessfully loaded ResNet50 base network.")
    # print(resnet50_base)

    # Test with a dummy image
    dummy_image = torch.randn(1, 3, 600, 800)  # Example input size
    feature_map = resnet50_base(dummy_image)
    print(f"Output feature map shape: {feature_map.shape}")

    # Expected output channels for ResNet50's last convolutional block is 2048
    assert feature_map.shape[1] == 2048, \
        f"Expected 2048 channels, but got {feature_map.shape[1]}"
    print("ResNet50 base network test passed.")

