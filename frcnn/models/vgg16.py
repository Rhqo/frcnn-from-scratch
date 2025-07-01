import torch
import torch.nn as nn
from torchvision.models import vgg16


def get_vgg16_base_net():
    """
    Returns the base VGG16 network (feature extractor) from a pre-trained model.

    The base network consists of all convolutional layers from VGG16, up to the
    last max-pooling layer (`conv5_3`). This is used to extract a feature map
    from the input image.

    The final fully connected layers and the final max-pooling layer of the
    original VGG16 are removed.

    Returns:
        torch.nn.Sequential: The VGG16 base network.
    """
    # Load a pre-trained VGG16 model
    model = vgg16(pretrained=True)

    # The feature extractor is the `features` part of the VGG16 model
    features = model.features

    # The original VGG16 `features` module has 31 layers.
    # The paper uses the output of the conv5_3 layer, which is at index 29.
    # We therefore take all layers up to index 29 (inclusive, so we slice up to 30).
    # This removes the final max-pooling layer, which has a different stride
    # and would change the feature map size in an undesirable way for the RPN.
    base_net = nn.Sequential(*list(features[:30].children()))

    # Freeze the first 4 convolutional blocks (10 layers) of the VGG16 network.
    # This is a common practice to prevent overfitting, as the earlier layers
    # detect more general features (edges, corners, etc.).
    for layer in base_net[:10]:
        for param in layer.parameters():
            param.requires_grad = False

    return base_net


if __name__ == '__main__':
    # This is a simple test to verify the output of the base network.
    # It creates a random image tensor, passes it through the network,
    # and prints the shape of the resulting feature map.

    print("--- Testing VGG16 Base Network ---")
    
    # Create a dummy input image (batch size 1, 3 channels, 600x800 pixels)
    # Note: A real image would be pre-processed (e.g., resized, normalized).
    dummy_image = torch.randn(1, 3, 600, 800)
    print(f"Input image shape: {dummy_image.shape}")

    # Get the base network
    vgg16_base = get_vgg16_base_net()
    print("\nSuccessfully loaded VGG16 base network.")
    # print(vgg16_base)

    # Pass the dummy image through the network
    feature_map = vgg16_base(dummy_image)

    # The expected feature map size is approximately 1/16th of the input image size.
    # For a 600x800 image, this would be around 37x50.
    print(f"\nOutput feature map shape: {feature_map.shape}")
    print("--- Test Complete ---")
