

import torch
import sys
import os
import numpy as np

# Add the project root to the Python path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from frcnn.models.roi_pooling import RoIPooling


def main():
    """
    Tests the RoIPooling layer.
    """
    print("--- Testing RoI Pooling Layer ---")

    # Create a dummy feature map
    feature_map = torch.randn(1, 512, 37, 50)
    print(f"Input feature map shape: {feature_map.shape}")

    # Create some dummy proposals (batch_index, x1, y1, x2, y2)
    rois = torch.tensor([
        [0, 100, 100, 200, 200],
        [0, 150, 150, 300, 300],
        [0, 50, 50, 100, 100]
    ], dtype=torch.float32)
    print(f"\nInput proposals shape: {rois.shape}")

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
    print("--- Test Complete ---")


if __name__ == '__main__':
    main()

