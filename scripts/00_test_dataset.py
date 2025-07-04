import os
import sys
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
import numpy as np

# Add the project root to the Python path
# This allows us to import modules from the 'frcnn' directory
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from frcnn.dataset import VOCDataset, VOC_CLASSES


def main():
    """
    Tests the VOCDataset class.
    """
    print("--- Testing VOCDataset ---")

    # Path to the Pascal VOC dataset root (VOCdevkit)
    # IMPORTANT: You must download and extract the dataset first.
    # See instructions in /data/README.md
    dataset_root = os.path.join(module_path, 'data', 'VOCdevkit')

    # Define a simple transformation to convert images to tensors
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Instantiate the dataset
    try:
        voc_dataset = VOCDataset(root_dir=dataset_root, split='trainval', transform=transform)
        print(f"Successfully loaded dataset. Total samples: {len(voc_dataset)}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please make sure you have downloaded and extracted the Pascal VOC 2007 dataset in the 'data' directory.")
        return

    # Fetch a sample from the dataset
    image_tensor, target = voc_dataset[5]  # Get the 6th sample

    print("\n--- Sample Details ---")
    print(f"Image tensor shape: {image_tensor.shape}")
    print(f"Image tensor type: {image_tensor.dtype}")
    print("\nTarget dictionary:")
    for key, value in target.items():
        print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")

    # Check the number of boxes and labels
    num_boxes = target['boxes'].shape[0]
    print(f"\nFound {num_boxes} objects in this sample.")

    # Print the actual annotation data
    print("\n--- Annotation Data ---")
    boxes = target['boxes'].numpy()
    labels = target['labels'].numpy()

    for i in range(num_boxes):
        box = boxes[i]
        class_name = VOC_CLASSES[labels[i]]
        print(f"  - Object {i+1}: Class='{class_name}', BBox=[{box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}, {box[3]:.2f}]")

    # --- Visualization ---
    # Convert image tensor to uint8 for drawing bounding boxes
    image_uint8 = (image_tensor * 255).to(torch.uint8)

    # Prepare labels for visualization
    display_labels = [VOC_CLASSES[label] for label in target['labels'].tolist()]

    # Draw bounding boxes
    drawn_image = draw_bounding_boxes(
        image=image_uint8,
        boxes=target['boxes'],
        labels=display_labels,
        colors="red",
        width=2
    )

    # Convert to numpy array for matplotlib and permute dimensions (C, H, W) -> (H, W, C)
    drawn_image_np = drawn_image.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.imshow(drawn_image_np)
    plt.title('Sample Image with Bounding Boxes')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()

