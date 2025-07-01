import torch
import sys
import os

# Add the project root to the Python path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from frcnn.models.faster_rcnn import FasterRCNN
from frcnn.dataset import VOCDataset, VOC_CLASSES



def visualize_predictions(image, predictions, model_name, score_threshold=0.7):
    # Convert image to PIL for drawing
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)

    # Convert PIL image to tensor for drawing bounding boxes
    img_tensor = transforms.ToTensor()(image)

    all_boxes = []
    all_labels = []
    all_scores = []

    if predictions:
        # Our model returns a list containing a single dictionary in eval mode
        p = predictions[0]
        if len(p['boxes']) > 0:
            if len(p['boxes']) > 0:
                # Filter by score threshold
                keep = p['scores'] > score_threshold
            all_boxes.append(p['boxes'][keep])
            # Use VOC_CLASSES for labels
            all_labels.extend([f"{VOC_CLASSES[l]} ({s:.2f})" for l, s in zip(p['labels'][keep], p['scores'][keep])])
            all_scores.append(p['scores'][keep])

    if len(all_boxes) > 0:
        boxes = torch.cat(all_boxes)
        scores = torch.cat(all_scores)
        labels = all_labels

        drawn_image = draw_bounding_boxes(img_tensor, boxes, labels, colors="red", width=2)
    else:
        drawn_image = img_tensor

    plt.figure(figsize=(10, 10))
    plt.imshow(drawn_image.permute(1, 2, 0).cpu().numpy())
    plt.title(f'Predictions from {model_name}')
    plt.axis('off')
    plt.show()


def main():
    """
    Tests the full FasterRCNN model forward pass.
    """
    print("--- Testing Full Faster R-CNN Model ---")

    # Path to the Pascal VOC dataset root (VOCdevkit)
    dataset_root = os.path.join(module_path, 'data', 'VOCdevkit')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    voc_dataset = VOCDataset(root_dir=dataset_root, split='trainval', transform=transform)

    # Fetch a sample from the dataset
    image_tensor, target = voc_dataset[5]  # Get the first sample
    original_image = transforms.ToPILImage()(image_tensor) # For visualization
    test_image = image_tensor.unsqueeze(0) # Add batch dimension
    img_size = image_tensor.shape[1:] # Get H, W from the image tensor
    print(f"Input image shape: {test_image.shape}")

    # Instantiate the pre-trained FasterRCNN model
    model = FasterRCNN(num_classes=len(VOC_CLASSES))
    model.eval() # Set model to evaluation mode
    print("\nSuccessfully loaded pre-trained Faster R-CNN ResNet50 FPN model.")

    # Perform a forward pass
    # The model now returns a list of dictionaries when in eval mode
    detections = model(test_image, img_size)

    # Visualize the detections
    visualize_predictions(original_image, detections, "Pre-trained Faster R-CNN Model Detections")

    print("--- Test Complete ---")


if __name__ == '__main__':
    main()
