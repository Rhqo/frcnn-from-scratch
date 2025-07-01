import torch
import sys
import os
from PIL import Image
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the Python path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)


from frcnn.models.faster_rcnn import FasterRCNN

def load_image(image_path):
    img = read_image(image_path)
    return img

def visualize_predictions(image, predictions, model_name, score_threshold=0.7, class_names=None):
    # Convert image to PIL for drawing
    if isinstance(image, torch.Tensor):
        image = T.ToPILImage()(image)

    # Convert PIL image to tensor for drawing bounding boxes
    img_tensor = T.ToTensor()(image)

    all_boxes = []
    all_labels = []
    all_scores = []

    if predictions and len(predictions) > 0:
        p = predictions[0]
        if p['boxes'].numel() > 0:
            # Filter by score threshold
            keep = p['scores'] > score_threshold
            all_boxes.append(p['boxes'][keep])
            
            if class_names:
                # Use provided class_names for labels
                all_labels.extend([f"{class_names[l]} ({s:.2f})" for l, s in zip(p['labels'][keep], p['scores'][keep])])
            else:
                # Fallback to default if class_names not provided
                all_labels.extend([f"Class: {l}, Score: {s:.2f}" for l, s in zip(p['labels'][keep], p['scores'][keep])])
            all_scores.append(p['scores'][keep])

    if len(all_boxes) > 0:
        boxes = torch.cat(all_boxes)
        scores = torch.cat(all_scores)
        labels = all_labels

        drawn_image = draw_bounding_boxes(img_tensor, boxes, labels, colors="red", width=2)
    else:
        drawn_image = img_tensor

    return drawn_image.permute(1, 2, 0).cpu().numpy(), model_name

def main():
    print("--- Comparing Trained Model with Pre-trained Model ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load your trained model ---
    num_classes = 21 # VOC classes + background
    trained_model = FasterRCNN(num_classes=num_classes).to(device)
    trained_model_path = os.path.join(module_path, 'models', 'faster_rcnn_epoch_1.pth')

    if os.path.exists(trained_model_path):
        trained_model.load_state_dict(torch.load(trained_model_path, map_location=device))
        trained_model.eval()
        print(f"Successfully loaded trained model from {trained_model_path}")
    else:
        print(f"Trained model not found at {trained_model_path}. Please ensure it exists.")
        return

    # --- 2. Load a pre-trained model from torchvision ---
    # Use the default weights for the pre-trained model
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    pretrained_model = fasterrcnn_resnet50_fpn(weights=weights).to(device)
    pretrained_model.eval()
    print("Successfully loaded pre-trained Faster R-CNN ResNet50 FPN model.")

    # --- 3. Prepare a sample image ---
    # Try to find a sample image in VOCdevkit
    sample_image_path = os.path.join(module_path, 'data', 'VOCdevkit', 'VOC2007', 'JPEGImages', '000005.jpg') # Example image
    if not os.path.exists(sample_image_path):
        print(f"Sample image not found at {sample_image_path}. Using a dummy image.")
        # Create a dummy image if a real one isn't found
        dummy_image = torch.randint(0, 256, (3, 600, 800), dtype=torch.uint8)
        img_tensor = T.ToDtype(torch.float32, scale=True)(dummy_image)
        original_image = T.ToPILImage()(dummy_image)
    else:
        print(f"Using sample image: {sample_image_path}")
        original_image = Image.open(sample_image_path).convert("RGB")
        img_tensor = T.ToTensor()(original_image)

    # Add batch dimension
    input_tensor = img_tensor.unsqueeze(0).to(device)

    # --- 4. Run inference and get images for plotting ---
    print("\nRunning inference with trained model...")
    with torch.no_grad():
        img_size = input_tensor.shape[2:] # Get H, W from the input tensor
        trained_predictions = trained_model(input_tensor, img_size)
    trained_image_np, trained_title = visualize_predictions(original_image, trained_predictions, "Your Trained Model")

    print("\nRunning inference with pre-trained model...")
    with torch.no_grad():
        pretrained_predictions = pretrained_model(input_tensor)
    pretrained_image_np, pretrained_title = visualize_predictions(original_image, pretrained_predictions, "Torchvision Pre-trained Model")

    # --- 5. Plot both images in one figure ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    axes[0].imshow(trained_image_np)
    axes[0].set_title(trained_title)
    axes[0].axis('off')

    axes[1].imshow(pretrained_image_np)
    axes[1].set_title(pretrained_title)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    print("\n--- Comparison Complete ---")

if __name__ == '__main__':
    main()