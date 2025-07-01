

import torch
import sys
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.image_list import ImageList


# Add the project root to the Python path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from frcnn.models.fast_rcnn_head import FastRCNNHead

from frcnn.models.rpn import RPN
from frcnn.models.proposal_layer import ProposalLayer
from frcnn.models.roi_pooling import RoIPooling
from frcnn.dataset import VOCDataset, VOC_CLASSES
from frcnn.utils.bbox_tools import loc2bbox # Import loc2bbox





def main():
    """
    Tests the FastRCNNHead.
    """
    print("--- Testing Fast R-CNN Head ---")

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
    print("\nSuccessfully loaded pre-trained Faster R-CNN ResNet50 FPN model.")

    # Use backbone, RPN, and RoI pooling from pre-trained model
    base_net = pretrained_model.backbone
    rpn = pretrained_model.rpn
    box_roi_pool = pretrained_model.roi_heads.box_roi_pool

    # Create ImageList for torchvision RPN
    image_list = ImageList(dummy_image, [(img_size[0], img_size[1])])

    # Get the feature map from the base network
    feature_maps = base_net(dummy_image) # This will be an OrderedDict
    print(f"Feature map keys: {feature_maps.keys()}")

    # Pass the feature map through the RPN to get proposals
    proposals, _ = rpn(image_list, feature_maps)
    proposals = proposals[0] # Take the first element from the list (batch_size=1)
    print(f"\nGenerated ROIs shape (from pre-trained RPN): {proposals.shape}")

    # Get the RoI-pooled features using torchvision's RoI pooler
    # box_roi_pool expects a list of feature maps and a list of proposals
    # proposals need to be a list of tensors, even for batch_size=1
    pooled_features = box_roi_pool(feature_maps, [proposals], [img_size])
    print(f"Input RoI-pooled features shape: {pooled_features.shape}")

    # Define parameters for the head
    in_channels = 256  # From ResNet50 FPN feature map
    num_classes = 21   # 20 Pascal VOC classes + 1 background class

    # Instantiate the FastRCNNHead
    fast_rcnn_head = FastRCNNHead(in_channels, num_classes, (7, 7)) # Use (7,7) as output size
    print("\nSuccessfully loaded Fast R-CNN Head.")

    # Pass the pooled features through the head
    cls_scores, bbox_preds = fast_rcnn_head(pooled_features)

    # Print the shapes of the output tensors
    print(f"\nClassification scores shape: {cls_scores.shape}")
    print(f"Bounding box predictions shape: {bbox_preds.shape}")

    # --- Visualization ---
    # Apply softmax to RoI classification scores
    roi_probs = torch.softmax(cls_scores, dim=1)

    # Get the predicted class labels and scores
    scores, labels = roi_probs.max(dim=1)

    # Filter out background class (label 0)
    foreground_indices = (labels != 0).nonzero(as_tuple=True)[0]

    if foreground_indices.numel() == 0:
        print("\nNo foreground objects detected by Fast R-CNN Head.")
        # Display original image if no detections
        plt.figure(figsize=(10, 10))
        plt.imshow(original_image)
        plt.title('Original Image (No Detections)')
        plt.axis('off')
        plt.show()
    else:
        scores = scores[foreground_indices]
        labels = labels[foreground_indices]
        bbox_preds = bbox_preds[foreground_indices]
        rois = proposals[foreground_indices] # Use proposals from pre-trained RPN

        # Convert rois from (x1, y1, x2, y2) to (x, y, w, h) for loc2bbox
        rois_xywh = torch.stack([
            rois[:, 0],
            rois[:, 1],
            rois[:, 2] - rois[:, 0],
            rois[:, 3] - rois[:, 1]
        ], dim=1)

        # Select the bbox_preds corresponding to the predicted labels
        selected_bbox_preds = torch.empty((foreground_indices.numel(), 4), dtype=torch.float32, device=bbox_preds.device)
        for i, label_idx in enumerate(labels):
            selected_bbox_preds[i] = bbox_preds[i, label_idx * 4 : (label_idx + 1) * 4]

        # Decode bounding box predictions
        decoded_boxes = loc2bbox(rois_xywh, selected_bbox_preds)

        # Clip boxes to image boundaries
        img_h, img_w = img_size
        decoded_boxes[:, 0::2] = decoded_boxes[:, 0::2].clamp(min=0, max=img_w)
        decoded_boxes[:, 1::2] = decoded_boxes[:, 1::2].clamp(min=0, max=img_h)

        # Convert decoded_boxes from (y1, x1, y2, x2) to (x1, y1, x2, y2) for draw_bounding_boxes
        final_boxes = torch.stack([
            decoded_boxes[:, 1],  # x1
            decoded_boxes[:, 0],  # y1
            decoded_boxes[:, 3],  # x2
            decoded_boxes[:, 2]   # y2
        ], dim=1)

        # Debugging: Check for invalid box dimensions
        if (final_boxes[:, 0] >= final_boxes[:, 2]).any() or \
           (final_boxes[:, 1] >= final_boxes[:, 3]).any():
            print("WARNING: Some bounding boxes have xmin >= xmax or ymin >= ymax after conversion!")
            # Filter out invalid boxes for visualization
            valid_boxes_mask = (final_boxes[:, 0] < final_boxes[:, 2]) & \
                               (final_boxes[:, 1] < final_boxes[:, 3])
            final_boxes = final_boxes[valid_boxes_mask]
            labels = labels[valid_boxes_mask] # Filter labels too
            scores = scores[valid_boxes_mask] # Filter scores too
            num_detections = final_boxes.shape[0]
            print(f"Filtered to {num_detections} valid detections.")

        # Convert image tensor to uint8 for drawing bounding boxes
        image_uint8 = (image_tensor * 255).to(torch.uint8)

        num_detections = final_boxes.shape[0]
        print(f"\nNumber of detections visualized: {num_detections}")

        # Generate unique colors for each bounding box
        colors = [cm.viridis(i / num_detections) for i in range(num_detections)]
        colors_rgb = [tuple(int(c * 255) for c in color[:3]) for color in colors]

        # Prepare labels for visualization (using COCO_CLASSES)
        display_labels = [f"{VOC_CLASSES[labels[i] - 1]} ({scores[i]:.2f})" for i in range(num_detections)]

        drawn_image = draw_bounding_boxes(
            image=image_uint8,
            boxes=final_boxes,
            labels=display_labels,
            colors=colors_rgb,
            width=2
        )

        # Convert to numpy array for matplotlib and permute dimensions (C, H, W) -> (H, W, C)
        drawn_image_np = drawn_image.permute(1, 2, 0).cpu().numpy()

        plt.figure(figsize=(10, 10))
        plt.imshow(drawn_image_np)
        plt.title('Fast R-CNN Head Detections (Pre-trained Components)')
        plt.axis('off')
        plt.show()

    print("--- Test Complete ---")


if __name__ == '__main__':
    main()

