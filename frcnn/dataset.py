
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

# Note: this is a simplified list. Real-world applications might need a more robust mapping.
VOC_CLASSES: Tuple[str, ...] = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)


class VOCDataset(Dataset):
    """
    PyTorch Dataset for the Pascal VOC 2007 dataset.

    Args:
        root_dir (str): Path to the VOCdevkit directory.
        split (str): The dataset split to use (e.g., 'train', 'val', 'trainval', 'test').
        transform (callable, optional): A function/transform to apply to the images.
    """

    def __init__(self, root_dir: str, split: str = "trainval", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.class_to_idx = {cls: i for i, cls in enumerate(VOC_CLASSES)}
        self.idx_to_class = {i: cls for i, cls in enumerate(VOC_CLASSES)}

        self.image_dir = os.path.join(root_dir, "VOC2007", "JPEGImages")
        self.annotation_dir = os.path.join(root_dir, "VOC2007", "Annotations")
        self.image_set_dir = os.path.join(root_dir, "VOC2007", "ImageSets", "Main")

        if not os.path.isdir(self.image_set_dir):
            raise FileNotFoundError(
                f"ImageSets directory not found at {self.image_set_dir}. "
                "Please ensure the VOC dataset is correctly extracted."
            )

        split_file = os.path.join(self.image_set_dir, f"{self.split}.txt")
        with open(split_file) as f:
            self.image_ids = [line.strip() for line in f.readlines()]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_id = self.image_ids[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")

        # Load annotations
        annotation_path = os.path.join(self.annotation_dir, f"{image_id}.xml")
        boxes, labels = self._parse_annotation(annotation_path)

        target: Dict[str, torch.Tensor] = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }

        if self.transform:
            image = self.transform(image)

        return image, target

    def _parse_annotation(self, path: str) -> Tuple[List[List[float]], List[int]]:
        """Parses a VOC XML annotation file."""
        tree = ET.parse(path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall("object"):
            # Skip difficult or truncated objects
            if obj.find("difficult") and obj.find("difficult").text == "1":
                continue
            if obj.find("truncated") and obj.find("truncated").text == "1":
                continue

            label_name = obj.find("name").text.lower().strip()
            if label_name in self.class_to_idx:
                labels.append(self.class_to_idx[label_name])
                
                bbox = obj.find("bndbox")
                # VOC format is (xmin, ymin, xmax, ymax)
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
                boxes.append([xmin, ymin, xmax, ymax])

        return boxes, labels

    def collate_fn(self, batch):
        """
        Custom collate_fn for the DataLoader.
        
        Since each image can have a different number of objects, we need a way
        to batch them together. This function pads the targets to the max
        number of objects in a batch.
        """
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        
        # Images are typically transformed into tensors of the same size.
        # If not, they would need padding as well.
        images = torch.stack(images, 0)
        
        return images, targets
