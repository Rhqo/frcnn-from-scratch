from dataclasses import dataclass, field
from typing import List

@dataclass
class TVFasterRCNNConfig:
    # Dataset Configs
    im_train_path: str = './data/VOC2007/JPEGImages'
    ann_train_path: str = './data/VOC2007/Annotations'
    im_test_path: str = './data/VOC2007-test/JPEGImages'
    ann_test_path: str = './data/VOC2007-test/Annotations'
    num_classes: int = 21

    # Torchvision Model Configs
    batch_size: int = 4
    num_workers: int = 4
    weight_decay: float = 5e-5
    momentum: float = 0.9
    min_size: int = 600
    max_size: int = 1000
    num_classes: int = 21
    anchor_sizes: List[List[int]] = field(default_factory=lambda: [[32, 64, 128, 256, 512]])
    anchor_aspect_ratios: List[List[float]] = field(default_factory=lambda: [[0.5, 1.0, 2.0]])
    rpn_pre_nms_top_n_train: int = 12000
    rpn_pre_nms_top_n_test: int = 6000
    rpn_post_nms_top_n_test: int = 300
    box_batch_size_per_image: int = 128
    task_name: str = 'checkpoint'
    seed: int = 42
    num_epochs: int = 10
    lr: float = 0.001
    ckpt_name: str = 'voc2007.pth'