from dataclasses import dataclass, field
from typing import List

@dataclass
class FasterRCNNConfig:
    # Dataset Configs
    im_train_path: str = './data/VOC2007/JPEGImages'
    ann_train_path: str = './data/VOC2007/Annotations'
    im_test_path: str = './data/VOC2007-test/JPEGImages'
    ann_test_path: str = './data/VOC2007-test/Annotations'
    num_classes: int = 21

    # Model Configs
    ## backbone
    im_channels: int = 3
    aspect_ratios: List[float] = field(default_factory=lambda: [0.5, 1, 2])
    scales: List[int] = field(default_factory=lambda: [128, 256, 512])
    min_im_size: int = 600
    max_im_size: int = 1000
    backbone_out_channels: int = 512
    fc_inner_dim: int = 1024
    ## rpn
    rpn_bg_threshold: float = 0.3
    rpn_fg_threshold: float = 0.7
    rpn_nms_threshold: float = 0.7
    rpn_train_prenms_topk: int = 12000
    rpn_test_prenms_topk: int = 6000
    rpn_train_topk: int = 2000
    rpn_test_topk: int = 300
    rpn_batch_size: int = 256
    rpn_pos_fraction: float = 0.5
    ## roi
    roi_iou_threshold: float = 0.5
    roi_low_bg_iou: float = 0.0
    roi_pool_size: int = 7
    roi_nms_threshold: float = 0.3
    roi_topk_detections: int = 100
    roi_score_threshold: float = 0.05
    roi_batch_size: int = 128
    roi_pos_fraction: float = 0.25

    # Training Configs
    task_name: str = 'checkpoint'
    seed: int = 42
    acc_steps: int = 1
    num_epochs: int = 10
    lr_steps: List[int] = field(default_factory=lambda: [12, 16])
    lr: float = 0.001
    ckpt_name: str = 'faster_rcnn_voc2007.pth'
