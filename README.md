# Faster R-CNN from Scratch

This project implements the Faster R-CNN object detection algorithm from scratch using PyTorch.

## Dataset Setup (Pascal VOC 2007)

To train and evaluate the model, you need to download the Pascal VOC 2007 dataset.

1.  **Download the dataset:**
    *   VOC2007 train/val: [http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
    *   VOC2007 test: [http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)

2.  **Extract the archives:**
    Extract both `VOCtrainval_06-Nov-2007.tar` and `VOCtest_06-Nov-2007.tar` into the `data/VOCdevkit/` directory. After extraction, you should have a directory structure like this:

    ```
    data/
    └── VOCdevkit/
        └── VOC2007/
            ├── Annotations/
            ├── ImageSets/
            ├── JPEGImages/
            └── ...
    ```

    Make sure the `VOC2007` directory is directly inside `data/VOCdevkit/`.

## Installation

This project uses `uv` for dependency management. To install the required packages, run:

```bash
uv sync
```

## Training

To train the Faster R-CNN model, run the `train.py` script:

```bash
uv run python train.py
```

## Test Scripts Summary

`scripts/` 디렉토리에는 Faster R-CNN 모델의 각 구성 요소를 테스트하고 이해하는 데 도움이 되는 스크립트가 포함되어 있습니다.

### `00_test_dataset.py`
*   **Input:** Pascal VOC 2007 Dataset (파일 시스템에서 로드).
*   **Output:**
    *   `image_tensor`: `torch.Tensor` 형태의 이미지 데이터.
    *   `target`: 이미지에 대한 Ground Truth Bounding Box 및 Label 정보를 담고 있는 Dictionary.
*   **의미:** `VOCDataset` 클래스가 데이터를 올바르게 로드하고 전처리하는지 확인합니다. 출력된 `image_tensor`와 `target`은 모델 학습에 사용될 데이터의 형식을 보여줍니다. Matplotlib을 사용하여 이미지와 해당 Bounding Box를 시각화합니다.

### `01_test_rpn.py`
*   **Input:** `VOCDataset`에서 로드된 실제 이미지 (Batch Dimension이 추가된 `torch.Tensor`).
*   **Output:**
    *   `rpn_cls_scores`: RPN의 Classification Score (`torch.Tensor`). 각 Anchor에 대한 Foreground/Background Score를 나타냅니다.
    *   `rpn_bbox_preds`: RPN의 Bounding Box Prediction (`torch.Tensor`). 각 Anchor에 대한 Bounding Box Regression Offset을 나타냅니다.
*   **의미:** Region Proposal Network (RPN)의 순방향 전달(Forward Pass)을 테스트합니다. RPN이 Feature Map에서 Objectness Score와 Bounding Box Regression 값을 올바르게 생성하는지 확인합니다. Matplotlib을 사용하여 RPN Objectness Heatmap을 시각화합니다.

### `02_test_proposal_layer.py`
*   **Input:**
    *   `rpn_cls_scores`: RPN Classification Score.
    *   `rpn_bbox_preds`: RPN Bounding Box Prediction.
    *   `img_size`: 원본 이미지의 크기 (Height, Width).
*   **Output:**
    *   `proposals`: `torch.Tensor` 형태의 Region Proposal (ROI). `(batch_index, x1, y1, x2, y2)` 형식의 Bounding Box 좌표를 포함합니다.
*   **의미:** `ProposalLayer`가 RPN의 출력을 기반으로 유효한 Region Proposal을 생성하는지 테스트합니다. 이 Proposal들은 다음 단계인 RoI Pooling의 입력으로 사용됩니다. Matplotlib을 사용하여 원본 이미지에 생성된 Region Proposal들을 시각화합니다.

### `03_test_roi_pooling.py`
*   **Input:**
    *   `feature_map`: Backbone Network에서 추출된 Feature Map.
    *   `rois`: `ProposalLayer`에서 생성된 Region Proposal (ROI).
*   **Output:**
    *   `pooled_features`: `torch.Tensor` 형태의 RoI-Pooled Feature. 각 RoI에 대해 고정된 크기(예: 7x7)로 추출된 Feature Map입니다.
*   **의미:** `RoIPooling` 레이어가 다양한 크기의 RoI에서 고정된 크기의 Feature를 올바르게 추출하는지 테스트합니다. 이는 Fast R-CNN Head의 입력으로 사용됩니다. Matplotlib을 사용하여 원본 이미지에 RoI Pooling에 사용된 RoI들을 시각화합니다.

### `04_test_fast_rcnn_head.py`
*   **Input:**
    *   `pooled_features`: `RoIPooling` 레이어에서 추출된 RoI-Pooled Feature.
*   **Output:**
    *   `cls_scores`: Fast R-CNN Head의 Classification Score (`torch.Tensor`). 각 RoI에 대한 최종 클래스 Score를 나타냅니다.
    *   `bbox_preds`: Fast R-CNN Head의 Bounding Box Prediction (`torch.Tensor`). 각 RoI에 대한 최종 Bounding Box Regression Offset을 나타냅니다.
*   **의미:** Fast R-CNN Head가 RoI-Pooled Feature를 기반으로 최종 객체 분류 Score와 Bounding Box Regression 값을 올바르게 예측하는지 테스트합니다. 시각화는 몇몇 RoI에 대한 Top Predicted Class와 Score를 텍스트로 출력합니다.

### `05_test_faster_rcnn_model.py`
*   **Input:** `VOCDataset`에서 로드된 실제 이미지.
*   **Output:**
    *   `detections`: 모델이 예측한 최종 객체 Detection 목록. 각 Detection은 Bounding Box, Class Label, Confidence Score를 포함합니다.
*   **의미:** Faster R-CNN 모델 전체의 순방향 전달(Forward Pass)을 테스트합니다. 모델이 입력 이미지에 대해 최종 객체 Detection을 올바르게 생성하는지 확인합니다. Matplotlib을 사용하여 원본 이미지에 예측된 Bounding Box와 Label, Score를 시각화합니다.

### `06_test_losses.py`
*   **Input:** Dummy Prediction 및 Ground Truth Tensor.
*   **Output:**
    *   `loss_l1`: Smooth L1 Loss 값.
    *   `loss_ce`: Cross-Entropy Loss 값.
*   **의미:** 모델 학습에 사용되는 `smooth_l1_loss`와 `cross_entropy_loss` 함수가 올바르게 작동하는지 테스트합니다. 출력은 각 Loss 함수의 계산 결과를 보여줍니다.

### `07_compare_models.py`
*   **Input:** `VOCDataset`에서 로드된 실제 이미지, 학습된 Faster R-CNN 모델 (`models/faster_rcnn_epoch_1.pth`), Pre-trained Torchvision Faster R-CNN 모델.
*   **Output:**
    *   두 모델의 예측 결과 (Bounding Box, Label, Score).
*   **의미:** 사용자가 학습시킨 모델과 Torchvision에서 제공하는 Pre-trained 모델의 성능을 시각적으로 비교합니다. Matplotlib을 사용하여 두 모델의 예측 결과를 각각의 이미지에 시각화하여 보여줍니다.