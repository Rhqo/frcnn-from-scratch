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

## 프로젝트 구조

`frcnn/` 디렉토리는 Faster R-CNN 구현의 핵심 구성 요소를 포함합니다:

### `frcnn/dataset.py`
*   Pascal VOC 2007 dataset 로딩 및 전처리를 담당하며, VOC_CLASSES 정의 및 XML annotation 파싱을 포함합니다.

### `frcnn/losses.py`
*   Faster R-CNN model 학습에 필수적인 bounding box regression을 위한 smooth_l1_loss와 classification을 위한 cross_entropy_loss 구현을 포함합니다.

### `frcnn/trainer.py`
*   Faster R-CNN model의 학습 프로세스를 관리하며, RPN 및 Fast R-CNN head 모두에 대한 forward pass, loss 계산, target assignment를 처리합니다.

### `frcnn/models/`
이 디렉토리는 Faster R-CNN model의 핵심 neural network 구성 요소를 포함합니다:
*   **`fast_rcnn_head.py`**: RoI-pooled features를 받아 최종 object classification 및 bounding box regression을 수행하는 Fast R-CNN head를 구현합니다.
*   **`faster_rcnn.py`**: backbone network, RPN, RoI Pooling, Fast R-CNN head를 통합하는 완전한 Faster R-CNN model을 정의합니다.
*   **`proposal_layer.py`**: RPN output으로부터 region proposal을 생성하고 필터링하는 Proposal Layer를 구현합니다.
*   **`resnet50.py`**: Faster R-CNN model의 feature extractor (backbone)로 사용되는 ResNet50 base network를 제공합니다.
*   **`roi_pooling.py`**: 가변 크기의 region of interest로부터 고정 크기의 feature map을 추출하는 RoI Pooling layer를 구현합니다.
*   **`rpn.py`**: candidate object region을 제안하고 objectness score를 할당하는 Region Proposal Network (RPN)를 구현합니다.

### `frcnn/utils/`
이 디렉토리는 Faster R-CNN 구현을 지원하는 utility function을 포함합니다:
*   **`anchors.py`**: 다양한 scale 및 aspect ratio를 가진 anchor box 생성을 위한 function을 제공합니다.
*   **`bbox_tools.py`**: bounding box regression offset 인코딩/디코딩 및 Intersection over Union (IoU) 계산을 포함한 bounding box 조작을 위한 tool을 포함합니다.

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

> [!NOTE]
> **참고: `02_test_proposal_layer.py`와 `03_test_roi_pooling.py`에서 제안(proposal) 차이**
> 
> `ProposalLayer`와 `RoIPooling`은 Faster R-CNN 파이프라인에서 서로 다른 역할을 수행합니다.
> 
> *   **`ProposalLayer` (scripts/02_test_proposal_layer.py):** 이 레이어는 Region Proposal Network (RPN)의 출력으로부터 **영역 제안(Region Proposals, ROIs)을 생성하고 필터링**하는 역할을 합니다. RPN의 원시 출력에 NMS(Non-Maximum Suppression)를 적용하고 점수 기반 필터링을 통해 고정된 수의 고품질 제안(예: 학습 시 2000개, 테스트 시 300개)을 선택합니다. 이 레이어의 출력 자체가 제안들의 집합입니다.
> 
> *   **`RoIPooling` (scripts/03_test_roi_pooling.py):** 이 레이어는 **제안의 개수를 줄이지 않습니다.** 대신, 이미 생성된 제안(ROIs)을 입력으로 받아, 각 제안에 대해 백본 네트워크의 특징 맵(feature map)에서 고정된 크기의 특징 맵을 추출합니다. `RoIPooling`의 목적은 원래 크기나 종횡비에 관계없이 모든 제안이 일관된 차원(예: 7x7)의 특징 맵을 생성하여 Fast R-CNN 헤드로 전달될 수 있도록 하는 것입니다.
> 
> 요약하자면:
> *   `ProposalLayer`는 *몇 개의* 제안이 다음 단계로 전달될지 결정합니다.
> *   `RoIPooling`은 *각 제안*을 균일한 특징 표현으로 처리하지만, 제안의 *개수*는 변경하지 않습니다.
> 
> 따라서 두 스크립트에서 보이는 제안의 개수는 `ProposalLayer`의 필터링 결과이며, `RoIPooling`은 단순히 이 고정된 제안 집합에 대해 작동하는 것입니다.


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
*   **의미:** 사용자가 학습시킨 모델과 Torchvision에서 제공하는 Pre-trained 모델의 성능을 시각적으로 비교합니다. Matplotlib을 사용하여 두 모델의 예측 결과를 **하나의 플롯에 나란히 시각화**하여 보여줍니다. 예측된 **클래스 ID 대신 클래스 이름**을 표시합니다.
