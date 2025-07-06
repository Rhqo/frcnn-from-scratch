# Faster-RCNN from scratch

## Download Pascal VOC dataset

```bash
mkdir -p ./data

curl -L -o VOCtrainval_06-Nov-2007.tar http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar -xf VOCtrainval_06-Nov-2007.tar
mv VOCdevkit/VOC2007 ./data/VOC2007

curl -L -o VOCtest_06-Nov-2007.tar http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar
mv VOCdevkit/VOC2007 ./data/VOC2007-test

rm -rf VOCdevkit
rm VOCtrainval_06-Nov-2007.tar
rm VOCtest_06-Nov-2007.tar
```

## `train.py`

Faster R-CNN 모델 training code.

**Usage:**

```bash
uv run train.py
```

## `inference.py`

Faster R-CNN 모델 inference code. \
학습한 custom model과 torchvision에서 불러온 모델 사용 가능

**Usage:** 

`--infer`: custom model로 sample image에 inference 실행
```bash
uv run inference.py --infer
```

`--eval`: custom model로 test set에 mAP 평가
```bash
uv run inference.py --eval 
```

`--torchvision`: torchvision model로 sample image에 inference 실행
```bash
uv run inference.py --torchvision 
```

`--image_path`: custom model로 단일 이미지 inference
```bash
uv run inference.py --image_path ./image.jpg 
```

`--image_path_torchvision`: torchvision model로 단일 이미지 inference
```bash
uv run inference.py --image_path_torchvision ./image.jpg 
```
