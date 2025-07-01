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

