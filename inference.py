import torch
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
import random
import os
from tqdm import tqdm
import torchvision

from torch.utils.data.dataloader import DataLoader

from config.voc_config import FasterRCNNConfig
from model.utils import get_iou
from model.faster_rcnn import FasterRCNN
from dataset.voc_loader import VOCDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image normalization values (from FasterRCNN model)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

def get_iou(det, gt):
    det_x1, det_y1, det_x2, det_y2 = det
    gt_x1, gt_y1, gt_x2, gt_y2 = gt
    
    x_left = max(det_x1, gt_x1)
    y_top = max(det_y1, gt_y1)
    x_right = min(det_x2, gt_x2)
    y_bottom = min(det_y2, gt_y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    area_intersection = (x_right - x_left) * (y_bottom - y_top)
    det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    area_union = float(det_area + gt_area - area_intersection + 1E-6)
    iou = area_intersection / area_union
    return iou


def compute_map(det_boxes, gt_boxes, iou_threshold=0.5, method='area'):
    # det_boxes = [
    #   {
    #       'person' : [[x1, y1, x2, y2, score], ...],
    #       'car' : [[x1, y1, x2, y2, score], ...]
    #   }
    #   {det_boxes_img_2},
    #   ...
    #   {det_boxes_img_N},
    # ]
    #
    # gt_boxes = [
    #   {
    #       'person' : [[x1, y1, x2, y2], ...],
    #       'car' : [[x1, y1, x2, y2], ...]
    #   },
    #   {gt_boxes_img_2},
    #   ...
    #   {gt_boxes_img_N},
    # ]
    
    gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()}
    gt_labels = sorted(gt_labels)
    all_aps = {}
    # average precisions for ALL classes
    aps = []
    for idx, label in enumerate(gt_labels):
        # Get detection predictions of this class
        cls_dets = [
            [im_idx, im_dets_label] for im_idx, im_dets in enumerate(det_boxes)
            if label in im_dets for im_dets_label in im_dets[label]
        ]
        
        # cls_dets = [
        #   (0, [x1_0, y1_0, x2_0, y2_0, score_0]),
        #   ...
        #   (0, [x1_M, y1_M, x2_M, y2_M, score_M]),
        #   (1, [x1_0, y1_0, x2_0, y2_0, score_0]),
        #   ...
        #   (1, [x1_N, y1_N, x2_N, y2_N, score_N]),
        #   ...
        # ]
        
        # Sort them by confidence score
        cls_dets = sorted(cls_dets, key=lambda k: -k[1][-1])
        
        # For tracking which gt boxes of this class have already been matched
        gt_matched = [[False for _ in im_gts[label]] for im_gts in gt_boxes]
        # Number of gt boxes for this class for recall calculation
        num_gts = sum([len(im_gts[label]) for im_gts in gt_boxes])
        tp = [0] * len(cls_dets)
        fp = [0] * len(cls_dets)
        
        # For each prediction
        for det_idx, (im_idx, det_pred) in enumerate(cls_dets):
            # Get gt boxes for this image and this label
            im_gts = gt_boxes[im_idx][label]
            max_iou_found = -1
            max_iou_gt_idx = -1
            
            # Get best matching gt box
            for gt_box_idx, gt_box in enumerate(im_gts):
                gt_box_iou = get_iou(det_pred[:-1], gt_box)
                if gt_box_iou > max_iou_found:
                    max_iou_found = gt_box_iou
                    max_iou_gt_idx = gt_box_idx
            # TP only if iou >= threshold and this gt has not yet been matched
            if max_iou_found < iou_threshold or gt_matched[im_idx][max_iou_gt_idx]:
                fp[det_idx] = 1
            else:
                tp[det_idx] = 1
                # If tp then we set this gt box as matched
                gt_matched[im_idx][max_iou_gt_idx] = True
        # Cumulative tp and fp
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts, eps)
        precisions = tp / np.maximum((tp + fp), eps)

        if method == 'area':
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))
            
            # Replace precision values with recall r with maximum precision value
            # of any recall value >= r
            # This computes the precision envelope
            for i in range(precisions.size - 1, 0, -1):
                precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
            # For computing area, get points where recall changes value
            i = np.where(recalls[1:] != recalls[:-1])[0]
            # Add the rectangular areas to get ap
            ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
        elif method == 'interp':
            ap = 0.0
            for interp_pt in np.arange(0, 1 + 1E-3, 0.1):
                # Get precision values for recall values >= interp_pt
                prec_interp_pt = precisions[recalls >= interp_pt]
                
                # Get max of those precision values
                prec_interp_pt = prec_interp_pt.max() if prec_interp_pt.size > 0.0 else 0.0
                ap += prec_interp_pt
            ap = ap / 11.0
        else:
            raise ValueError('Method can only be area or interp')
        if num_gts > 0:
            aps.append(ap)
            all_aps[label] = ap
        else:
            all_aps[label] = np.nan
    # compute mAP at provided iou threshold
    mean_ap = sum(aps) / len(aps)
    return mean_ap, all_aps


def load_model_and_dataset(args):
    # Read the config file #
    config = FasterRCNNConfig()
    
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    voc = VOCDataset('test', im_dir=config.im_test_path, ann_dir=config.ann_test_path)
    test_dataset = DataLoader(voc, batch_size=1, shuffle=False)
    
    faster_rcnn_model = FasterRCNN(config)
    faster_rcnn_model.eval()
    faster_rcnn_model.to(device)
    faster_rcnn_model.load_state_dict(torch.load(os.path.join(config.task_name,
                                                              config.ckpt_name),
                                                 map_location=device))
    return faster_rcnn_model, voc, test_dataset


def infer(args):
    out_dir = 'output'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    faster_rcnn_model, voc, test_dataset = load_model_and_dataset(args)
    
    # Hard coding the low score threshold for inference on images for now
    # Should come from config
    faster_rcnn_model.roi_head.low_score_threshold = 0.7
    
    for sample_count in tqdm(range(5)):
        random_idx = random.randint(0, len(voc))
        im, target, fname = voc[random_idx]
        im = im.unsqueeze(0).float().to(device)

        gt_im = cv2.imread(fname)
        gt_im_copy = gt_im.copy()
        
        # Saving images with ground truth boxes
        for idx, box in enumerate(target['bboxes']):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            cv2.rectangle(gt_im, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            cv2.rectangle(gt_im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            text = voc.idx2label[target['labels'][idx].detach().cpu().item()]
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(gt_im_copy , (x1, y1), (x1 + 10+text_w, y1 + 10+text_h), [255, 255, 255], -1)
            cv2.putText(gt_im, text=voc.idx2label[target['labels'][idx].detach().cpu().item()],
                        org=(x1+5, y1+15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(gt_im_copy, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
        cv2.addWeighted(gt_im_copy, 0.7, gt_im, 0.3, 0, gt_im)
        
        # Getting predictions from trained model
        rpn_output, frcnn_output = faster_rcnn_model(im, None)
        boxes = frcnn_output['boxes']
        labels = frcnn_output['labels']
        scores = frcnn_output['scores']
        im = cv2.imread(fname)
        im_copy = im.copy()
        
        # Saving images with predicted boxes
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(im, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            cv2.rectangle(im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            text = '{} : {:.2f}'.format(voc.idx2label[labels[idx].detach().cpu().item()],
                                        scores[idx].detach().cpu().item())
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(im_copy , (x1, y1), (x1 + 10+text_w, y1 + 10+text_h), [255, 255, 255], -1)
            cv2.putText(im, text=text,
                        org=(x1+5, y1+15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(im_copy, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
        cv2.addWeighted(im_copy, 0.7, im, 0.3, 0, im)
        
        # Concatenate images horizontally
        concatenated_im = cv2.hconcat([im, gt_im])
        cv2.imwrite(f'{out_dir}/comparison_{sample_count}.png', concatenated_im)


def evaluate_map(args):
    faster_rcnn_model, voc, test_dataset = load_model_and_dataset(args)
    gts = []
    preds = []
    for im, target, fname in tqdm(test_dataset):
        im_name = fname
        im = im.float().to(device)
        target_boxes = target['bboxes'].float().to(device)[0]
        target_labels = target['labels'].long().to(device)[0]
        rpn_output, frcnn_output = faster_rcnn_model(im, None)

        boxes = frcnn_output['boxes']
        labels = frcnn_output['labels']
        scores = frcnn_output['scores']
        
        pred_boxes = {}
        gt_boxes = {}
        for label_name in voc.label2idx:
            pred_boxes[label_name] = []
            gt_boxes[label_name] = []
        
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            label = labels[idx].detach().cpu().item()
            score = scores[idx].detach().cpu().item()
            label_name = voc.idx2label[label]
            pred_boxes[label_name].append([x1, y1, x2, y2, score])
        for idx, box in enumerate(target_boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            label = target_labels[idx].detach().cpu().item()
            label_name = voc.idx2label[label]
            gt_boxes[label_name].append([x1, y1, x2, y2])
        
        gts.append(gt_boxes)
        preds.append(pred_boxes)
   
    mean_ap, all_aps = compute_map(preds, gts, method='interp')
    print('Class Wise Average Precisions')
    for idx in range(len(voc.idx2label)):
        print('AP for class {} = {:.4f}'.format(voc.idx2label[idx], all_aps[voc.idx2label[idx]]))
    print('Mean Average Precision : {:.4f}'.format(mean_ap))


def infer_torchvision(args):
    out_dir = 'output_torchvision'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    config = FasterRCNNConfig()
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    voc = VOCDataset('test', im_dir=config.im_test_path, ann_dir=config.ann_test_path)
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    model.to(device)

    COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    for sample_count in tqdm(range(5)):
        random_idx = random.randint(0, len(voc))
        im, target, fname = voc[random_idx]
        im_tensor = im.to(device)

        gt_im = cv2.imread(fname)
        gt_im_copy = gt_im.copy()
        
        for idx, box in enumerate(target['bboxes']):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            cv2.rectangle(gt_im, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            cv2.rectangle(gt_im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            text = voc.idx2label[target['labels'][idx].detach().cpu().item()]
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(gt_im_copy , (x1, y1), (x1 + 10+text_w, y1 + 10+text_h), [255, 255, 255], -1)
            cv2.putText(gt_im, text=voc.idx2label[target['labels'][idx].detach().cpu().item()],
                        org=(x1+5, y1+15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(gt_im_copy, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
        cv2.addWeighted(gt_im_copy, 0.7, gt_im, 0.3, 0, gt_im)
        
        predictions = model([im_tensor])
        
        boxes = predictions[0]['boxes']
        labels = predictions[0]['labels']
        scores = predictions[0]['scores']
        im = cv2.imread(fname)
        im_copy = im.copy()
        
        for idx, box in enumerate(boxes):
            if scores[idx] < 0.7:
                continue
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(im, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            cv2.rectangle(im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            label_name = COCO_INSTANCE_CATEGORY_NAMES[labels[idx].item()]
            text = '{} : {:.2f}'.format(label_name, scores[idx].item())
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(im_copy , (x1, y1), (x1 + 10+text_w, y1 + 10+text_h), [255, 255, 255], -1)
            cv2.putText(im, text=text,
                        org=(x1+5, y1+15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(im_copy, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
        cv2.addWeighted(im_copy, 0.7, im, 0.3, 0, im)
        
        concatenated_im = cv2.hconcat([im, gt_im])
        cv2.imwrite(f'{out_dir}/comparison_{sample_count}.png', concatenated_im)


def preprocess_image_for_inference(image_path):
    im = cv2.imread(image_path)
    if im is None:
        print(f"Error: Image not found at {image_path}")
        return None, None

    original_im_bgr = im.copy() # Keep original BGR for drawing
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # Convert to RGB for model input

    # Convert to tensor and normalize
    im_tensor = torch.from_numpy(im).permute(2, 0, 1).float().to(device) / 255.0
    mean = torch.as_tensor(IMAGE_MEAN, dtype=torch.float32, device=device)
    std = torch.as_tensor(IMAGE_STD, dtype=torch.float32, device=device)
    im_tensor = (im_tensor - mean[:, None, None]) / std[:, None, None]
    im_tensor = im_tensor.unsqueeze(0) # Add batch dimension

    return im_tensor, original_im_bgr


def infer_single_image(image_path, args):
    out_dir = 'output'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    faster_rcnn_model, voc, _ = load_model_and_dataset(args)
    
    # Hard coding the low score threshold for inference on images for now
    # Should come from config
    faster_rcnn_model.roi_head.low_score_threshold = 0.7

    # --- Preprocessing ---
    im_tensor, original_im_bgr = preprocess_image_for_inference(image_path)
    if im_tensor is None:
        return
    
    # --- Inference ---
    # The model's forward method handles resizing and normalization internally
    _, frcnn_output = faster_rcnn_model(im_tensor, None)
    boxes = frcnn_output['boxes']
    labels = frcnn_output['labels']
    scores = frcnn_output['scores']

    # --- Postprocessing ---
    im_final = original_im_bgr.copy()

    for idx, box in enumerate(boxes):
        # The model's output boxes are already scaled back to original image size
        x1, y1, x2, y2 = box.detach().cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw bounding box
        cv2.rectangle(im_final, (x1, y1), (x2, y2), thickness=2, color=(0, 0, 255))

        # Prepare text
        label_name = voc.idx2label[labels[idx].detach().cpu().item()]
        score_val = scores[idx].detach().cpu().item()
        text = f'{label_name}: {score_val:.2f}'

        # Get text size
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1.2, 1)
        text_w, text_h = text_size

        # Position for text box (above the bounding box)
        text_x = x1
        text_y = y1 - 5

        # Draw text background and text
        cv2.rectangle(im_final, (text_x, text_y - text_h - 2), (text_x + text_w + 2, text_y), (255, 255, 255), -1)
        cv2.putText(im_final, text, (text_x + 2, text_y - 2), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 0), 1)

    # Save the output image
    file_name = os.path.basename(image_path)
    output_path = os.path.join(out_dir, f"pred_{file_name}")
    cv2.imwrite(output_path, im_final)
    print(f"Inference result saved to {output_path}")


def infer_single_image_torchvision(image_path, args):
    out_dir = 'output_torchvision'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    model.to(device)

    COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    im = cv2.imread(image_path)
    if im is None:
        print(f"Error: Image not found at {image_path}")
        return

    original_im_bgr = im.copy()
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_tensor = torchvision.transforms.functional.to_tensor(im_rgb).to(device)

    predictions = model([im_tensor])
    
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']
    
    im_final = original_im_bgr.copy()

    for idx, box in enumerate(boxes):
        if scores[idx] < 0.7: # Confidence threshold
            continue
        x1, y1, x2, y2 = box.detach().cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(im_final, (x1, y1), (x2, y2), thickness=2, color=(0, 0, 255))

        label_name = COCO_INSTANCE_CATEGORY_NAMES[labels[idx].item()]
        text = f'{label_name}: {scores[idx].item():.2f}'
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1.2, 1)
        text_w, text_h = text_size

        text_x = x1
        text_y = y1 - 5

        cv2.rectangle(im_final, (text_x, text_y - text_h - 2), (text_x + text_w + 2, text_y), (255, 255, 255), -1)
        cv2.putText(im_final, text, (text_x + 2, text_y - 2), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 0), 1)

    file_name = os.path.basename(image_path)
    output_path = os.path.join(out_dir, f"pred_torchvision_{file_name}")
    cv2.imwrite(output_path, im_final)
    print(f"Torchvision inference result saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn inference')
    parser.add_argument('--eval', action='store_true', help='Evaluate mAP on test set with custom model')
    parser.add_argument('--infer', action='store_true', help='Run inference on sample images with custom model')
    parser.add_argument('--torchvision', action='store_true', help='Run inference on sample images with torchvision model')
    parser.add_argument('--image_path', type=str, default=None, help='Path to a single image for inference with custom model')
    parser.add_argument('--image_path_torchvision', type=str, default=None, help='Path to a single image for inference with torchvision model')
    args = parser.parse_args()

    if args.image_path_torchvision:
        infer_single_image_torchvision(args.image_path_torchvision, args)
    elif args.image_path:
        infer_single_image(args.image_path, args)
    elif args.torchvision:
        infer_torchvision(args)
    elif args.eval:
        evaluate_map(args)
    elif args.infer:
        infer(args)
    else:
        infer(args)
