import torch
import os
import numpy as np
import random
from tqdm import tqdm
import torchvision

from torch.utils.data.dataloader import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset.voc_loader import VOCDataset
from config.tv_voc_config import TVFasterRCNNConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_function(data):
    return tuple(zip(*data))

def train():
    config = TVFasterRCNNConfig()
    
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    voc = VOCDataset('train',
                     im_dir=config.im_train_path,
                     ann_dir=config.ann_train_path)

    train_dataset = DataLoader(voc,
                               batch_size=config.batch_size,
                               shuffle=True,
                               num_workers=config.num_workers,
                               collate_fn=collate_function)

    faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
                                                                             min_size=config.min_size,
                                                                             max_size=config.max_size,
    )
    in_features = faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features
    faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=config.num_classes)

    faster_rcnn_model.train()
    faster_rcnn_model.to(device)
    if not os.path.exists(config.task_name):
        os.mkdir(config.task_name)

    optimizer = torch.optim.SGD(lr=config.lr,
                                params=filter(lambda p: p.requires_grad, faster_rcnn_model.parameters()),
                                weight_decay=config.weight_decay, momentum=config.momentum)

    num_epochs = config.num_epochs

    for i in range(num_epochs):
        rpn_classification_losses = []
        rpn_localization_losses = []
        frcnn_classification_losses = []
        frcnn_localization_losses = []
        for ims, targets, _ in tqdm(train_dataset):
            optimizer.zero_grad()
            for target in targets:
                target['boxes'] = target['bboxes'].float().to(device)
                del target['bboxes']
                target['labels'] = target['labels'].long().to(device)
            images = [im.float().to(device) for im in ims]
            batch_losses = faster_rcnn_model(images, targets)
            loss = batch_losses['loss_classifier']
            loss += batch_losses['loss_box_reg']
            loss += batch_losses['loss_rpn_box_reg']
            loss += batch_losses['loss_objectness']

            rpn_classification_losses.append(batch_losses['loss_objectness'].item())
            rpn_localization_losses.append(batch_losses['loss_rpn_box_reg'].item())
            frcnn_classification_losses.append(batch_losses['loss_classifier'].item())
            frcnn_localization_losses.append(batch_losses['loss_box_reg'].item())

            loss.backward()
            optimizer.step()
        print('Finished epoch {}'.format(i))
        
        torch.save(faster_rcnn_model.state_dict(), os.path.join(config.task_name,
                                                                'tv_frcnn_r50fpn_' + config.ckpt_name))
        loss_output = ''
        loss_output += 'RPN Classification Loss : {:.4f}'.format(np.mean(rpn_classification_losses))
        loss_output += ' | RPN Localization Loss : {:.4f}'.format(np.mean(rpn_localization_losses))
        loss_output += ' | FRCNN Classification Loss : {:.4f}'.format(np.mean(frcnn_classification_losses))
        loss_output += ' | FRCNN Localization Loss : {:.4f}'.format(np.mean(frcnn_localization_losses))
        print(loss_output)
    print('Done Training...')


if __name__ == '__main__':
    train()