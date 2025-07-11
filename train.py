import torch
import os
import numpy as np
import random
from tqdm import tqdm

from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from model.faster_rcnn import FasterRCNN
from dataset.voc_loader import VOCDataset
from config.voc_config import FasterRCNNConfig


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train():
    # Read the config file
    config = FasterRCNNConfig()
    
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    # Load training dataset
    voc = VOCDataset('train', im_dir=config.im_train_path, ann_dir=config.ann_train_path)
    train_dataset = DataLoader(voc, batch_size=1, shuffle=True, num_workers=4)
    
    # Load model
    faster_rcnn_model = FasterRCNN(config)
    faster_rcnn_model.train()
    faster_rcnn_model.to(device)

    if not os.path.exists(config.task_name):
        os.mkdir(config.task_name)
    optimizer = torch.optim.SGD(lr=config.lr,
                                params=filter(lambda p: p.requires_grad,
                                              faster_rcnn_model.parameters()),
                                weight_decay=5E-4,
                                momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=config.lr_steps, gamma=0.1)
    
    acc_steps = config.acc_steps
    num_epochs = config.num_epochs
    step_count = 1

    # Start training
    for i in range(num_epochs):
        rpn_classification_losses = []
        rpn_localization_losses = []
        frcnn_classification_losses = []
        frcnn_localization_losses = []
        optimizer.zero_grad()
        
        for im, target, fname in tqdm(train_dataset):
            im = im.float().to(device)
            target['bboxes'] = target['bboxes'].float().to(device)
            target['labels'] = target['labels'].long().to(device)
            rpn_output, frcnn_output = faster_rcnn_model(im, target)
            
            rpn_loss = rpn_output['rpn_classification_loss'] + rpn_output['rpn_localization_loss']
            frcnn_loss = frcnn_output['frcnn_classification_loss'] + frcnn_output['frcnn_localization_loss']
            loss = rpn_loss + frcnn_loss
            
            rpn_classification_losses.append(rpn_output['rpn_classification_loss'].item())
            rpn_localization_losses.append(rpn_output['rpn_localization_loss'].item())
            frcnn_classification_losses.append(frcnn_output['frcnn_classification_loss'].item())
            frcnn_localization_losses.append(frcnn_output['frcnn_localization_loss'].item())
            loss = loss / acc_steps
            loss.backward()
            if step_count % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            step_count += 1
        print('Finished epoch {}'.format(i))
        optimizer.step()
        optimizer.zero_grad()
        torch.save(faster_rcnn_model.state_dict(), os.path.join(config.task_name,
                                                                config.ckpt_name))
        loss_output = ''
        loss_output += 'RPN Classification Loss : {:.4f}'.format(np.mean(rpn_classification_losses))
        loss_output += ' | RPN Localization Loss : {:.4f}'.format(np.mean(rpn_localization_losses))
        loss_output += ' | FRCNN Classification Loss : {:.4f}'.format(np.mean(frcnn_classification_losses))
        loss_output += ' | FRCNN Localization Loss : {:.4f}'.format(np.mean(frcnn_localization_losses))
        print(loss_output)
        scheduler.step()
    print('Done Training...')


if __name__ == '__main__':
    train()
