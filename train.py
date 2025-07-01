import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from tqdm import tqdm

from frcnn.dataset import VOCDataset
from frcnn.models.faster_rcnn import FasterRCNN
from frcnn.trainer import FasterRCNNTrainer # Import the trainer


def train_one_epoch(trainer, dataloader, device, epoch):
    trainer.faster_rcnn.train()
    total_rpn_loc_loss = 0
    total_rpn_cls_loss = 0
    total_roi_loc_loss = 0
    total_roi_cls_loss = 0

    for i, (images, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        images = images.to(device)
        # Assuming batch size is 1 for now
        bboxes = [target['boxes'].to(device) for target in targets]
        labels = [target['labels'].to(device) for target in targets]
        scale = 1.0 # Assuming no image scaling for now

        # Forward pass and loss calculation through the trainer
        rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss = trainer(images, bboxes, labels, scale)

        loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss

        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()

        total_rpn_loc_loss += rpn_loc_loss.item()
        total_rpn_cls_loss += rpn_cls_loss.item()
        total_roi_loc_loss += roi_loc_loss.item()
        total_roi_cls_loss += roi_cls_loss.item()

    avg_rpn_loc_loss = total_rpn_loc_loss / len(dataloader)
    avg_rpn_cls_loss = total_rpn_cls_loss / len(dataloader)
    avg_roi_loc_loss = total_roi_loc_loss / len(dataloader)
    avg_roi_cls_loss = total_roi_cls_loss / len(dataloader)

    return avg_rpn_loc_loss, avg_rpn_cls_loss, avg_roi_loc_loss, avg_roi_cls_loss


def main():
    # Configuration
    dataset_root = './data/VOCdevkit'
    num_classes = 21  # 20 VOC classes + background
    batch_size = 1    # Object detection often uses batch size 1 due to variable image/object sizes
    learning_rate = 1e-4
    num_epochs = 1
    save_dir = './models'
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add more transforms like resizing, normalization later
    ])
    train_dataset = VOCDataset(root_dir=dataset_root, split='trainval', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    # Model, Optimizer, and Trainer
    model = FasterRCNN(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    trainer = FasterRCNNTrainer(model, optimizer)

    print("Starting training...")
    for epoch in range(num_epochs):
        avg_rpn_loc_loss, avg_rpn_cls_loss, avg_roi_loc_loss, avg_roi_cls_loss = train_one_epoch(trainer, train_dataloader, device, epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"RPN Loc Loss: {avg_rpn_loc_loss:.4f}, "
              f"RPN Cls Loss: {avg_rpn_cls_loss:.4f}, "
              f"RoI Loc Loss: {avg_roi_loc_loss:.4f}, "
              f"RoI Cls Loss: {avg_roi_cls_loss:.4f}")

        # Save model checkpoint
        torch.save(model.state_dict(), os.path.join(save_dir, f'faster_rcnn_epoch_{epoch+1}.pth'))
        print(f"Model saved to {os.path.join(save_dir, f'faster_rcnn_epoch_{epoch+1}.pth')}")

    print("Training complete.")


if __name__ == '__main__':
    main()