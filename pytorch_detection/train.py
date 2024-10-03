import argparse
import torch
from torch.utils.data import DataLoader

from src.config import get_config
from src.model import get_model
from src.utils import CustomDataset, get_train_transform, collate_fn
from src.trainer import train_fn

def main(args):
    config = get_config()
    train_dataset = CustomDataset(args.annotation, args.data_dir, get_train_transform()) 
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    model = get_model(args.num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    train_fn(args.num_epochs, train_data_loader, optimizer, model, device, args.save_path)

## 
## add_argument의 파일 경로를 수정할 필요가 있습니다.
##
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Faster R-CNN Training")
    parser.add_argument('--annotation', type=str, default='/data/ephemeral/home/deamin/dataset/train.json', help='path to annotation file')
    parser.add_argument('--data_dir', type=str, default='/data/ephemeral/home/deamin/dataset/', help='path to data directory')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--num_classes', type=int, default=11, help='number of classes')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--num_epochs', type=int, default=12, help='number of epochs')
    parser.add_argument('--save_path', type=str, default='./checkpoints/faster_rcnn_torchvision_checkpoints.pth', help='path to save model')
    
    args = parser.parse_args()
    
    main(args)