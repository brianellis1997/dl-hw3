import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
from tqdm import tqdm

from homework.models import load_model, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import ConfusionMatrix


def train_epoch(model, dataloader, seg_criterion, depth_criterion, optimizer, device, seg_weight=1.0, depth_weight=1.0):
    model.train()
    total_loss = 0
    total_seg_loss = 0
    total_depth_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        images = batch['image']
        track_labels = batch['track'].long()
        depth_labels = batch['depth']
        
        optimizer.zero_grad()
        
        logits, pred_depth = model(images)
        
        seg_loss = seg_criterion(logits, track_labels)
        depth_loss = depth_criterion(pred_depth, depth_labels)
        
        loss = seg_weight * seg_loss + depth_weight * depth_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_seg_loss += seg_loss.item()
        total_depth_loss += depth_loss.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_seg_loss = total_seg_loss / len(dataloader)
    avg_depth_loss = total_depth_loss / len(dataloader)
    
    return avg_loss, avg_seg_loss, avg_depth_loss


def validate(model, dataloader, seg_criterion, depth_criterion, device):
    model.eval()
    total_seg_loss = 0
    total_depth_loss = 0
    
    confusion_matrix = ConfusionMatrix(num_classes=3)
    depth_errors = []
    
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Validation"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            images = batch['image']
            track_labels = batch['track'].long()
            depth_labels = batch['depth']
            
            logits, pred_depth = model(images)
            
            seg_loss = seg_criterion(logits, track_labels)
            depth_loss = depth_criterion(pred_depth, depth_labels)
            
            total_seg_loss += seg_loss.item()
            total_depth_loss += depth_loss.item()
            
            predictions = logits.argmax(dim=1)
            confusion_matrix.add(predictions.cpu(), track_labels.cpu())
            
            depth_error = torch.abs(pred_depth - depth_labels).mean().item()
            depth_errors.append(depth_error)
    
    avg_seg_loss = total_seg_loss / len(dataloader)
    avg_depth_loss = total_depth_loss / len(dataloader)
    avg_depth_error = sum(depth_errors) / len(depth_errors)
    
    metrics = confusion_matrix.iou()
    mean_iou = sum(metrics.values()) / len(metrics)
    accuracy = confusion_matrix.global_accuracy()
    
    return avg_seg_loss, avg_depth_loss, mean_iou, accuracy, avg_depth_error


def main():
    parser = argparse.ArgumentParser(description='Train detection model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--seg_weight', type=float, default=1.0, help='Weight for segmentation loss')
    parser.add_argument('--depth_weight', type=float, default=1.0, help='Weight for depth loss')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--log_dir', type=str, default='runs/detection', help='Tensorboard log directory')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading data...")
    train_loader = load_data(
        'drive_data/train',
        transform_pipeline='default',
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_loader = load_data(
        'drive_data/val',
        transform_pipeline='default',
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    print("Creating model...")
    model = load_model('detector', with_weights=False)
    model = model.to(device)
    
    seg_criterion = nn.CrossEntropyLoss()
    depth_criterion = nn.L1Loss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    writer = SummaryWriter(args.log_dir)
    
    best_iou = 0
    best_depth_error = float('inf')
    
    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_seg_loss, train_depth_loss = train_epoch(
            model, train_loader, seg_criterion, depth_criterion, 
            optimizer, device, args.seg_weight, args.depth_weight
        )
        
        val_seg_loss, val_depth_loss, mean_iou, accuracy, depth_error = validate(
            model, val_loader, seg_criterion, depth_criterion, device
        )
        
        total_val_loss = val_seg_loss + val_depth_loss
        scheduler.step(total_val_loss)
        
        print(f"Train - Total: {train_loss:.4f}, Seg: {train_seg_loss:.4f}, Depth: {train_depth_loss:.4f}")
        print(f"Val - Seg: {val_seg_loss:.4f}, Depth: {val_depth_loss:.4f}")
        print(f"Val - IoU: {mean_iou:.4f}, Accuracy: {accuracy:.4f}, Depth Error: {depth_error:.4f}")
        
        writer.add_scalar('Loss/train_total', train_loss, epoch)
        writer.add_scalar('Loss/train_seg', train_seg_loss, epoch)
        writer.add_scalar('Loss/train_depth', train_depth_loss, epoch)
        writer.add_scalar('Loss/val_seg', val_seg_loss, epoch)
        writer.add_scalar('Loss/val_depth', val_depth_loss, epoch)
        writer.add_scalar('Metrics/iou', mean_iou, epoch)
        writer.add_scalar('Metrics/accuracy', accuracy, epoch)
        writer.add_scalar('Metrics/depth_error', depth_error, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        if mean_iou > best_iou or (mean_iou == best_iou and depth_error < best_depth_error):
            best_iou = mean_iou
            best_depth_error = min(depth_error, best_depth_error)
            save_path = save_model(model)
            print(f"Saved best model - IoU: {mean_iou:.4f}, Depth Error: {depth_error:.4f}")
        
        if mean_iou >= 0.75 and depth_error < 0.05:
            print(f"Reached target metrics! IoU: {mean_iou:.4f}, Depth Error: {depth_error:.4f}")
    
    writer.close()
    print(f"\nTraining complete!")
    print(f"Best IoU: {best_iou:.4f}")
    print(f"Best Depth Error: {best_depth_error:.4f}")
    
    if best_iou < 0.75:
        print("Warning: Did not reach target IoU of 0.75")
    if best_depth_error >= 0.05:
        print("Warning: Did not reach target depth error < 0.05")


if __name__ == '__main__':
    main()