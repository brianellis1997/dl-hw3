import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
from tqdm import tqdm

from homework.models import load_model, save_model
from homework.datasets.classification_dataset import load_data
from homework.metrics import AccuracyMetric


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    metric = AccuracyMetric()
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Training")):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        metric.add(predictions.cpu(), labels.cpu())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = metric.compute()['accuracy']
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    metric = AccuracyMetric()
    
    with torch.inference_mode():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            metric.add(predictions.cpu(), labels.cpu())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = metric.compute()['accuracy']
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train classification model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--log_dir', type=str, default='runs/classification', help='Tensorboard log directory')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading data...")
    train_loader = load_data(
        'classification_data/train',
        transform_pipeline='aug',
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_loader = load_data(
        'classification_data/val',
        transform_pipeline='default',
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    print("Creating model...")
    model = load_model('classifier', with_weights=False)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)
    
    writer = SummaryWriter(args.log_dir)
    
    best_val_acc = 0
    
    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = save_model(model)
            print(f"Saved best model with val accuracy: {val_acc:.4f}")
        
        if val_acc >= 0.80:
            print(f"Reached target accuracy of 0.80! Current: {val_acc:.4f}")
    
    writer.close()
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
    
    if best_val_acc < 0.80:
        print("Warning: Did not reach target accuracy of 0.80")
        print("Consider: increasing epochs, adjusting hyperparameters, or improving data augmentation")


if __name__ == '__main__':
    main()