#!/usr/bin/env python3
"""
Train the image classification model for deepfake detection
"""
import sys
sys.path.insert(0, 'src/modules')

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dotenv import load_dotenv

from img_classifier import IMG_Classifier
from img_dataset import DF_Dataset

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 16
LR = 1e-3
PATIENCE = 5  # Early stopping patience

print(f"Using device: {DEVICE}")

# Load environment variables
load_dotenv()
IMG_DATASET_PATH = os.environ.get("IMG_DATASET_PATH")
IMG_WEIGHTS_PATH = "IMG-WEIGHT.pth"

if not IMG_DATASET_PATH or not os.path.exists(IMG_DATASET_PATH):
    print(f"ERROR: Dataset path not found: {IMG_DATASET_PATH}")
    print("Please set IMG_DATASET_PATH in .env file")
    sys.exit(1)

print(f"Dataset path: {IMG_DATASET_PATH}")

# Training functions
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    loop = tqdm(dataloader, desc="Training", leave=False)
    
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        logits = model(images).squeeze()
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct_preds += (preds == labels).sum().item()
        total_preds += labels.size(0)
        
        loop.set_postfix(loss=loss.item(), acc=correct_preds/total_preds)
        
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct_preds / total_preds
    
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images).squeeze()
            loss = criterion(logits, labels)
            
            running_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)
            
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct_preds / total_preds
    
    return epoch_loss, epoch_acc


# Initialize model
print("\nInitializing model...")
classifier = IMG_Classifier().to(DEVICE)

# Load datasets
print("Loading datasets...")
train_ds = DF_Dataset(IMG_DATASET_PATH, training=True)
val_ds = DF_Dataset(IMG_DATASET_PATH, training=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"Training samples: {len(train_ds)}")
print(f"Validation samples: {len(val_ds)}")

# Setup training
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(classifier.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

best_val_loss = float('inf')
best_val_acc = 0.0
patience_counter = 0

print(f"\nStarting training for {EPOCHS} epochs...")
print("=" * 70)

# Training loop
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    # Train
    train_loss, train_acc = train_epoch(classifier, train_loader, criterion, optimizer, DEVICE)
    
    # Validate
    val_loss, val_acc = validate_epoch(classifier, val_loader, criterion, DEVICE)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Logging
    print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%")
    print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}%")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(classifier.state_dict(), IMG_WEIGHTS_PATH)
        print(f"✓ Model saved! (Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%)")
    else:
        patience_counter += 1
        print(f"No improvement ({patience_counter}/{PATIENCE})")
    
    # Early stopping
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping triggered after {epoch+1} epochs")
        break

print("\n" + "=" * 70)
print("Training completed!")
print(f"Best Val Loss: {best_val_loss:.4f}")
print(f"Best Val Acc: {best_val_acc*100:.2f}%")
print(f"Model saved to: {IMG_WEIGHTS_PATH}")
