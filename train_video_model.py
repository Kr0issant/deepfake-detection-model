#!/usr/bin/env python3
"""
Train the video classification model for deepfake detection
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

from vid_classifier import VID_Classifier
from vid_dataset import DF_Dataset

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5  # Reduced for 30-min training
BATCH_SIZE = 2  # Reduced for GPU memory
LR = 1e-3
PATIENCE = 3  # Reduced patience
CHUNK_SIZE = 24  # Reduced chunk size to save memory

print(f"Using device: {DEVICE}")

# Load environment variables
load_dotenv()
VID_DATASET_PATH = os.environ.get("VID_DATASET_PATH")
VID_WEIGHTS_PATH = "VID-WEIGHT.pth"

if not VID_DATASET_PATH or not os.path.exists(VID_DATASET_PATH):
    print(f"ERROR: Dataset path not found: {VID_DATASET_PATH}")
    print("Please set VID_DATASET_PATH in .env file")
    sys.exit(1)

print(f"Dataset path: {VID_DATASET_PATH}")


def collate_variable_length_videos(batch):
    """
    Custom collate function to handle variable-length video sequences.
    Pads all videos to the max length in the batch.
    
    Args:
        batch: List of tuples (video_tensor, label_tensor, mask_tensor)
            video: [C, T, H, W] (channels first from frames_to_tensor)
            label: [T]
            mask: [T]
    
    Returns:
        padded_videos: [B, T_max, C, H, W]
        padded_labels: [B, T_max]
        padded_masks: [B, T_max]
    """
    videos, labels, masks = zip(*batch)
    
    # Videos are [C, T, H, W], we need temporal length at index 1
    lengths = [v.shape[1] for v in videos]
    max_len = max(lengths)
    
    batch_size = len(videos)
    c, h, w = videos[0].shape[0], videos[0].shape[2], videos[0].shape[3]
    
    # Create padded tensors
    padded_videos = torch.zeros(batch_size, max_len, c, h, w, dtype=videos[0].dtype)
    padded_labels = torch.zeros(batch_size, max_len, dtype=labels[0].dtype)
    padded_masks = torch.zeros(batch_size, max_len, dtype=masks[0].dtype)
    
    # Fill with actual data
    for i, (video, label, mask) in enumerate(zip(videos, labels, masks)):
        length = video.shape[1]  # Temporal dimension is at index 1
        # Transpose from [C, T, H, W] to [T, C, H, W] for each video
        padded_videos[i, :length] = video.permute(1, 0, 2, 3)
        padded_labels[i, :length] = label
        padded_masks[i, :length] = mask
    
    return padded_videos, padded_labels, padded_masks


def masked_bce_loss(logits, labels, mask):
    """BCE loss with masking for variable-length sequences"""
    bce = nn.BCEWithLogitsLoss(reduction='none')
    loss = bce(logits, labels)
    masked_loss = loss * mask
    return masked_loss.sum() / (mask.sum() + 1e-8)


def train_epoch(model, dataloader, optimizer, device, chunk_size=32):
    model.train()
    total_loss = 0
    total_correct = 0
    total_frames = 0
    
    loop = tqdm(dataloader, desc="Training", leave=False)
    
    for video_batch, label_batch, mask_batch in loop:
        video_batch = video_batch.to(device).float()
        label_batch = label_batch.to(device).float()
        mask_batch = mask_batch.to(device).float()
        
        optimizer.zero_grad()
        
        hidden_state = None
        b, t_total, c, h, w = video_batch.shape
        epoch_loss = 0
        
        # Process video in chunks
        for t in range(0, t_total, chunk_size):
            end_t = min(t + chunk_size, t_total)
            
            x_chunk = video_batch[:, t:end_t]
            y_chunk = label_batch[:, t:end_t]
            m_chunk = mask_batch[:, t:end_t]
            
            if x_chunk.shape[1] == 0:
                break
            
            logits, hidden_state = model(x_chunk, hidden_state)
            
            loss = masked_bce_loss(logits.squeeze(2), y_chunk, m_chunk)
            loss.backward()
            
            if hidden_state is not None:
                hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
            
            epoch_loss += loss.item()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += epoch_loss
        
        # Calculate accuracy
        with torch.no_grad():
            probs = torch.sigmoid(logits.squeeze(2))
            preds = (probs > 0.5).float()
            correct = ((preds == y_chunk) * m_chunk).sum().item()
            total_correct += correct
            total_frames += m_chunk.sum().item()
        
        loop.set_postfix(loss=epoch_loss, acc=total_correct/(total_frames+1e-8))
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / (total_frames + 1e-8)
    
    return avg_loss, avg_acc


def validate_epoch(model, dataloader, device, chunk_size=32):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_frames = 0
    
    with torch.no_grad():
        for video_batch, label_batch, mask_batch in dataloader:
            video_batch = video_batch.to(device).float()
            label_batch = label_batch.to(device).float()
            mask_batch = mask_batch.to(device).float()
            
            hidden_state = None
            b, t_total, c, h, w = video_batch.shape
            epoch_loss = 0
            
            for t in range(0, t_total, chunk_size):
                end_t = min(t + chunk_size, t_total)
                
                x_chunk = video_batch[:, t:end_t]
                y_chunk = label_batch[:, t:end_t]
                m_chunk = mask_batch[:, t:end_t]
                
                if x_chunk.shape[1] == 0:
                    break
                
                logits, hidden_state = model(x_chunk, hidden_state)
                
                loss = masked_bce_loss(logits.squeeze(2), y_chunk, m_chunk)
                
                if hidden_state is not None:
                    hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
                
                epoch_loss += loss.item()
            
            total_loss += epoch_loss
            
            probs = torch.sigmoid(logits.squeeze(2))
            preds = (probs > 0.5).float()
            correct = ((preds == y_chunk) * m_chunk).sum().item()
            total_correct += correct
            total_frames += m_chunk.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / (total_frames + 1e-8)
    
    return avg_loss, avg_acc


# Initialize model
print("\nInitializing model...")
classifier = VID_Classifier(LSTM_hidden_size=256, num_layers=2).to(DEVICE)

# Load datasets
print("Loading datasets...")
train_ds = DF_Dataset(VID_DATASET_PATH, epoch_size=12, training=True)  # 6 batches
val_ds = DF_Dataset(VID_DATASET_PATH, epoch_size=4, training=False)  # 2 batches

train_loader = DataLoader(
    train_ds, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=2,
    collate_fn=collate_variable_length_videos
)
val_loader = DataLoader(
    val_ds, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=2,
    collate_fn=collate_variable_length_videos
)

print(f"Training videos per epoch: {len(train_ds)}")
print(f"Validation videos per epoch: {len(val_ds)}")

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
    train_loss, train_acc = train_epoch(classifier, train_loader, optimizer, DEVICE, CHUNK_SIZE)
    
    # Validate
    val_loss, val_acc = validate_epoch(classifier, val_loader, DEVICE, CHUNK_SIZE)
    
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
        torch.save(classifier.state_dict(), VID_WEIGHTS_PATH)
        print(f"✓ Model saved! (Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%)")
    else:
        patience_counter += 1
        print(f"No improvement ({patience_counter}/{PATIENCE})")
    
    # Early stopping
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping triggered after {epoch+1} epochs")
        break
    
    # Close dataset workers to prevent memory leaks
    train_ds.close()
    val_ds.close()

print("\n" + "=" * 70)
print("Training completed!")
print(f"Best Val Loss: {best_val_loss:.4f}")
print(f"Best Val Acc: {best_val_acc*100:.2f}%")
print(f"Model saved to: {VID_WEIGHTS_PATH}")
