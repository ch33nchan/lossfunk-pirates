# src/train_world_model.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import torch.nn as nn
from tqdm import tqdm

from data_loader import PiratesLazyDataset, collate_fn
from models import WorldModel_ViT

# --- Config ---
BASE_DIR = '..' 
DATASET_DIR = os.path.join(BASE_DIR, 'PIRATES_DATASET')
VIDEO_DIR = os.path.join(DATASET_DIR, 'videos')
LOG_PATH = os.path.join(DATASET_DIR, 'raw_data/raw_data.csv')
CHECKPOINT_DIR = 'checkpoints'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
BATCH_SIZE = 16 
LEARNING_RATE = 1e-4
EPOCHS = 3
SEQUENCE_LENGTH = 16

def train():
    # --- FIX: Define the device at the very beginning ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Initializing lazy dataset...")
    full_dataset = PiratesLazyDataset(
        video_dir=VIDEO_DIR, 
        log_path=LOG_PATH, 
        sequence_length=SEQUENCE_LENGTH
    )
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    print(f"Training on {len(train_dataset)} sequences, validating on {len(val_dataset)}.")

    # --- Model and Training Loop ---
    model = WorldModel_ViT(num_frames=SEQUENCE_LENGTH).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    recon_criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
        for batch in progress_bar:
            if batch[0] is None: continue
            frame_seq, _, next_frame = batch
            
            frame_seq = frame_seq.to(device)
            next_frame = next_frame.to(device)
            
            optimizer.zero_grad()
            recon_frame, vq_loss, _ = model(frame_seq)
            recon_loss = recon_criterion(recon_frame, next_frame)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': total_train_loss / (progress_bar.n + 1)})

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
             for batch in val_loader:
                if batch[0] is None: continue
                frame_seq, _, next_frame = batch
                frame_seq, next_frame = frame_seq.to(device), next_frame.to(device)
                
                recon_frame, vq_loss, _ = model(frame_seq)
                recon_loss = recon_criterion(recon_frame, next_frame)
                loss = recon_loss + vq_loss
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        print(f"Epoch {epoch+1}/{EPOCHS} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f'world_model_epoch_{epoch+1}.pth'))

if __name__ == '__main__':
    train()