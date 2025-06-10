import torch
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm
import argparse

from models import WorldModel_ViT
from data_loader import PiratesLazyDataset, collate_fn

def extract_codes(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing dataset to extract latent codes...")
    full_dataset = PiratesLazyDataset(
        video_dir=args.video_dir, 
        log_path=args.log_path, 
        sequence_length=args.sequence_length
    )
    data_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)

    print(f"Loading trained world model from: {args.checkpoint_path}")
    model = WorldModel_ViT(num_frames=args.sequence_length).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    all_pairs = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting latent codes"):
            if batch[0] is None: continue
            frame_seq, actions, _ = batch
            
            frame_seq = frame_seq.to(device)
            
            _, _, indices = model(frame_seq)
            
            actions_list = actions.cpu().numpy().tolist()
            indices_list = indices.cpu().numpy().tolist()
            
            for i in range(len(actions_list)):
                all_pairs.append({
                    'action': actions_list[i],
                    'latent_code': indices_list[i]
                })

    print(f"Extraction complete. Found {len(all_pairs)} pairs.")
    print(f"Saving pairs to {args.output_json}...")
    with open(args.output_json, 'w') as f:
        json.dump(all_pairs, f)
    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract (action, latent_code) pairs using a trained World Model.")
    parser.add_argument('--video_dir', type=str, default='../PIRATES_DATASET/videos')
    parser.add_argument('--log_path', type=str, default='../PIRATES_DATASET/raw_data/raw_data.csv')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/world_model_epoch_3.pth')
    parser.add_argument('--output_json', type=str, default='data/pirates_action_latent_pairs.json')
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    extract_codes(args)