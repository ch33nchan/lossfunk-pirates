import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm

class PiratesLazyDataset(Dataset):
    """
    Processes video data on-the-fly to avoid memory and disk space issues.
    This version manually counts frames to be robust to corrupted video metadata.
    """
    def __init__(self, video_dir, log_path, game_name_in_filename="platform", sequence_length=16, frame_size=(128, 128)):
        self.video_dir = video_dir
        self.log_path = log_path
        self.game_name_filter = f"_{game_name_in_filename}_"
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        
        print("Scanning dataset to build sequence index...")
        self.video_files = self._get_game_videos(self.video_dir)
        self.logs = pd.read_csv(self.log_path, low_memory=False)
        self.index_map = self._create_index_map()
        
        if not self.index_map:
            raise RuntimeError("Failed to find any valid video sequences. Check video paths and file integrity.")
            
        print(f"Index created. Found {len(self.index_map)} possible sequences.")

    def _get_game_videos(self, video_dir):
        return [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.webm') and self.game_name_filter in f]

    def _create_index_map(self):
        index_map = []
        for video_path in tqdm(self.video_files, desc="Building Index"):
            try:
                cap = cv2.VideoCapture(video_path)
                num_frames = 0
                while True:
                    ret, _ = cap.read()
                    if not ret:
                        break
                    num_frames += 1
                cap.release()
            except Exception as e:
                print(f"Warning: Could not process video {os.path.basename(video_path)}. Error: {e}")
                continue

            if num_frames > self.sequence_length:
                for i in range(num_frames - self.sequence_length):
                    index_map.append({'video_path': video_path, 'start_frame': i})
        return index_map

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        item_info = self.index_map[idx]
        video_path = item_info['video_path']
        start_frame = item_info['start_frame']
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for _ in range(self.sequence_length + 1):
            ret, frame = cap.read()
            if not ret: return None
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
            frames.append(frame / 255.0)
        cap.release()

        frames = np.array(frames, dtype=np.float32)
        
        frame_seq = frames[:self.sequence_length]
        next_frame = frames[self.sequence_length]

        video_name = os.path.basename(video_path)
        session_id = video_name.split('_')[-1].replace('.webm', '')
        session_logs = self.logs[self.logs['[control]session_id'] == session_id].sort_values(by='[control]time_stamp')
        
        frame_rate = 30
        timestamp_t = (start_frame + self.sequence_length - 1) / frame_rate
        log_entry = session_logs[session_logs['[control]time_stamp'] <= timestamp_t]
        action_t = int(log_entry.iloc[-1]['[string]key_press_count'] > 0) if not log_entry.empty else 0
        
        frame_seq_tensor = torch.from_numpy(frame_seq).permute(0, 3, 1, 2)
        next_frame_tensor = torch.from_numpy(next_frame).permute(2, 0, 1)
        
        return frame_seq_tensor, action_t, next_frame_tensor

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, None
    return torch.utils.data.dataloader.default_collate(batch)