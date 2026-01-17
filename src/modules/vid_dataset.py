import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import cv2, json, random
import face_detector

from pathlib import Path
src_path = Path(__file__).resolve().parent.parent

import os
from dotenv import load_dotenv
load_dotenv()
VID_DATASET_PATH = os.getenv("VID_DATASET_PATH")

class DF_Dataset(Dataset):
    def __init__(self, dataset_path: str = ".", epoch_size: int = 400, training: bool = True):
        self.dataset_path = dataset_path
        self.epoch_size = epoch_size
        self.training = training

        if training:
            with open(str(src_path / "data/vid_train.json"), "r") as f:
                self.data = json.load(f)
        else:
            with open(str(src_path / "data/vid_test.json"), "r") as f:
                self.data = json.load(f)

    def __len__(self):
        return self.epoch_size
    
    def __getitem__(self, index):
        return self.create_video(True)

    # def split_ids(split: int = 0.7):
    #     ids = [i for i in range(62)]
    #     train_ids = sorted(random.sample(ids, int(62 * split)))
    #     test_ids = [i for i in ids if i not in train_ids]

    #     return {"train": train_ids, "test": test_ids}
    
    def create_video(self, tensorize = False):
        frames = []
        label_mask = []

        original_id, original_scene = self.get_random_originals()

        video_length = self.data[original_id][original_scene]["length"]
        segments = random.randint(1, 5)
        segment_length = video_length // segments

        face_detector.reset()

        for i in range(segments):
            f_range = [i * segment_length, ((i + 1) * segment_length) - 1]

            if random.random() < 0.3: # REAL
                frames.extend(self.get_video_frames(self.get_video_path(original_id, original_id, original_scene), f_range))
                label_mask.extend([0] * segment_length)
            else: # FAKE
                fake_id = self.get_random_fake_id(original_id, original_scene)
                frames.extend(self.get_video_frames(self.get_video_path(original_id, fake_id, original_scene), f_range))
                label_mask.extend([1] * segment_length)

        # print("video created", random.randint(1, 100))

        video_tensor = self.frames_to_tensor(frames) if tensorize else frames
        mask_tensor = torch.tensor(label_mask, dtype=torch.float32)

        return video_tensor, mask_tensor
    

    # Helper Functions
    def get_video_path(self, id1: str, id2: str, scene: str):
        return self.dataset_path + f"/id{id1}_id{id2}_{scene}.mp4"
    
    def get_video_frames(self, path: str, f_range: list[int]):
        cap = cv2.VideoCapture(path)
        frames = []

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_range[0])

            for _ in range(f_range[0], f_range[1] + 1):
                ret, frame = cap.read()
                
                if not ret: break
                
                frame = face_detector.extract_face(frame)
                frames.append(frame)
        finally:
            cap.release()

        return np.array(frames)
        
    def frames_to_tensor(self, frames):
        frames_np = np.array(frames, dtype=np.uint8) 
        return torch.from_numpy(frames_np).permute(3, 0, 1, 2)
    
    def get_random_originals(self):
        id: str = random.choice(list(self.data.keys()))
        scene: str = random.choice(list(self.data[id].keys()))
        num_variations = len(self.data[id][scene]["variations"])

        while (num_variations < 2):
            scene: str = random.choice(list(self.data[id].keys()))
            num_variations = len(self.data[id][scene]["variations"])
            
        return id, scene

    def get_random_fake_id(self, original_id: str, original_scene: str):
        id = random.choice(self.data[original_id][original_scene]["variations"])
        while id == original_id:
            id = random.choice(self.data[original_id][original_scene]["variations"])

        return str(id)


def variable_length_collate(batch):
    videos, masks = zip(*batch)
    videos = [v.permute(1, 0, 2, 3) for v in videos] 
    videos_padded = pad_sequence(videos, batch_first=True, padding_value=0)
    masks_padded = pad_sequence(masks, batch_first=True, padding_value=-1)
    
    return videos_padded, masks_padded


if __name__ == "__main__":
    BATCH_SIZE = 8
    EPOCH_SIZE = 400
    NUM_EPOCHS = 32

    ds = DF_Dataset(VID_DATASET_PATH, EPOCH_SIZE, training=True)

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=variable_length_collate)

    # Training Loop
    for i in range(NUM_EPOCHS):
        for batch_idx, (videos, labels) in enumerate(loader):
            videos = videos.float() / 255.0
            print("Batch processed.")
        print(f"Epoch - {i} completed.")