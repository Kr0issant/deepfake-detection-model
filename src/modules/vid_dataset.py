import numpy as np
import torch
from torch.utils.data import Dataset
import cv2, json, random
from pathlib import Path
from src.modules.face_detector import FaceDetector


src_path = Path(__file__).resolve().parent.parent

class DF_Dataset(Dataset):
    def __init__(self, dataset_path: str = ".", epoch_size: int = 400, training: bool = True):
        self.dataset_path = dataset_path
        self.epoch_size = epoch_size
        self.training = training
        self.face_detector = None
        
        # 1. Load Data
        json_path = src_path / ("data/vid_train.json" if training else "data/vid_test.json")
        with open(str(json_path), "r") as f:
            self.data = json.load(f)
            
        # 2. Pre-Calculate Valid Scenarios (The Fix)
        # We find all IDs that have at least 1 variation so we never get stuck looping
        self.valid_scenarios = []
        for id_key in self.data.keys():
            for scene_key in self.data[id_key].keys():
                # We need at least 1 variation to create a fake segment
                if len(self.data[id_key][scene_key].get("variations", [])) >= 1:
                    self.valid_scenarios.append((id_key, scene_key))
        
        print(f"✅ Loaded {len(self.valid_scenarios)} valid scenarios from {json_path.name}")
        
        if len(self.valid_scenarios) == 0:
            raise RuntimeError(f"❌ No valid videos found in {json_path}. Check your JSON structure!")
    @property
    def detector(self):
        if self.face_detector is None:
            self.face_detector = FaceDetector()
        
        return self.face_detector
    def __len__(self):
        return self.epoch_size
    
    def __getitem__(self, index):
        return self.create_video(True)

    def create_video(self, tensorize = False):
        frames = []
        label_mask = []
        
        # 3. Instant Safe Selection (No more while loops)
        original_id, original_scene = random.choice(self.valid_scenarios)

        video_length = self.data[original_id][original_scene]["length"]
        # Clamp length to prevent OOM
        video_length = min(video_length, 300)
        
        segments = random.randint(1, 5)
        segment_length = video_length // segments
        if segment_length == 0: segment_length = 1

        self.detector.reset()

        for i in range(segments):
            start = i * segment_length
            end = ((i + 1) * segment_length) - 1
            f_range = [start, end]
            
            # Coin flip: 30% Real, 70% Fake
            if random.random() < 0.3: # REAL
                video_path = self.get_video_path(original_id, original_id, original_scene)
                new_frames = self.get_video_frames(video_path, f_range)
                frames.extend(new_frames)
                label_mask.extend([0] * len(new_frames)) 
            else: # FAKE
                fake_id = self.get_random_fake_id(original_id, original_scene)
                video_path = self.get_video_path(original_id, fake_id, original_scene)
                new_frames = self.get_video_frames(video_path, f_range)
                frames.extend(new_frames)
                label_mask.extend([1] * len(new_frames))

        # Handle case where video read failed completely
        if len(frames) == 0:
            # Create a single black frame to prevent crash
            frames = [np.zeros((256, 256, 3), dtype=np.uint8)]
            label_mask = [0]

        video_tensor = self.frames_to_tensor(frames) if tensorize else frames
        mask_tensor = torch.tensor(label_mask, dtype=torch.float32)

        return video_tensor, mask_tensor
    

    # --- HELPER FUNCTIONS ---

    def get_random_fake_id(self, original_id: str, original_scene: str):
        variations = self.data[original_id][original_scene]["variations"]
        
        # Try to find a variation that isn't the original ID
        candidates = [v for v in variations if str(v) != original_id]
        
        if candidates:
            return str(random.choice(candidates))
        else:
            # Fallback: If only the original exists in variations (data error), return it
            return str(variations[0])

    def get_video_path(self, id1: str, id2: str, scene: str):
        return self.dataset_path + f"/id{id1}_id{id2}_{scene}.mp4"
    
    def get_video_frames(self, path: str, f_range: list[int]):
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_range[0])
            count = 0
            max_frames_per_chunk = f_range[1] - f_range[0] + 1
            
            while count < max_frames_per_chunk:
                ret, frame = cap.read()
                if not ret: break
                
                frame = self.detector.extract_face(frame)
                
                if frame is not None:
                    # --- CRITICAL FIX: RESIZE TO 256x256 ---
                    # Prevents "mat1 and mat2 shapes cannot be multiplied"
                    frame = cv2.resize(frame, (256, 256))
                    frames.append(frame)
                
                count += 1
        except Exception as e:
            print(f"Error reading frames from {path}: {e}")
        finally:
            cap.release()

        return np.array(frames)
        
    def frames_to_tensor(self, frames):
        frames_np = np.array(frames, dtype=np.uint8) 
        return torch.from_numpy(frames_np).permute(3, 0, 1, 2)