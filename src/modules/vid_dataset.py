import numpy as np
import torch
from torch.utils.data import Dataset
import cv2, json, random
from pathlib import Path
from face_detector import FaceDetector

src_path = Path(__file__).resolve().parent.parent

class DF_Dataset(Dataset):
    def __init__(self, dataset_path: str = ".", epoch_size: int = 400, training: bool = True):
        self.dataset_path = dataset_path
        self.epoch_size = epoch_size
        self.training = training
        self.face_detector = None
        
        if training:
            with open(str(src_path / "data/vid_train.json"), "r") as f:
                self.data = json.load(f)
        else:
            with open(str(src_path / "data/vid_test.json"), "r") as f:
                self.data = json.load(f)
        
    @property
    def detector(self):
        if self.face_detector is None:
            self.face_detector = FaceDetector()
        
        return self.face_detector
    
    def close(self):
        if self.face_detector is not None:
            self.face_detector.close()
            self.face_detector = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['face_detector'] = None 
        return state
    
    def __len__(self):
        return self.epoch_size
    
    def __getitem__(self, index):
        return self.create_video(True)

    def create_video(self, tensorize = False):
        frames = []
        label_mask = []

        original_id, original_scene = self.get_random_originals()

        video_length = self.data[original_id][original_scene]["length"]
        segments = random.randint(1, 5)
        segment_length = video_length // segments

        self.detector.reset()

        for i in range(segments):
            f_range = [i * segment_length, ((i + 1) * segment_length) - 1]

            if random.random() < 0.3: # REAL
                frames.extend(self.get_video_frames(self.get_video_path(original_id, original_id, original_scene), f_range))
                label_mask.extend([0] * segment_length)
            else: # FAKE
                fake_id = self.get_random_fake_id(original_id, original_scene)
                frames.extend(self.get_video_frames(self.get_video_path(original_id, fake_id, original_scene), f_range))
                label_mask.extend([1] * segment_length)

        if len(frames) == 0:
            # Create a single black frame to prevent crash
            frames = [np.zeros((256, 256, 3), dtype=np.uint8)]
            label_mask = [0]

        # print("video created", random.randint(1, 100))

        video_tensor = self.frames_to_tensor(frames) if tensorize else frames
        mask_tensor = torch.tensor(label_mask, dtype=torch.float32)

        return video_tensor, mask_tensor
    

    # --- HELPER FUNCTIONS ---

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
