import numpy as np
import cv2, tempfile, os
import torch
from PIL import Image
# from face_detector import FaceDetector # Deprecated
from videoface_extractor import VideoFaceExtractor
from postprocessor import Postprocessor

from img_classifier import IMG_Classifier
from vid_classifier import VID_Classifier

from pathlib import Path
src_path = Path(__file__).resolve().parent.parent

class Inference:
    def __init__(self):
        # self.face_detector = FaceDetector()
        self.face_extractor = VideoFaceExtractor(min_detection_confidence=0.5, target_size=(256, 256))
        self.post_processor = Postprocessor()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.img_classifier = IMG_Classifier().to(device)
        self.img_classifier.load_state_dict(torch.load(str(Path(src_path / "models/img-weights/IMG-WEIGHT.pth")), map_location=device))
        self.img_classifier.eval()

        self.vid_classifier = VID_Classifier(256, 2).to(device)
        self.vid_classifier.load_state_dict(torch.load(str(Path(src_path / "models/vid-weights/VID-WEIGHT.pth")), map_location=device))
        self.vid_classifier.eval()

    def update_variables(self, inertia = 10, sensitivity = 0.5):
        self.post_processor.update_variables(inertia, sensitivity)

    def analyze_image(self, file_bytes):
        try:
            # Handle both bytes and file-like objects
            if isinstance(file_bytes, bytes):
                import io
                img = np.asarray(Image.open(io.BytesIO(file_bytes)))
            else:
                img = np.asarray(Image.open(file_bytes))
                
            # face = self.face_detector.extract_face(img)
            # face = cv2.resize(face, (256, 256))
            face = self.face_extractor.extract_face(img)
            
            if face is None:
                print("Error: No face detected in image.")
                return 0, 0.0

            face_tensor = torch.tensor(face).float() / 255.0
            # Add batch dimension and permute to (B, C, H, W)
            face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)
            
            # Ensure device compatibility
            device = next(self.img_classifier.parameters()).device
            face_tensor = face_tensor.to(device)
            
            # ImageNet Normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            face_tensor = (face_tensor - mean) / std

            output = self.img_classifier.forward(face_tensor)
            
            # Apply sigmoid to get probability of REAL (assuming model outputs Real probability)
            # User wants FAKE to be 1 and REAL to be 0.
            # So we invert: prob_fake = 1 - prob_real
            prob_real = torch.sigmoid(output).item()
            prob_fake = 1.0 - prob_real
            
            # Determine label (0 for Real, 1 for Fake) based on sensitivity
            # If prob_fake > sensitivity -> FAKE (1)
            label = 1 if prob_fake > self.post_processor.sensitivity else 0
            
            # Relative Confidence:
            # If Label is FAKE, confidence is prob_fake.
            # If Label is REAL, confidence is 1 - prob_fake (which is prob_real).
            confidence = prob_fake if label == 1 else 1.0 - prob_fake
            
            return label, confidence

        except Exception as e:
            print("Error: Could not open image: ", e)
            return 0, 0.0

    def analyze_video(self, file_bytes):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, "tmp.mp4")

            with open(tmp_path, "wb") as f:
                # Handle both bytes and file-like objects
                if hasattr(file_bytes, 'read'):
                    file_bytes.seek(0)
                    f.write(file_bytes.read())
                else:
                    f.write(file_bytes)

            cap = cv2.VideoCapture(tmp_path)

            if not cap.isOpened():
                print("Error: Could not open temporary video file.")
                return [], 0.0
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0: fps = 30.0 # Default fallback
            
            frames = []
            
            try:
                while cap.isOpened():
                    ret, frame = cap.read()

                    if not ret: break

                    try:
                        # face = self.face_detector.extract_face(frame)
                        # face = cv2.resize(face, (256, 256))
                        face = self.face_extractor.extract_face(frame)
                        
                        if face is not None:
                            face_tensor = torch.tensor(face).float() / 255.0
                            frames.append(face_tensor)
                    except Exception:
                        continue

            finally:
                cap.release()
                cv2.destroyAllWindows()

            if not frames:
                return [], 0.0

            # Stack frames: (T, H, W, C) -> (T, C, H, W) -> (B, T, C, H, W)
            # We will process in chunks to avoid OOM
            
            device = next(self.vid_classifier.parameters()).device
            
            # ImageNet Normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1).to(device)
            
            chunk_size = 10 # Process 10 frames at a time
            all_outputs = []
            hidden_state = None
            
            with torch.no_grad():
                for i in range(0, len(frames), chunk_size):
                    chunk_frames = frames[i : i + chunk_size]
                    
                    # (T_chunk, H, W, C) -> (T_chunk, C, H, W) -> (1, T_chunk, C, H, W)
                    chunk_tensor = torch.stack(chunk_frames).permute(0, 3, 1, 2).unsqueeze(0)
                    chunk_tensor = chunk_tensor.to(device)
                    
                    # Apply Normalization
                    chunk_tensor = (chunk_tensor - mean) / std
                    
                    output, hidden_state = self.vid_classifier(chunk_tensor, hidden_state)
                    
                    # Detach hidden state to prevent graph retention (though no_grad handles this, good practice)
                    if isinstance(hidden_state, tuple):
                        hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
                    else:
                        hidden_state = hidden_state.detach()
                        
                    all_outputs.append(output.cpu())
                    
                    # Clean up
                    del chunk_tensor
                    del output
                    torch.cuda.empty_cache()

            # Concatenate all outputs: (1, T, 1)
            full_output = torch.cat(all_outputs, dim=1)
            
            # Apply sigmoid to get prob_real
            probs_real = torch.sigmoid(full_output).squeeze().detach().cpu().numpy()
            
            if probs_real.ndim == 0:
                probs_real = np.array([probs_real.item()])
                
            # Invert to get prob_fake
            probs_fake = 1.0 - probs_real
            
            # Post-process to get streaks of FAKE (high prob_fake)
            # streaks format: (start_frame, end_frame, avg_conf_of_segment)
            streaks = self.post_processor.process_sequence(probs_fake)
            
            # Calculate Weighted Average Confidence for Video
            if streaks:
                total_weighted_conf = 0.0
                total_frames = 0
                
                for start, end, conf in streaks:
                    length = end - start + 1
                    total_weighted_conf += conf * length
                    total_frames += length
                    
                avg_confidence = total_weighted_conf / total_frames if total_frames > 0 else 0.0
                
                # Convert to time streaks
                time_streaks = self.post_processor.frame_streaks_to_time_streaks(streaks, fps)
            else:
                # If Real (no streaks), confidence is 1 - average fake probability
                avg_confidence = 1.0 - np.mean(probs_fake) if len(probs_fake) > 0 else 0.0
                time_streaks = []
            
            return time_streaks, avg_confidence