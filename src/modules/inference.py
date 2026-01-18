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
        self.vid_classifier.load_state_dict(torch.load(str(Path(src_path.parent / "VID-WEIGHT.pth")), map_location=device))
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
            
            # No ImageNet normalization - model was trained on [0,1] range images
            
            output = self.img_classifier.forward(face_tensor)
            
            # Apply sigmoid to get probability
            # BCEWithLogitsLoss trains model where sigmoid(logit) = P(label=1)
            # In IMAGE dataset CSV: 1=REAL, 0=FAKE
            # Therefore: sigmoid(output) = P(REAL)
            prob_real = torch.sigmoid(output).item()
            prob_fake = 1.0 - prob_real
            
            # Determine label (0 for Real, 1 for Fake) based on sensitivity
            # Higher sensitivity (e.g., 1.0) = lower threshold = detect more fakes
            # Lower sensitivity (e.g., 0.0) = higher threshold = detect fewer fakes
            threshold = 1.0 - self.post_processor.sensitivity
            label = 1 if prob_fake > threshold else 0
            
            # Actual Model Confidence: Maximum probability between real and fake
            # This shows the model's actual confidence in its prediction
            confidence = max(prob_real, prob_fake)
            
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
            
            total_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frames = []
            frame_numbers = []  # Track which video frames we actually processed
            
            try:
                frame_idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()

                    if not ret: break

                    try:
                        # Extract face from frame
                        # Note: Currently extracts the most prominent face
                        # For multiple faces, the face detector picks the highest confidence one
                        face = self.face_extractor.extract_face(frame)
                        
                        if face is not None:
                            face_tensor = torch.tensor(face).float() / 255.0
                            frames.append(face_tensor)
                            frame_numbers.append(frame_idx)
                    except Exception:
                        pass
                    
                    frame_idx += 1

            finally:
                cap.release()
                cv2.destroyAllWindows()

            if not frames:
                print("Warning: No faces detected in video")
                return [], 0.0
            
            print(f"Processed {len(frames)} frames with faces out of {total_frames_count} total frames")

            # Stack frames: (T, H, W, C) -> (T, C, H, W) -> (B, T, C, H, W)
            # We will process in chunks to avoid OOM
            
            device = next(self.vid_classifier.parameters()).device
            
            # No ImageNet normalization - model was trained on [0,1] range images
            
            chunk_size = 10 # Process 10 frames at a time
            all_outputs = []
            hidden_state = None
            
            with torch.no_grad():
                for i in range(0, len(frames), chunk_size):
                    chunk_frames = frames[i : i + chunk_size]
                    
                    # (T_chunk, H, W, C) -> (T_chunk, C, H, W) -> (1, T_chunk, C, H, W)
                    chunk_tensor = torch.stack(chunk_frames).permute(0, 3, 1, 2).unsqueeze(0)
                    chunk_tensor = chunk_tensor.to(device)
                    
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
            
            # Apply sigmoid to get probabilities
            # BCEWithLogitsLoss trains model where sigmoid(logit) = P(label=1)
            # In VIDEO dataset: 0=REAL, 1=FAKE (different from image dataset!)
            # Therefore: sigmoid(output) = P(FAKE)
            probs_fake = torch.sigmoid(full_output).squeeze().detach().cpu().numpy()
            
            if probs_fake.ndim == 0:
                probs_fake = np.array([probs_fake.item()])
            
            probs_real = 1.0 - probs_fake
            
            # Post-process to get streaks of FAKE (high prob_fake)
            # streaks format: (start_frame_idx, end_frame_idx, avg_conf_of_segment)
            # These are indices into our processed frames list, not original video frames
            streaks = self.post_processor.process_sequence(probs_fake)
            
            # Calculate Actual Model Confidence for Video
            # Use the average of max probabilities across all frames
            max_probs = np.maximum(probs_real, probs_fake)
            avg_confidence = np.mean(max_probs) if len(max_probs) > 0 else 0.0
            
            # Create JSON-compatible output structure
            video_analysis = {
                "video_info": {
                    "total_frames": total_frames_count,
                    "processed_frames": len(frames),
                    "fps": float(fps),
                    "duration_seconds": total_frames_count / fps if fps > 0 else 0,
                    "overall_confidence": float(avg_confidence)
                },
                "fake_segments": []
            }
            
            if streaks:
                # Convert processed frame indices to actual video timestamps
                for segment_idx, (start_idx, end_idx, conf) in enumerate(streaks):
                    # Get the actual video frame numbers
                    actual_start_frame = frame_numbers[start_idx] if start_idx < len(frame_numbers) else 0
                    actual_end_frame = frame_numbers[min(end_idx, len(frame_numbers)-1)]
                    
                    # Convert to time
                    start_time_sec = actual_start_frame / fps
                    end_time_sec = actual_end_frame / fps
                    
                    # Format as MM:SS
                    start_min = int(start_time_sec // 60)
                    start_sec = int(start_time_sec % 60)
                    end_min = int(end_time_sec // 60)
                    end_sec = int(end_time_sec % 60)
                    
                    segment_data = {
                        "segment_id": segment_idx + 1,
                        "start_frame": int(actual_start_frame),
                        "end_frame": int(actual_end_frame),
                        "start_time_seconds": float(start_time_sec),
                        "end_time_seconds": float(end_time_sec),
                        "duration_seconds": float(end_time_sec - start_time_sec),
                        "timestamp_formatted": f"{start_min}:{start_sec:02d} - {end_min}:{end_sec:02d}",
                        "confidence": float(conf)
                    }
                    
                    video_analysis["fake_segments"].append(segment_data)
            
            return video_analysis