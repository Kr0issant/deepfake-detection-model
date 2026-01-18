import cv2
import numpy as np
from pathlib import Path
import urllib.request

class VideoFaceExtractor:
    def __init__(self, min_detection_confidence=0.5, target_size=(224, 224)):
        """
        Initialize video face extractor with MediaPipe
        
        Args:
            min_detection_confidence: Minimum confidence for face detection (0.0-1.0)
            target_size: Output face size (width, height)
        """
        self.target_size = target_size
        self.detector = None
        
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core import base_options as base_options_module
            
            # Download model if needed
            model_path = self._download_mediapipe_model()
            
            # Create FaceDetector with new API
            base_options = base_options_module.BaseOptions(model_asset_path=model_path)
            options = vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=min_detection_confidence
            )
            self.detector = vision.FaceDetector.create_from_options(options)
            print("✅ Using MediaPipe face detector")
            
        except Exception as e:
            print(f"❌ Failed to initialize MediaPipe: {e}")
            raise
    
    def _download_mediapipe_model(self):
        """Download MediaPipe face detection model if not present"""
        model_path = Path("blaze_face_short_range.tflite")
        
        if not model_path.exists():
            print("📥 Downloading MediaPipe model (~2MB)...")
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            urllib.request.urlretrieve(url, model_path)
            print("✅ Model downloaded")
        
        return str(model_path)
    
    def extract_faces_from_video(self, video_path, max_faces=None, padding_factor=0.3):
        """
        Extract faces from video
        
        Args:
            video_path: Path to video file
            max_faces: Maximum number of faces to extract (None = all frames)
            padding_factor: Padding around detected face (0.3 = 30%)
        
        Returns:
            List of face images (numpy arrays)
        """
        import mediapipe as mp
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            return []
        
        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if max_faces is None:
            max_faces = total_frames_in_video
        
        faces = []
        total_frames_processed = 0
        
        print(f"📹 Processing video: {video_path}")
        print(f"   Total frames: {total_frames_in_video}, FPS: {fps:.2f}")
        print(f"   Extracting up to {max_faces} faces...\n")
        
        while cap.isOpened() and len(faces) < max_faces:
            success, frame = cap.read()
            if not success:
                break
            
            total_frames_processed += 1
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Detect faces
            results = self.detector.detect(mp_image)
            
            if results.detections:
                for detection in results.detections:
                    bbox = detection.bounding_box
                    ih, iw = frame.shape[:2]
                    
                    # Get bounding box coordinates (already in pixels)
                    x_min = max(0, bbox.origin_x)
                    y_min = max(0, bbox.origin_y)
                    x_max = min(iw, bbox.origin_x + bbox.width)
                    y_max = min(ih, bbox.origin_y + bbox.height)
                    
                    # Add padding
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    expand_w = int(width * padding_factor)
                    expand_h = int(height * padding_factor)
                    
                    x_min = max(0, x_min - expand_w)
                    x_max = min(iw, x_max + expand_w)
                    y_min = max(0, y_min - expand_h)
                    y_max = min(ih, y_max + expand_h)
                    
                    # Crop face
                    face = frame[y_min:y_max, x_min:x_max]
                    
                    if face.size == 0:
                        continue
                    
                    # Resize to target size
                    face_resized = cv2.resize(face, self.target_size, interpolation=cv2.INTER_AREA)
                    
                    faces.append(face_resized)
                    
                    # Progress indicator
                    if len(faces) % 10 == 0:
                        print(f"   Extracted {len(faces)} faces (frame {total_frames_processed}/{total_frames_in_video})")
                    
                    if len(faces) >= max_faces:
                        break
        
        cap.release()
        
        print(f"\n✅ Extraction complete:")
        print(f"   Processed {total_frames_processed} frames")
        print(f"   Extracted {len(faces)} faces")
        
        return faces
    
    def extract_faces_uniform_sampling(self, video_path, num_samples=50, padding_factor=0.3):
        """
        Extract faces by uniformly sampling frames from video
        
        Args:
            video_path: Path to video file
            num_samples: Number of frames to sample
            padding_factor: Padding around face
        
        Returns:
            List of face images
        """
        import mediapipe as mp
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to sample
        if total_frames <= num_samples:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
        
        faces = []
        
        print(f"📹 Sampling {len(frame_indices)} frames from {video_path} ({total_frames} total frames)")
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            
            if not success:
                continue
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Detect faces
            results = self.detector.detect(mp_image)
            
            if results.detections:
                # Take the first (most confident) detection
                detection = results.detections[0]
                bbox = detection.bounding_box
                ih, iw = frame.shape[:2]
                
                # Get coordinates
                x_min = max(0, bbox.origin_x)
                y_min = max(0, bbox.origin_y)
                x_max = min(iw, bbox.origin_x + bbox.width)
                y_max = min(ih, bbox.origin_y + bbox.height)
                
                # Add padding
                width = x_max - x_min
                height = y_max - y_min
                
                expand_w = int(width * padding_factor)
                expand_h = int(height * padding_factor)
                
                x_min = max(0, x_min - expand_w)
                x_max = min(iw, x_max + expand_w)
                y_min = max(0, y_min - expand_h)
                y_max = min(ih, y_max + expand_h)
                
                # Crop and resize
                face = frame[y_min:y_max, x_min:x_max]
                
                if face.size > 0:
                    face_resized = cv2.resize(face, self.target_size, interpolation=cv2.INTER_AREA)
                    faces.append(face_resized)
        
        cap.release()
        
        print(f"✅ Extracted {len(faces)} faces from {len(frame_indices)} sampled frames\n")
        
        return faces
    
    def save_faces(self, faces, output_folder="extracted_faces"):
        """
        Save extracted faces to folder
        
        Args:
            faces: List of face images
            output_folder: Output directory path
        """
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, face in enumerate(faces):
            output_file = output_path / f"face_{i:04d}.jpg"
            cv2.imwrite(str(output_file), face)
        
        print(f"💾 Saved {len(faces)} faces to {output_folder}/")
    
    def extract_face(self, image, padding_factor=0.3):
        """
        Extract face from a single image
        
        Args:
            image: Input image (numpy array, BGR or RGB)
            padding_factor: Padding around detected face
            
        Returns:
            Resized face image (numpy array) or None if no face found
        """
        import mediapipe as mp
        
        # Convert BGR to RGB if needed (assuming input might be BGR from cv2)
        # MediaPipe expects RGB
        if image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = self.detector.detect(mp_image)
        
        if not results.detections:
            return None
            
        # Take the first detection
        detection = results.detections[0]
        bbox = detection.bounding_box
        ih, iw = image.shape[:2]
        
        x_min = max(0, bbox.origin_x)
        y_min = max(0, bbox.origin_y)
        x_max = min(iw, bbox.origin_x + bbox.width)
        y_max = min(ih, bbox.origin_y + bbox.height)
        
        width = x_max - x_min
        height = y_max - y_min
        
        expand_w = int(width * padding_factor)
        expand_h = int(height * padding_factor)
        
        x_min = max(0, x_min - expand_w)
        x_max = min(iw, x_max + expand_w)
        y_min = max(0, y_min - expand_h)
        y_max = min(ih, y_max + expand_h)
        
        face = image[y_min:y_max, x_min:x_max]
        
        if face.size == 0:
            return None
            
        face_resized = cv2.resize(face, self.target_size, interpolation=cv2.INTER_AREA)
        return face_resized

    def __del__(self):
        """Clean up resources"""
        if self.detector is not None:
            try:
                self.detector.close()
            except:
                pass


if __name__ == "__main__":
    # Initialize extractor
    extractor = VideoFaceExtractor(
        min_detection_confidence=0.5,
        target_size=(256, 256)
    )
    
    # Extract faces from video
    video_path = "vid.mp4"
    
    # Method 1: Extract all faces (or up to max_faces)
    faces = extractor.extract_faces_from_video(video_path, max_faces=100000)
    
    # Method 2: Uniform sampling (faster, good coverage)
    # faces = extractor.extract_faces_uniform_sampling(video_path, num_samples=50)
    
    # Save faces to disk
    if faces:
        extractor.save_faces(faces, output_folder="video_faces")
        print(f"✅ Total faces extracted: {len(faces)}")
    else:
        print("❌ No faces extracted")