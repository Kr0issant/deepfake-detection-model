import sys
import os
import torch
import numpy as np
from PIL import Image
import cv2
import io

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import Inference

def test_inference():
    print("Initializing Inference...")
    try:
        inf = Inference()
    except Exception as e:
        print(f"Failed to initialize Inference: {e}")
        return

    # Test Image
    print("\nTesting Image Analysis...")
    try:
        # Create dummy image
        img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        img_pil = Image.fromarray(img)
        img_byte_arr = io.BytesIO()
        img_pil.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0) # Reset pointer
        
        # We need to mock face detector or use an image with a face.
        # Since we can't easily generate a face, we expect the code to handle "no face found" gracefully or we mock extract_face.
        # Let's mock extract_face on the instance
        
        # Mocking extract_face to return a dummy face
        original_extract = inf.face_extractor.extract_face
        inf.face_extractor.extract_face = lambda x: np.zeros((256, 256, 3), dtype=np.uint8)
        
        label, conf = inf.analyze_image(img_byte_arr)
        print(f"Image Result: Label={label}, Conf={conf}")
        
        # Restore
        inf.face_extractor.extract_face = original_extract
        
    except Exception as e:
        print(f"Image Analysis Failed: {e}")
        import traceback
        traceback.print_exc()

    # Test Video
    print("\nTesting Video Analysis...")
    try:
        # Create dummy video
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = tmp.name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tmp_path, fourcc, 30.0, (500, 500))
        for _ in range(30):
            frame = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        
        with open(tmp_path, 'rb') as f:
            video_bytes = f.read()
            
        # Mock extract_face again
        inf.face_extractor.extract_face = lambda x: np.zeros((256, 256, 3), dtype=np.uint8)
        
        streaks, avg_conf = inf.analyze_video(video_bytes)
        print(f"Video Result: Streaks={streaks}, Avg Conf={avg_conf}")
        
        os.remove(tmp_path)
        
    except Exception as e:
        print(f"Video Analysis Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference()
