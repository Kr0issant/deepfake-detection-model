import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

class FaceDetector:
    def __init__(self, smoothing: float = 0.2, detection_interval: int = 2):
        self.src_path = Path(__file__).resolve().parent.parent
        self.model_path = str(self.src_path / "models" / "face-detector" / "blaze_face_short_range.tflite")
        
        base_options = python.BaseOptions(
            model_asset_path=self.model_path,
            # delegate=python.BaseOptions.Delegate.GPU
        )
        options = vision.FaceDetectorOptions(base_options=base_options)
        self.detector = vision.FaceDetector.create_from_options(options)
        
        self.smoothing = smoothing
        self.detection_interval = detection_interval
        self.reset()

    def reset(self):
        self.smoothed_center = None
        self.last_bbox = None
        self.frame_count = 0

    def extract_face(self, frame, target_size=256):
        ih, iw, _ = frame.shape
        
        if self.frame_count % self.detection_interval == 0:
            self.frame_count = 0
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            results = self.detector.detect(mp_image)
            
            if results.detections:
                self.last_bbox = results.detections[0].bounding_box
        self.frame_count += 1

        if self.last_bbox:
            curr_cx = self.last_bbox.origin_x + (self.last_bbox.width / 2)
            curr_cy = self.last_bbox.origin_y + (self.last_bbox.height / 2) - (self.last_bbox.height * 0.1)

            if self.smoothed_center is None:
                self.smoothed_center = (curr_cx, curr_cy)
            else:
                self.smoothed_center = (
                    self.smoothed_center[0] * (1 - self.smoothing) + curr_cx * self.smoothing,
                    self.smoothed_center[1] * (1 - self.smoothing) + curr_cy * self.smoothing
                )
        
        cx, cy = self.smoothed_center if self.smoothed_center else (iw // 2, ih // 2)

        x_min, y_min = int(cx - target_size // 2), int(cy - target_size // 2)
        x_max, y_max = x_min + target_size, y_min + target_size

        c_ymin, c_ymax = max(0, y_min), min(ih, y_max)
        c_xmin, c_xmax = max(0, x_min), min(iw, x_max)
        face_crop = frame[c_ymin:c_ymax, c_xmin:c_xmax]

        if face_crop.shape[0] != target_size or face_crop.shape[1] != target_size:
            pad_t, pad_b = max(0, -y_min), max(0, y_max - ih)
            pad_l, pad_r = max(0, -x_min), max(0, x_max - iw)
            face_crop = cv2.copyMakeBorder(
                face_crop, pad_t, pad_b, pad_l, pad_r, 
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )

        return face_crop

# Create a default instance for backward compatibility if needed, 
# but mostly we expect instantiation. 
# However, to avoid breaking other imports immediately, we won't export a default instance 
# unless the user asks, but the implementation plan says to refactor module to remove global state.
# So I will just provide the class.