import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pathlib import Path
src_path = Path(__file__).resolve().parent.parent
model_path = str(src_path / "models" / "face-detector" / "blaze_face_short_range.tflite")

base_options = python.BaseOptions(
    model_asset_path=model_path,
    # delegate=python.BaseOptions.Delegate.GPU
)

options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

frame_count = 0

smoothed_center = None
SMOOTHING = 0.2
DETECTION_INTERVAL = 2
last_bbox = None

def extract_face(frame, target_size=256):
    global smoothed_center, last_bbox, frame_count
    ih, iw, _ = frame.shape
    
    if frame_count % DETECTION_INTERVAL == 0:
        frame_count = 0
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        results = detector.detect(mp_image)
        
        if results.detections:
            last_bbox = results.detections[0].bounding_box
    frame_count += 1

    if last_bbox:
        curr_cx = last_bbox.origin_x + (last_bbox.width / 2)
        curr_cy = last_bbox.origin_y + (last_bbox.height / 2) - (last_bbox.height * 0.1)

        if smoothed_center is None:
            smoothed_center = (curr_cx, curr_cy)
        else:
            smoothed_center = (
                smoothed_center[0] * (1 - SMOOTHING) + curr_cx * SMOOTHING,
                smoothed_center[1] * (1 - SMOOTHING) + curr_cy * SMOOTHING
            )
    
    cx, cy = smoothed_center if smoothed_center else (iw // 2, ih // 2)

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

def reset():
    global smoothed_center, last_bbox, frame_count
    
    smoothed_center = None
    last_bbox = None
    frame_count = 0