import numpy as np
import cv2

def create_video_from_frames(frames, output_filename, fps):
    if not frames:
        print("No frames provided.")
        return
    height, width = frames[0].shape[:2]
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
    try:
        out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)
    except Exception as e:
        print(f"Error creating VideoWriter: {e}")
        return
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        out.write(frame)
    out.release()
    print(f"Video saved successfully as {output_filename}")