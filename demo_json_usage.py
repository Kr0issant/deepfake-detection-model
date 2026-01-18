#!/usr/bin/env python3
"""
Demo script showing how to get JSON output from video analysis
"""
import sys
sys.path.insert(0, 'src/modules')
import json

def demo_json_output():
    """Show example of how the JSON output works"""

    print("=" * 70)
    print("VIDEO ANALYSIS JSON OUTPUT DEMO")
    print("=" * 70)

    # Example of what analyze_video() returns
    example_output = {
        "video_info": {
            "total_frames": 900,
            "processed_frames": 720,
            "fps": 30.0,
            "duration_seconds": 30.0,
            "overall_confidence": 0.82
        },
        "fake_segments": [
            {
                "segment_id": 1,
                "start_frame": 90,
                "end_frame": 270,
                "start_time_seconds": 3.0,
                "end_time_seconds": 9.0,
                "duration_seconds": 6.0,
                "timestamp_formatted": "0:03 - 0:09",
                "confidence": 0.88
            },
            {
                "segment_id": 2,
                "start_frame": 450,
                "end_frame": 630,
                "start_time_seconds": 15.0,
                "end_time_seconds": 21.0,
                "duration_seconds": 6.0,
                "timestamp_formatted": "0:15 - 0:21",
                "confidence": 0.91
            }
        ]
    }

    print("🎬 VIDEO ANALYSIS RESULT:")
    print(json.dumps(example_output, indent=2))

    print("\n" + "=" * 70)
    print("HOW TO USE IN CODE:")
    print("=" * 70)
    print("""
# In your Python code:
from inference import Inference
inference = Inference()

# Analyze a video file
with open('video.mp4', 'rb') as f:
    result = inference.analyze_video(f.read())

# Access the data
video_info = result['video_info']
fake_segments = result['fake_segments']

print(f"Video duration: {video_info['duration_seconds']}s")
print(f"Fake segments found: {len(fake_segments)}")

for segment in fake_segments:
    print(f"Segment {segment['segment_id']}: {segment['timestamp_formatted']} "
          f"(confidence: {segment['confidence']:.1%})")

# Save to JSON file
with open('analysis_result.json', 'w') as f:
    json.dump(result, f, indent=2)
    """)

if __name__ == "__main__":
    demo_json_output()