#!/usr/bin/env python3
"""
Simple script to extract fake segments with timestamps from video analysis
"""
import json

def extract_fake_timestamps(json_result):
    """Extract just the fake segments with timestamps and confidence"""

    if not json_result.get('fake_segments'):
        return {"message": "No fake segments detected", "fake_segments": []}

    segments = []
    for segment in json_result['fake_segments']:
        segments.append({
            "start_time": segment['start_time_seconds'],
            "end_time": segment['end_time_seconds'],
            "formatted_time": segment['timestamp_formatted'],
            "confidence": segment['confidence'],
            "duration": segment['duration_seconds']
        })

    return {
        "total_fake_segments": len(segments),
        "video_duration": json_result['video_info']['duration_seconds'],
        "fake_segments": segments
    }

# Example usage
if __name__ == "__main__":
    # Mock result (what you'd get from inference.analyze_video())
    mock_result = {
        "video_info": {
            "total_frames": 1800,
            "processed_frames": 1440,
            "fps": 30.0,
            "duration_seconds": 60.0,
            "overall_confidence": 0.78
        },
        "fake_segments": [
            {
                "segment_id": 1,
                "start_frame": 300,
                "end_frame": 600,
                "start_time_seconds": 10.0,
                "end_time_seconds": 20.0,
                "duration_seconds": 10.0,
                "timestamp_formatted": "0:10 - 0:20",
                "confidence": 0.89
            },
            {
                "segment_id": 2,
                "start_frame": 1200,
                "end_frame": 1500,
                "start_time_seconds": 40.0,
                "end_time_seconds": 50.0,
                "duration_seconds": 10.0,
                "timestamp_formatted": "0:40 - 0:50",
                "confidence": 0.94
            }
        ]
    }

    # Extract just the timestamps
    timestamps_only = extract_fake_timestamps(mock_result)

    print("🎯 FAKE SEGMENTS WITH TIMESTAMPS:")
    print(json.dumps(timestamps_only, indent=2))

    print("\n" + "="*50)
    print("USAGE IN YOUR CODE:")
    print("="*50)
    print("""
# Get the analysis result
result = inference.analyze_video(video_bytes)

# Extract just the fake segments
from extract_timestamps import extract_fake_timestamps
timestamps = extract_fake_timestamps(result)

# Use the data
for segment in timestamps['fake_segments']:
    print(f"Fake from {segment['start_time']}s to {segment['end_time']}s "
          f"(confidence: {segment['confidence']:.1%})")
    """)