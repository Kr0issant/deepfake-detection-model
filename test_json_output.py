#!/usr/bin/env python3
"""
Test script to show the JSON output format for video analysis
"""
import sys
sys.path.insert(0, 'src/modules')

import json
from inference import Inference

# Mock video analysis result for demonstration
def demo_video_analysis():
    """Show what the JSON output looks like"""

    # This is what the actual analysis would return
    mock_result = {
        "video_info": {
            "total_frames": 1500,
            "processed_frames": 1200,
            "fps": 30.0,
            "duration_seconds": 50.0,
            "overall_confidence": 0.75
        },
        "fake_segments": [
            {
                "segment_id": 1,
                "start_frame": 150,
                "end_frame": 450,
                "start_time_seconds": 5.0,
                "end_time_seconds": 15.0,
                "duration_seconds": 10.0,
                "timestamp_formatted": "0:05 - 0:15",
                "confidence": 0.85
            },
            {
                "segment_id": 2,
                "start_frame": 900,
                "end_frame": 1200,
                "start_time_seconds": 30.0,
                "end_time_seconds": 40.0,
                "duration_seconds": 10.0,
                "timestamp_formatted": "0:30 - 0:40",
                "confidence": 0.92
            }
        ]
    }

    print("=" * 60)
    print("VIDEO ANALYSIS JSON OUTPUT FORMAT")
    print("=" * 60)
    print(json.dumps(mock_result, indent=2))

    print("\n" + "=" * 60)
    print("KEY FEATURES:")
    print("=" * 60)
    print("• video_info: Basic video metadata")
    print("• fake_segments: List of detected fake segments")
    print("• Each segment includes:")
    print("  - Frame numbers (start_frame, end_frame)")
    print("  - Time in seconds (start_time_seconds, end_time_seconds)")
    print("  - Formatted timestamp (MM:SS format)")
    print("  - Duration in seconds")
    print("  - Confidence score (0.0-1.0)")
    print("• overall_confidence: Average confidence across all frames")

if __name__ == "__main__":
    demo_video_analysis()