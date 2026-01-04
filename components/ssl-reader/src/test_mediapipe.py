"""
Test MediaPipe feature extraction on a single video.
Quick verification that MediaPipe setup works.
"""

import cv2
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing_mediapipe import MediaPipeFeatureExtractor

def main():
    print("Testing MediaPipe feature extraction...")
    
    # Find a sample video
    dataset_root = Path(__file__).parent.parent.parent.parent / 'datasets' / 'signVideo_subset50'
    
    # Find first video file
    video_files = list(dataset_root.rglob('*.mp4'))
    if not video_files:
        video_files = list(dataset_root.rglob('*.avi'))
    
    if not video_files:
        print("No video files found in dataset!")
        return
    
    test_video = video_files[0]
    print(f"Testing on: {test_video}")
    
    # Initialize extractor
    print("\nInitializing MediaPipe (downloading models if needed)...")
    extractor = MediaPipeFeatureExtractor(
        max_frames=30,
        use_hands=True,
        use_pose=False  # Pose model URL needs fixing
    )
    
    print(f"Feature dimension: {extractor.get_feature_dim()}")
    
    # Process video
    print(f"\nProcessing video...")
    features = extractor.process_video(str(test_video))
    
    if features is not None:
        print(f"✓ Success! Features shape: {features.shape}")
        print(f"  Expected: (30, {extractor.get_feature_dim()})")
        print(f"  Min value: {features.min():.4f}")
        print(f"  Max value: {features.max():.4f}")
        print(f"  Mean: {features.mean():.4f}")
        print(f"  Non-zero values: {(features != 0).sum()} / {features.size}")
    else:
        print("✗ Failed to extract features")

if __name__ == '__main__':
    main()
