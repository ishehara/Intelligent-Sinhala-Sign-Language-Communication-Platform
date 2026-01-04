"""
Test facial expression extraction with MediaPipe.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing_mediapipe import MediaPipeFeatureExtractor

def test_face_extraction():
    """Test facial landmark and blendshape extraction on a sample video."""
    
    print("="*60)
    print("Testing Facial Expression Extraction")
    print("="*60)
    
    # Initialize with face enabled
    print("\n[1/3] Initializing MediaPipe with facial expressions...")
    extractor = MediaPipeFeatureExtractor(
        max_frames=30,
        use_hands=True,
        use_face=True,
        use_pose=False
    )
    
    feature_dim = extractor.get_feature_dim()
    print(f"✓ Feature dimension: {feature_dim}")
    print(f"  - Hand landmarks: 126 (2 hands × 21 landmarks × 3 coords)")
    print(f"  - Face landmarks: 1404 (468 landmarks × 3 coords)")
    print(f"  - Face blendshapes: 52 (emotion-related expressions)")
    print(f"  - Total: {feature_dim}")
    
    # Find a test video
    print("\n[2/3] Looking for test video...")
    dataset_root = Path(__file__).parent.parent / "datasets" / "signVideo"
    
    if not dataset_root.exists():
        print(f"✗ Dataset not found at {dataset_root}")
        return
    
    # Find first video file
    video_file = None
    for category_dir in dataset_root.iterdir():
        if category_dir.is_dir():
            for class_dir in category_dir.iterdir():
                if class_dir.is_dir():
                    for video in class_dir.glob("*.mp4"):
                        video_file = video
                        break
                if video_file:
                    break
        if video_file:
            break
    
    if not video_file:
        print("✗ No video files found")
        return
    
    print(f"✓ Test video: {video_file.name}")
    print(f"  Category: {video_file.parent.parent.name}/{video_file.parent.name}")
    
    # Process video
    print("\n[3/3] Extracting features...")
    features = extractor.process_video(str(video_file))
    
    if features is None:
        print("✗ Failed to extract features")
        return
    
    print(f"✓ Features extracted successfully!")
    print(f"  Shape: {features.shape} (frames × features)")
    
    # Analyze facial blendshapes (last 52 features)
    blendshapes = features[:, -52:]  # Last 52 columns are blendshapes
    
    print(f"\n📊 Blendshape Analysis (Emotion Indicators):")
    print(f"  Blendshape range: [{blendshapes.min():.3f}, {blendshapes.max():.3f}]")
    print(f"  Mean activity: {blendshapes.mean():.3f}")
    print(f"  Active blendshapes (>0.1): {(blendshapes > 0.1).sum()}")
    
    # Check if facial data was captured
    face_landmarks = features[:, 126:126+1404]  # Face landmarks (after hands)
    face_activity = (face_landmarks != 0).any(axis=1).sum()
    
    print(f"\n📸 Face Detection:")
    print(f"  Frames with face detected: {face_activity}/{features.shape[0]}")
    print(f"  Detection rate: {face_activity/features.shape[0]*100:.1f}%")
    
    # Check hand data
    hand_landmarks = features[:, :126]  # First 126 features
    hand_activity = (hand_landmarks != 0).any(axis=1).sum()
    
    print(f"\n👋 Hand Detection:")
    print(f"  Frames with hands detected: {hand_activity}/{features.shape[0]}")
    print(f"  Detection rate: {hand_activity/features.shape[0]*100:.1f}%")
    
    print("\n" + "="*60)
    print("✓ Facial expression extraction is working!")
    print("="*60)
    print("\nYou can now run the full preprocessing:")
    print("  python src/preprocess_mediapipe.py")
    print("\nThis will extract both hand gestures and facial expressions")
    print("from all videos in your dataset.")

if __name__ == "__main__":
    test_face_extraction()
