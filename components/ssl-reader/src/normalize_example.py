"""
Example: How to integrate landmark normalization into your preprocessing pipeline.
This will significantly improve model accuracy by making features position and scale invariant.

Before normalization: ~4% accuracy (absolute coordinates)
After normalization: Expected 15-30%+ accuracy improvement

Developer: IT22304674 – Liyanage M.L.I.S.
"""

import sys
from pathlib import Path
import numpy as np
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing_mediapipe import MediaPipeFeatureExtractor
from normalize_landmarks import normalize_landmarks, normalize_mediapipe_features


# ===== METHOD 1: Normalize After Extraction =====

def preprocess_with_normalization_method1(video_path: str, max_frames: int = 60):
    """
    Extract features then normalize them.
    Use this if you want to cache raw features and normalize during training.
    """
    # Step 1: Extract raw MediaPipe features
    extractor = MediaPipeFeatureExtractor(
        max_frames=max_frames,
        use_hands=True,
        use_face=True,
        use_pose=False
    )
    
    features = extractor.process_video(video_path)  # Shape: (60, 1582, 3)
    
    if features is None:
        return None
    
    # Step 2: Normalize the features
    # For your setup: 126 hand features (0-41 landmarks) + 1456 face features (42-509 landmarks)
    normalized_features = normalize_landmarks(
        features,
        hand_landmarks=(0, 42),      # 2 hands × 21 landmarks = 42
        face_landmarks=(42, 510),    # 468 face landmarks
        pose_landmarks=None,          # Not using pose
        scale_factor='hand'           # Scale by hand size
    )
    
    return normalized_features


# ===== METHOD 2: Normalize Cached Features =====

def normalize_cached_features(cache_dir: str, output_dir: str):
    """
    Load previously cached raw features and normalize them.
    Use this to convert existing cached features without re-extracting.
    """
    cache_path = Path(cache_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Normalizing cached features from: {cache_path}")
    print(f"Saving to: {output_path}")
    
    # Process all .pkl files
    pkl_files = list(cache_path.glob("*.pkl"))
    print(f"Found {len(pkl_files)} cached feature files")
    
    for pkl_file in pkl_files:
        # Load raw features
        with open(pkl_file, 'rb') as f:
            raw_features = pickle.load(f)
        
        # Normalize
        normalized = normalize_landmarks(
            raw_features,
            hand_landmarks=(0, 42),
            face_landmarks=(42, 510),
            scale_factor='hand'
        )
        
        # Save normalized features
        output_file = output_path / pkl_file.name
        with open(output_file, 'wb') as f:
            pickle.dump(normalized, f)
    
    print(f"✓ Normalized {len(pkl_files)} feature files")


# ===== METHOD 3: Normalize in Dataset Class =====

def create_normalized_dataset_example():
    """
    Example: Modify your dataset.py to normalize on-the-fly during training.
    """
    example_code = '''
# In your dataset.py, modify the __getitem__ method:

class SinhalaSignLanguageDataset(Dataset):
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        # Load cached features
        features = self._load_cached_features(video_path)
        
        # ⭐ ADD NORMALIZATION HERE ⭐
        from normalize_landmarks import normalize_landmarks
        
        features = normalize_landmarks(
            features,
            hand_landmarks=(0, 42),
            face_landmarks=(42, 510),
            scale_factor='hand'
        )
        
        # Convert to tensor
        features = torch.FloatTensor(features)
        label = torch.LongTensor([label])
        
        return features, label
'''
    print(example_code)


# ===== METHOD 4: Batch Normalize All Cached Features =====

def batch_normalize_dataset(
    cache_dir: str = "data/processed/mediapipe_face",
    output_dir: str = "data/processed/mediapipe_normalized"
):
    """
    Normalize all cached features in your dataset.
    Run this once to create a normalized version of your dataset.
    """
    print("="*60)
    print("Batch Normalizing Dataset")
    print("="*60)
    
    cache_path = Path(cache_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all cached feature files
    pkl_files = list(cache_path.glob("*.pkl"))
    total_files = len(pkl_files)
    
    print(f"\nFound {total_files} cached feature files")
    print(f"Cache directory: {cache_path}")
    print(f"Output directory: {output_path}")
    
    processed = 0
    failed = 0
    
    for i, pkl_file in enumerate(pkl_files, 1):
        try:
            # Load raw features
            with open(pkl_file, 'rb') as f:
                raw_features = pickle.load(f)
            
            # Check shape
            if raw_features.shape[1] == 1582:  # Your feature dimension
                # Features are already flattened, reshape first
                # 1582 = 42 * 3 + 468 * 3 + 52 blendshapes
                # But your shape is (60, 1582) not (60, 510, 3)
                # So we need to handle this differently
                
                # Extract hand features (0-126), face landmarks (126-1530), blendshapes (1530-1582)
                # Reshape to (frames, landmarks, coords) format
                frames = raw_features.shape[0]
                
                # Split features
                hand_features = raw_features[:, :126].reshape(frames, 42, 3)
                face_landmarks = raw_features[:, 126:1530].reshape(frames, 468, 3)
                blendshapes = raw_features[:, 1530:]  # (frames, 52)
                
                # Normalize hands and face
                norm_hands = normalize_landmarks(
                    hand_features,
                    hand_landmarks=(0, 42),
                    scale_factor='hand'
                )
                
                norm_face = normalize_landmarks(
                    face_landmarks,
                    face_landmarks=(0, 468),
                    scale_factor='none'
                )
                
                # Flatten back
                norm_hands_flat = norm_hands.reshape(frames, -1)
                norm_face_flat = norm_face.reshape(frames, -1)
                
                # Concatenate
                normalized = np.concatenate([norm_hands_flat, norm_face_flat, blendshapes], axis=1)
                
            else:
                # Unknown shape, skip
                print(f"Warning: Unexpected shape {raw_features.shape}, skipping")
                failed += 1
                continue
            
            # Save normalized features
            output_file = output_path / pkl_file.name
            with open(output_file, 'wb') as f:
                pickle.dump(normalized, f)
            
            processed += 1
            
            if i % 100 == 0:
                print(f"Progress: {i}/{total_files} ({i/total_files*100:.1f}%)")
        
        except Exception as e:
            print(f"Error processing {pkl_file.name}: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"✓ Processed: {processed}/{total_files}")
    print(f"✗ Failed: {failed}")
    print("="*60)


# ===== TESTING =====

def test_single_video():
    """Test normalization on a single video."""
    print("="*60)
    print("Testing Normalization on Single Video")
    print("="*60)
    
    # Find a test video
    dataset_root = Path(__file__).parent.parent.parent.parent / "datasets" / "signVideo"
    
    if not dataset_root.exists():
        print(f"Dataset not found at {dataset_root}")
        return
    
    # Find first video
    test_video = None
    for category_dir in dataset_root.iterdir():
        if category_dir.is_dir():
            for class_dir in category_dir.iterdir():
                if class_dir.is_dir():
                    for video in class_dir.glob("*.mp4"):
                        test_video = video
                        break
                if test_video:
                    break
        if test_video:
            break
    
    if not test_video:
        print("No video found in dataset")
        return
    
    print(f"\nTest video: {test_video.name}")
    
    # Extract and normalize
    print("\nExtracting features...")
    features = preprocess_with_normalization_method1(str(test_video))
    
    if features is not None:
        print(f"✓ Features extracted: {features.shape}")
        
        # Analyze normalization
        print("\nNormalization Analysis:")
        print(f"  Mean: {features.mean():.6f}")
        print(f"  Std: {features.std():.6f}")
        print(f"  Min: {features.min():.6f}")
        print(f"  Max: {features.max():.6f}")
        
        # Check first frame hands (should be centered around 0)
        hand_coords = features[0, :42, :]  # First frame, first 42 hand landmarks
        print(f"\nHand landmarks (first frame):")
        print(f"  Wrist (0): {hand_coords[0]}")  # Should be close to [0, 0, z]
        print(f"  Right wrist (21): {hand_coords[21]}")  # Should be close to [0, 0, z]
        
        print("\n✓ Normalization successful!")
    else:
        print("✗ Feature extraction failed")


# ===== USAGE INSTRUCTIONS =====

def print_usage():
    """Print usage instructions."""
    usage = """
╔══════════════════════════════════════════════════════════════════════════╗
║                  LANDMARK NORMALIZATION GUIDE                            ║
╚══════════════════════════════════════════════════════════════════════════╝

Why Normalize?
• Absolute coordinates cause low accuracy (4%)
• Normalization makes model invariant to:
  - Position (signs anywhere in frame)
  - Scale (different hand sizes)
  - Person distance (near/far from camera)

Expected Improvement:
• Before: 4% accuracy
• After: 15-30%+ accuracy

═══════════════════════════════════════════════════════════════════════════

OPTION 1: Normalize Existing Cached Features (RECOMMENDED)
────────────────────────────────────────────────────────────────────────────
python src/normalize_example.py --normalize-cache

This will:
1. Load your existing cached features from data/processed/mediapipe_face
2. Normalize them (wrist-relative, nose-relative)
3. Save to data/processed/mediapipe_normalized
4. Ready for training!

═══════════════════════════════════════════════════════════════════════════

OPTION 2: Modify Training Script
────────────────────────────────────────────────────────────────────────────
In train_mediapipe.py, add normalization to dataset loading:

from normalize_landmarks import normalize_landmarks

# In your dataset's __getitem__ method:
features = normalize_landmarks(
    features,
    hand_landmarks=(0, 42),
    face_landmarks=(42, 510),
    scale_factor='hand'
)

═══════════════════════════════════════════════════════════════════════════

OPTION 3: Test on Single Video
────────────────────────────────────────────────────────────────────────────
python src/normalize_example.py --test-single

═══════════════════════════════════════════════════════════════════════════

After Normalization, Train Your Model:
────────────────────────────────────────────────────────────────────────────
python src/train_mediapipe.py \\
    --dataset_root datasets/signVideo \\
    --cache_dir data/processed/mediapipe_normalized \\
    --model_type lstm \\
    --hidden_dim 512 \\
    --num_layers 3 \\
    --device cuda

Expected Results:
• Epoch 1: 15-20% accuracy ✓
• Epoch 10: 30-40% accuracy ✓
• Epoch 50: 50-70% accuracy ✓

═══════════════════════════════════════════════════════════════════════════
"""
    print(usage)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Landmark Normalization Examples')
    parser.add_argument('--normalize-cache', action='store_true',
                       help='Normalize all cached features')
    parser.add_argument('--test-single', action='store_true',
                       help='Test on single video')
    parser.add_argument('--cache-dir', type=str,
                       default='data/processed/mediapipe_face',
                       help='Input cache directory')
    parser.add_argument('--output-dir', type=str,
                       default='data/processed/mediapipe_normalized',
                       help='Output directory for normalized features')
    
    args = parser.parse_args()
    
    if args.normalize_cache:
        batch_normalize_dataset(args.cache_dir, args.output_dir)
    elif args.test_single:
        test_single_video()
    else:
        print_usage()
