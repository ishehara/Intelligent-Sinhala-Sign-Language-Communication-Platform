"""
Preprocess videos with MediaPipe - extract and cache features with progress bars.
Run this BEFORE training to extract all features at once.
"""

import sys
from pathlib import Path
import argparse
from tqdm import tqdm
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing_mediapipe import MediaPipeFeatureExtractor, create_dataset_splits

def preprocess_all_videos(dataset_root: str, cache_dir: str, max_frames: int = 60):
    """Preprocess all videos and cache MediaPipe features."""
    
    print("\n" + "="*60)
    print("MediaPipe Feature Extraction")
    print("="*60)
    
    # Initialize MediaPipe
    print("\n[1/4] Initializing MediaPipe...")
    extractor = MediaPipeFeatureExtractor(max_frames=max_frames, use_hands=True, use_face=True, use_pose=False)
    print(f"✓ Feature dimension: {extractor.get_feature_dim()} (hands: 126 + face: 1456 = 1582)")
    
    # Get dataset splits
    print("\n[2/4] Loading dataset...")
    splits, label_map = create_dataset_splits(dataset_root, max_frames=max_frames)
    
    total_videos = len(splits['train']) + len(splits['val']) + len(splits['test'])
    print(f"✓ Found {len(label_map)} classes, {total_videos} videos")
    print(f"  - Train: {len(splits['train'])}")
    print(f"  - Val: {len(splits['val'])}")
    print(f"  - Test: {len(splits['test'])}")
    
    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Cache directory: {cache_path}")
    
    # Process each split
    for split_name, samples in [('train', splits['train']), ('val', splits['val']), ('test', splits['test'])]:
        print(f"\n[3/4] Processing {split_name} set ({len(samples)} videos)...")
        
        processed = 0
        cached = 0
        failed = 0
        
        with tqdm(total=len(samples), desc=f"  {split_name}", unit="video") as pbar:
            for video_path, label_idx in samples:
                # Check if already cached
                video_name = Path(video_path).stem
                video_parent = Path(video_path).parent.name
                cache_file = cache_path / f"{video_parent}_{video_name}.pkl"
                
                if cache_file.exists():
                    cached += 1
                    pbar.update(1)
                    continue
                
                # Extract features
                features = extractor.process_video(str(video_path))
                
                if features is not None:
                    # Save to cache
                    try:
                        with open(cache_file, 'wb') as f:
                            pickle.dump(features, f)
                        processed += 1
                    except Exception as e:
                        print(f"\n  ✗ Failed to save cache for {video_path}: {e}")
                        failed += 1
                else:
                    failed += 1
                
                pbar.update(1)
        
        print(f"  ✓ Processed: {processed}, Cached: {cached}, Failed: {failed}")
    
    print("\n" + "="*60)
    print("✓ Preprocessing Complete!")
    print("="*60)
    print(f"\nCached features saved to: {cache_path}")
    print(f"\nNow you can train with:")
    print(f"  python train_mediapipe.py --dataset_root \"{dataset_root}\" --device cuda")
    print("  (no --preprocess flag needed - will use cached features)")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess videos with MediaPipe')
    parser.add_argument('--dataset_root', type=str, 
                       default='../../../datasets/signVideo_subset50',
                       help='Path to dataset')
    parser.add_argument('--cache_dir', type=str,
                       default='../data/processed/mediapipe',
                       help='Cache directory')
    parser.add_argument('--max_frames', type=int, default=60,
                       help='Maximum frames per video')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent.parent.parent
    
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = workspace_root / dataset_root
    
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = script_dir.parent / args.cache_dir
    
    preprocess_all_videos(str(dataset_root), str(cache_dir), args.max_frames)
