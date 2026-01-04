"""
Preprocess all videos with MediaPipe and cache features.
Run this BEFORE training to extract features with progress tracking.

Usage:
    python preprocess_mediapipe_batch.py --dataset_root datasets/signVideo_subset50

This will extract MediaPipe hand landmarks from all videos and cache them,
making subsequent training runs much faster.
"""

import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import pickle

from preprocessing_mediapipe import MediaPipeFeatureExtractor, create_dataset_splits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Preprocess videos with MediaPipe')
    parser.add_argument('--dataset_root', type=str, required=True,
                       help='Path to dataset root')
    parser.add_argument('--max_frames', type=int, default=60,
                       help='Maximum frames per video')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Cache directory (default: data/processed/mediapipe)')
    
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    workspace_root = project_root.parent.parent
    dataset_root = Path(args.dataset_root)
    
    if not dataset_root.is_absolute():
        dataset_root = workspace_root / dataset_root
    
    if args.cache_dir is None:
        cache_dir = project_root / 'data' / 'processed' / 'mediapipe'
    else:
        cache_dir = Path(args.cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Dataset: {dataset_root}")
    logger.info(f"Cache: {cache_dir}")
    
    # Initialize MediaPipe
    logger.info("Initializing MediaPipe...")
    extractor = MediaPipeFeatureExtractor(max_frames=args.max_frames)
    logger.info(f"Feature dimension: {extractor.get_feature_dim()}")
    
    # Get dataset splits
    logger.info("Creating dataset splits...")
    splits, label_map = create_dataset_splits(str(dataset_root), max_frames=args.max_frames)
    
    # Combine all samples
    all_samples = splits['train'] + splits['val'] + splits['test']
    logger.info(f"Total videos to process: {len(all_samples)}")
    
    # Process with progress bar
    processed = 0
    cached = 0
    failed = 0
    
    with tqdm(total=len(all_samples), desc="Processing videos", unit="video") as pbar:
        for video_path, label_idx in all_samples:
            # Create cache path
            video_name = Path(video_path).stem
            video_parent = Path(video_path).parent.name
            cache_name = f"{video_parent}_{video_name}.pkl"
            cache_path = cache_dir / cache_name
            
            # Skip if already cached
            if cache_path.exists():
                cached += 1
                pbar.update(1)
                pbar.set_postfix({"processed": processed, "cached": cached, "failed": failed})
                continue
            
            # Extract features
            try:
                features = extractor.process_video(video_path)
                
                if features is not None:
                    # Save to cache
                    with open(cache_path, 'wb') as f:
                        pickle.dump(features, f)
                    processed += 1
                else:
                    failed += 1
                    logger.warning(f"Failed to extract features: {video_path}")
            
            except Exception as e:
                failed += 1
                logger.error(f"Error processing {video_path}: {e}")
            
            pbar.update(1)
            pbar.set_postfix({"processed": processed, "cached": cached, "failed": failed})
    
    logger.info("\n" + "="*50)
    logger.info(f"Preprocessing complete!")
    logger.info(f"  Newly processed: {processed}")
    logger.info(f"  Already cached: {cached}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Total: {len(all_samples)}")
    logger.info("="*50)
    
    if failed > 0:
        logger.warning(f"\n{failed} videos failed to process. Training will use zero features for these.")
    
    logger.info(f"\nFeatures cached to: {cache_dir}")
    logger.info("\nYou can now run training WITHOUT --preprocess flag:")
    logger.info("  python train_mediapipe.py --dataset_root \"datasets/signVideo_subset50\" --device cuda")


if __name__ == '__main__':
    main()
