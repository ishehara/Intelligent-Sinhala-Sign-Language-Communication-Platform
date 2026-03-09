"""
Pre-extract and cache all MediaPipe features to avoid DataLoader deadlock.
Run this BEFORE training to extract all 2623 videos.

Developer: IT22304674 – Liyanage M.L.I.S.
"""

import sys
import logging
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from preprocessing_mediapipe import MediaPipeFeatureExtractor, create_dataset_splits
from dataset import SinhalaSignLanguageDataset as SignLanguageDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Configuration (match your training settings)
    dataset_root = r"G:\research\Intelligent-Sinhala-Sign-Language-Communication-Platform\datasets\signVideo"
    cache_dir = "data/processed/mediapipe_features_457"
    max_frames = 60
    
    logger.info("=" * 80)
    logger.info("FEATURE PRE-EXTRACTION")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset_root}")
    logger.info(f"Cache: {cache_dir}")
    logger.info(f"Features: 457 dims (Hands 126 + Face 180 + Blendshapes 52 + Pose 99)")
    logger.info("=" * 80)
    
    # Initialize feature extractor
    logger.info("Initializing MediaPipe (hands + filtered face + pose)...")
    feature_extractor = MediaPipeFeatureExtractor(
        max_frames=max_frames,
        use_hands=True,
        use_pose=True,
        use_face=True,
        use_filtered_face=True,  # 60 key landmarks instead of 468
        use_blendshapes=True     # +52 blendshape dims → 457 total
    )
    
    logger.info(f"Feature dimension: {feature_extractor.get_feature_dim()}")
    
    # Create dataset splits
    logger.info("Creating dataset splits...")
    splits, label_map = create_dataset_splits(dataset_root, max_frames=max_frames)
    
    # Process each split
    for split_name in ['train', 'val', 'test']:
        samples = splits[split_name]
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {split_name.upper()} split: {len(samples)} videos")
        logger.info(f"{'='*80}")
        
        # Create dataset (no augmentation for caching)
        dataset = SignLanguageDataset(
            samples=samples,
            label_to_idx=label_map,
            feature_extractor=feature_extractor,
            cache_dir=cache_dir,
            use_cache=True,
            training=False,
            augment=False
        )
        
        # Extract all features with progress bar
        for idx in tqdm(range(len(dataset)), desc=f"{split_name.capitalize()}"):
            try:
                _ = dataset[idx]
            except Exception as e:
                logger.error(f"Failed to process sample {idx}: {e}")
                continue
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ FEATURE EXTRACTION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"All features cached to: {cache_dir}")
    logger.info("Now run training - it will use cached features (fast!)")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
