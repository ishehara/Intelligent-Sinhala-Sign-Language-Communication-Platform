"""
Pre-extract and cache all MediaPipe features to avoid DataLoader deadlock.
Run this BEFORE training to extract all 2623 videos.

Developer: IT22304674 – Liyanage M.L.I.S.
"""

import sys
import logging
import pickle
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from preprocessing_mediapipe import MediaPipeFeatureExtractor, create_dataset_splits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Configuration (match your training settings)
    dataset_root = r"G:\research\Intelligent-Sinhala-Sign-Language-Communication-Platform\datasets\signVideo"
    cache_dir = Path("data/processed/mediapipe_features_457")
    cache_dir.mkdir(parents=True, exist_ok=True)
    max_frames = 60

    logger.info("=" * 80)
    logger.info("FEATURE PRE-EXTRACTION")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset_root}")
    logger.info(f"Cache: {cache_dir}")
    logger.info(f"Features: 457 dims (Hands 126 + Face 232 + Pose 99)")
    logger.info("=" * 80)

    # Initialize feature extractor
    logger.info("Initializing MediaPipe (hands + filtered face + blendshapes + pose)...")
    feature_extractor = MediaPipeFeatureExtractor(
        max_frames=max_frames,
        use_hands=True,
        use_pose=True,           # Re-enabled: body anchor for hand positions
        use_face=True,
        use_filtered_face=True,
        use_blendshapes=True     # Re-enabled: 52 dims of facial muscle movement
    )
    logger.info(f"Feature dimension: {feature_extractor.get_feature_dim()}")

    # Create dataset splits
    logger.info("Creating dataset splits...")
    splits, label_map = create_dataset_splits(dataset_root, max_frames=max_frames)

    # Process all splits
    all_samples = []
    for split_name in ['train', 'val', 'test']:
        for video_path, label in splits[split_name]:
            all_samples.append(video_path)

    logger.info(f"\nTotal videos to process: {len(all_samples)}")

    success = 0
    failed = 0

    for video_path in tqdm(all_samples, desc="Extracting features"):
        # Compute cache filename
        p = Path(video_path)
        video_parent = p.parent.name
        video_name = p.stem
        cache_name = f"{video_parent}_{video_name}.pkl"
        cache_path = cache_dir / cache_name

        # Skip if already cached
        if cache_path.exists():
            success += 1
            continue

        try:
            features = feature_extractor.process_video(video_path)
            if features is None:
                import numpy as np
                features = np.zeros((max_frames, feature_extractor.get_feature_dim()))
                failed += 1
            else:
                success += 1

            with open(cache_path, 'wb') as f:
                pickle.dump(features, f)
        except Exception as e:
            logger.error(f"Failed: {video_path} - {e}")
            failed += 1

    logger.info("\n" + "=" * 80)
    logger.info(f"✓ EXTRACTION COMPLETE: {success} success, {failed} failed")
    logger.info(f"Cache saved to: {cache_dir}")
    logger.info("Now run training - it will use cached features (fast!)")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

