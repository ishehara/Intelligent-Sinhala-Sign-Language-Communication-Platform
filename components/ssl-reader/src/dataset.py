"""
PyTorch Dataset and DataLoader for Sinhala Sign Language Recognition.

Developer: IT22304674 – Liyanage M.L.I.S.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

from preprocessing_simple import VideoFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SinhalaSignLanguageDataset(Dataset):
    """PyTorch Dataset for Sinhala Sign Language videos."""
    
    def __init__(
        self,
        samples: List[Tuple[str, str]],
        label_to_idx: Dict[str, int],
        feature_extractor: VideoFeatureExtractor,
        cache_dir: Optional[str] = None,
        use_cache: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            samples: List of (video_path, label) tuples
            label_to_idx: Dictionary mapping label names to indices
            feature_extractor: VideoFeatureExtractor instance
            cache_dir: Directory to cache preprocessed features
            use_cache: Whether to use cached features
        """
        self.samples = samples
        self.label_to_idx = label_to_idx
        self.feature_extractor = feature_extractor
        self.use_cache = use_cache
        
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _get_cache_path(self, video_path: str) -> Optional[Path]:
        """Get the cache file path for a video."""
        if self.cache_dir is None:
            return None
        
        # Create unique cache filename from video path
        video_name = Path(video_path).stem
        video_parent = Path(video_path).parent.name
        cache_name = f"{video_parent}_{video_name}.pkl"
        return self.cache_dir / cache_name
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, label) tensors
        """
        video_path, label_idx = self.samples[idx]
        
        # label_idx is already an integer, no need to look it up
        
        # Try to load from cache
        cache_path = self._get_cache_path(video_path)
        if self.use_cache and cache_path and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    features = pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache for {video_path}: {e}")
                features = None
        else:
            features = None
        
        # Extract features if not cached
        if features is None:
            features = self.feature_extractor.process_video(video_path)
            
            if features is None:
                # Return zeros if extraction failed
                features = np.zeros((
                    self.feature_extractor.max_frames,
                    self.feature_extractor.get_feature_dim()
                ))
            
            # Save to cache
            if self.use_cache and cache_path:
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(features, f)
                except Exception as e:
                    logger.warning(f"Failed to save cache for {video_path}: {e}")
        
        # Convert to tensors
        features_tensor = torch.FloatTensor(features)
        label_tensor = torch.LongTensor([label_idx])
        
        return features_tensor, label_tensor.squeeze()
    
    def preprocess_all(self, num_workers: int = 4):
        """
        Preprocess all videos and cache them.
        
        Args:
            num_workers: Number of parallel workers (not implemented yet, sequential)
        """
        logger.info(f"Preprocessing {len(self.samples)} videos...")
        
        for idx in tqdm(range(len(self.samples)), desc="Processing videos"):
            _ = self.__getitem__(idx)


def create_data_loaders(
    dataset_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    max_frames: int = 60,
    cache_dir: Optional[str] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    preprocess: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], int]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        dataset_root: Root directory of the video dataset
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        max_frames: Maximum frames per video
        cache_dir: Directory to cache preprocessed features
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        preprocess: Whether to preprocess all videos before training
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, label_to_idx, num_classes)
    """
    from preprocessing_simple import create_dataset_splits
    
    # Create splits
    splits, label_to_idx = create_dataset_splits(
        dataset_root,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        max_frames=max_frames
    )
    
    num_classes = len(label_to_idx)
    
    # Create feature extractors (with augmentation for training only)
    train_feature_extractor = VideoFeatureExtractor(max_frames=max_frames, use_augmentation=True)
    val_test_feature_extractor = VideoFeatureExtractor(max_frames=max_frames, use_augmentation=False)
    
    # Create datasets
    train_dataset = SinhalaSignLanguageDataset(
        samples=splits['train'],
        label_to_idx=label_to_idx,
        feature_extractor=train_feature_extractor,
        cache_dir=cache_dir,
        use_cache=(cache_dir is not None)
    )
    
    val_dataset = SinhalaSignLanguageDataset(
        samples=splits['val'],
        label_to_idx=label_to_idx,
        feature_extractor=val_test_feature_extractor,
        cache_dir=cache_dir,
        use_cache=(cache_dir is not None)
    )
    
    test_dataset = SinhalaSignLanguageDataset(
        samples=splits['test'],
        label_to_idx=label_to_idx,
        feature_extractor=val_test_feature_extractor,
        cache_dir=cache_dir,
        use_cache=(cache_dir is not None)
    )
    
    # Preprocess all videos if requested
    if preprocess and cache_dir:
        logger.info("Preprocessing training set...")
        train_dataset.preprocess_all()
        logger.info("Preprocessing validation set...")
        val_dataset.preprocess_all()
        logger.info("Preprocessing test set...")
        test_dataset.preprocess_all()
    
    # Create data loaders (num_workers=0 for Windows compatibility)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders:")
    logger.info(f"  Train: {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_loader)} batches")
    logger.info(f"  Num classes: {num_classes}")
    
    return train_loader, val_loader, test_loader, label_to_idx, num_classes


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    dataset_root = str(project_root / 'datasets' / 'signVideo')
    cache_dir = str(Path(__file__).parent.parent / 'data' / 'processed')
    
    train_loader, val_loader, test_loader, label_to_idx, num_classes = create_data_loaders(
        dataset_root=dataset_root,
        batch_size=8,
        num_workers=2,
        max_frames=60,
        cache_dir=cache_dir,
        preprocess=False  # Set to True to preprocess all videos
    )
    
    # Test loading a batch
    for features, labels in train_loader:
        print(f"Batch features shape: {features.shape}")
        print(f"Batch labels shape: {labels.shape}")
        print(f"Number of classes: {num_classes}")
        break
