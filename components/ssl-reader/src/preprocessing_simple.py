"""
Simple video preprocessing using basic CV features (no MediaPipe).
Falls back to simple frame-based features for training.

Developer: IT22304674 – Liyanage M.L.I.S.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json

try:
    from augmentation import VideoAugmentation
    HAS_AUGMENTATION = True
except ImportError:
    HAS_AUGMENTATION = False
    logger.warning("Augmentation module not found, augmentation disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoFeatureExtractor:
    """Extracts simple CNN-based features from sign language videos."""
    
    def __init__(
        self,
        max_frames: int = 60,
        frame_size: Tuple[int, int] = (224, 224),
        use_augmentation: bool = False
    ):
        """
        Initialize simple feature extractor.
        
        Args:
            max_frames: Maximum number of frames to extract per video
            frame_size: Resize dimensions for each frame
            use_augmentation: Whether to apply data augmentation
        """
        self.max_frames = max_frames
        self.frame_size = frame_size
        self.use_augmentation = use_augmentation and HAS_AUGMENTATION
        if use_augmentation and not HAS_AUGMENTATION:
            logger.warning("Augmentation requested but not available")
        
    def extract_frame_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract features from a single frame.
        Uses simple spatial features (HOG-like + color histograms).
        
        Args:
            frame: BGR frame from video
            
        Returns:
            Feature vector for the frame
        """
        # Resize frame
        frame_resized = cv2.resize(frame, self.frame_size)
        
        # Convert to different color spaces
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
        
        # Extract gradient features (simple version of HOG)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Create histogram of gradients (4x4 cells, 8 orientation bins)
        hog_features = self._compute_hog(magnitude, direction, cell_size=56, bins=8)
        
        # Color histogram features
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
        
        # Normalize histograms
        hist_h = hist_h / (hist_h.sum() + 1e-7)
        hist_s = hist_s / (hist_s.sum() + 1e-7)
        
        # Spatial statistics
        mean_intensity = gray.mean()
        std_intensity = gray.std()
        
        # Combine all features
        features = np.concatenate([
            hog_features,
            hist_h,
            hist_s,
            [mean_intensity, std_intensity]
        ])
        
        return features
    
    def _compute_hog(self, magnitude, direction, cell_size=56, bins=8):
        """Simple HOG computation."""
        h, w = magnitude.shape
        n_cells_y = h // cell_size
        n_cells_x = w // cell_size
        
        hog = []
        for i in range(n_cells_y):
            for j in range(n_cells_x):
                cell_mag = magnitude[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
                cell_dir = direction[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
                
                # Compute histogram for this cell
                hist, _ = np.histogram(cell_dir, bins=bins, range=(-np.pi, np.pi), weights=cell_mag)
                hog.extend(hist)
        
        hog = np.array(hog)
        # Normalize
        hog = hog / (np.linalg.norm(hog) + 1e-7)
        return hog
    
    def process_video(self, video_path: str) -> Optional[np.ndarray]:
        """
        Process a single video file and extract features.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            numpy array of shape (max_frames, feature_dim) or None if failed
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return None
            
            frames_data = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame indices to sample uniformly
            if total_frames <= self.max_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames-1, self.max_frames, dtype=int)
            
            current_frame = 0
            next_sample_idx = 0
            
            while cap.isOpened() and next_sample_idx < len(frame_indices):
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Check if this is a frame we want to sample
                if current_frame == frame_indices[next_sample_idx]:
                    # Apply augmentation if enabled
                    if self.use_augmentation:
                        frame = VideoAugmentation.augment_frame(frame, training=True)
                    
                    # Extract features from this frame
                    features = self.extract_frame_features(frame)
                    frames_data.append(features)
                    next_sample_idx += 1
                
                current_frame += 1
            
            cap.release()
            
            if len(frames_data) == 0:
                logger.warning(f"No frames extracted from {video_path}")
                return None
            
            # Convert to numpy array
            features_array = np.array(frames_data)
            
            # Pad if necessary
            if len(features_array) < self.max_frames:
                padding = np.zeros((self.max_frames - len(features_array), features_array.shape[1]))
                features_array = np.vstack([features_array, padding])
            
            return features_array
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            return None
    
    def get_feature_dim(self) -> int:
        """Return the dimension of extracted features per frame."""
        # HOG: 4x4 cells * 8 bins = 128
        # H histogram: 32
        # S histogram: 32
        # Stats: 2
        # Total: 194
        return 194


def create_dataset_splits(
    dataset_root: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_frames: int = 60
) -> Tuple[Dict[str, List], Dict[str, int]]:
    """
    Create train/val/test splits from the video dataset.
    
    Args:
        dataset_root: Path to the dataset root containing category folders
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        max_frames: Maximum frames to process per video
        
    Returns:
        Tuple of (splits dict, label_to_idx mapping)
    """
    dataset_path = Path(dataset_root)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")
    
    # Collect all video files organized by category
    categories = {}
    label_to_idx = {}
    current_label = 0
    
    logger.info(f"Scanning dataset at: {dataset_path}")
    
    # Iterate through category directories
    for category_dir in dataset_path.iterdir():
        if not category_dir.is_dir():
            continue
        
        category_name = category_dir.name
        logger.info(f"Processing category: {category_name}")
        
        # Iterate through subcategories (e.g., "Hello", "Good morning")
        for sign_dir in category_dir.iterdir():
            if not sign_dir.is_dir():
                continue
            
            sign_name = sign_dir.name
            full_label = f"{category_name}/{sign_name}"
            
            # Collect video files
            video_files = list(sign_dir.glob("*.mp4")) + list(sign_dir.glob("*.avi"))
            
            if len(video_files) > 0:
                if full_label not in label_to_idx:
                    label_to_idx[full_label] = current_label
                    current_label += 1
                    categories[full_label] = []
                
                categories[full_label].extend([str(v) for v in video_files])
    
    logger.info(f"Found {len(categories)} sign classes with {sum(len(v) for v in categories.values())} total videos")
    
    # Create splits
    splits = {'train': [], 'val': [], 'test': []}
    
    for label, video_paths in categories.items():
        n_videos = len(video_paths)
        n_train = int(n_videos * train_ratio)
        n_val = int(n_videos * val_ratio)
        
        # Shuffle videos
        np.random.shuffle(video_paths)
        
        # Split
        train_vids = video_paths[:n_train]
        val_vids = video_paths[n_train:n_train+n_val]
        test_vids = video_paths[n_train+n_val:]
        
        # Add to splits with labels
        label_idx = label_to_idx[label]
        splits['train'].extend([(v, label_idx) for v in train_vids])
        splits['val'].extend([(v, label_idx) for v in val_vids])
        splits['test'].extend([(v, label_idx) for v in test_vids])
    
    # Shuffle splits
    for split in splits.values():
        np.random.shuffle(split)
    
    logger.info(f"Dataset splits - Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    
    # Save label mapping
    label_mapping_path = dataset_path.parent / "label_mapping.json"
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    with open(label_mapping_path, 'w') as f:
        json.dump({'label_to_idx': label_to_idx, 'idx_to_label': idx_to_label}, f, indent=2)
    logger.info(f"Saved label mapping to {label_mapping_path}")
    
    return splits, label_to_idx
