"""
Video preprocessing module for Sinhala Sign Language Recognition.
Extracts multimodal features (hands, face, pose) from video files using MediaPipe.

Developer: IT22304674 – Liyanage M.L.I.S.
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoFeatureExtractor:
    """Extracts hand, face, and pose landmarks from sign language videos."""
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_frames: int = 60  # Maximum frames to process per video
    ):
        """
        Initialize MediaPipe models for holistic feature extraction.
        
        Args:
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            max_frames: Maximum number of frames to extract per video
        """
        self.max_frames = max_frames
        
        # Initialize MediaPipe Holistic (combines hands, face, pose)
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=False
        )
        
    def extract_landmarks(self, results) -> Dict[str, np.ndarray]:
        """
        Extract landmarks from MediaPipe results.
        
        Args:
            results: MediaPipe holistic results
            
        Returns:
            Dictionary containing landmark arrays for different body parts
        """
        landmarks = {}
        
        # Left hand (21 landmarks × 3 coordinates = 63 features)
        if results.left_hand_landmarks:
            landmarks['left_hand'] = np.array([
                [lm.x, lm.y, lm.z] 
                for lm in results.left_hand_landmarks.landmark
            ]).flatten()
        else:
            landmarks['left_hand'] = np.zeros(63)
        
        # Right hand (21 landmarks × 3 coordinates = 63 features)
        if results.right_hand_landmarks:
            landmarks['right_hand'] = np.array([
                [lm.x, lm.y, lm.z] 
                for lm in results.right_hand_landmarks.landmark
            ]).flatten()
        else:
            landmarks['right_hand'] = np.zeros(63)
        
        # Face (468 landmarks × 3 coordinates = 1404 features)
        # We'll use a subset for efficiency (68 key points)
        if results.face_landmarks:
            # Select key facial landmarks for emotion
            key_indices = list(range(0, 468, 7))  # Sample every 7th landmark
            landmarks['face'] = np.array([
                [results.face_landmarks.landmark[i].x,
                 results.face_landmarks.landmark[i].y,
                 results.face_landmarks.landmark[i].z]
                for i in key_indices
            ]).flatten()
        else:
            landmarks['face'] = np.zeros(len(range(0, 468, 7)) * 3)
        
        # Pose (33 landmarks × 4 coordinates (x,y,z,visibility) = 132 features)
        # We'll use upper body for sign language
        if results.pose_landmarks:
            # Upper body indices (0-16: face, shoulders, arms, hands)
            upper_body_indices = list(range(17))
            landmarks['pose'] = np.array([
                [results.pose_landmarks.landmark[i].x,
                 results.pose_landmarks.landmark[i].y,
                 results.pose_landmarks.landmark[i].z,
                 results.pose_landmarks.landmark[i].visibility]
                for i in upper_body_indices
            ]).flatten()
        else:
            landmarks['pose'] = np.zeros(17 * 4)
        
        return landmarks
    
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
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame sampling rate to get max_frames
            skip_rate = max(1, total_frames // self.max_frames)
            
            while cap.isOpened() and frame_count < self.max_frames:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Skip frames if needed
                if len(frames_data) > 0 and len(frames_data) % skip_rate != 0:
                    continue
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame with MediaPipe
                results = self.holistic.process(frame_rgb)
                
                # Extract landmarks
                landmarks = self.extract_landmarks(results)
                
                # Concatenate all features
                frame_features = np.concatenate([
                    landmarks['left_hand'],
                    landmarks['right_hand'],
                    landmarks['face'],
                    landmarks['pose']
                ])
                
                frames_data.append(frame_features)
                frame_count += 1
            
            cap.release()
            
            if len(frames_data) == 0:
                logger.warning(f"No frames extracted from {video_path}")
                return None
            
            # Pad or truncate to max_frames
            frames_array = np.array(frames_data)
            
            if len(frames_array) < self.max_frames:
                # Pad with zeros
                padding = np.zeros((self.max_frames - len(frames_array), frames_array.shape[1]))
                frames_array = np.vstack([frames_array, padding])
            else:
                # Truncate
                frames_array = frames_array[:self.max_frames]
            
            logger.info(f"Processed {video_path}: shape {frames_array.shape}")
            return frames_array
            
        except Exception as e:
            logger.error(f"Error processing {video_path}: {str(e)}")
            return None
    
    def get_feature_dim(self) -> int:
        """
        Get the total feature dimension per frame.
        
        Returns:
            Total feature dimension
        """
        # left_hand(63) + right_hand(63) + face(~201) + pose(68) ≈ 395
        return 63 + 63 + len(range(0, 468, 7)) * 3 + 17 * 4
    
    def close(self):
        """Release MediaPipe resources."""
        self.holistic.close()


def create_dataset_splits(
    dataset_root: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Create train/val/test splits from the video dataset.
    
    Args:
        dataset_root: Root directory of the dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys containing (video_path, label) tuples
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    np.random.seed(seed)
    dataset_path = Path(dataset_root)
    
    # Collect all videos with their labels
    all_samples = []
    label_to_idx = {}
    idx = 0
    
    # Iterate through all category folders
    for category_dir in dataset_path.iterdir():
        if not category_dir.is_dir():
            continue
        
        # Iterate through all sign folders within category
        for sign_dir in category_dir.iterdir():
            if not sign_dir.is_dir():
                continue
            
            label = sign_dir.name
            
            if label not in label_to_idx:
                label_to_idx[label] = idx
                idx += 1
            
            # Get all video files in this sign folder
            video_files = list(sign_dir.glob("*.mp4"))
            
            for video_file in video_files:
                all_samples.append((str(video_file), label))
    
    logger.info(f"Found {len(all_samples)} videos across {len(label_to_idx)} classes")
    
    # Shuffle samples
    np.random.shuffle(all_samples)
    
    # Split data
    n_samples = len(all_samples)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    splits = {
        'train': all_samples[:train_end],
        'val': all_samples[train_end:val_end],
        'test': all_samples[val_end:]
    }
    
    logger.info(f"Split sizes - Train: {len(splits['train'])}, "
                f"Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    
    # Save label mapping
    import json
    label_map_path = Path(dataset_root).parent / 'label_mapping.json'
    with open(label_map_path, 'w', encoding='utf-8') as f:
        json.dump(label_to_idx, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved label mapping to {label_map_path}")
    
    return splits, label_to_idx


if __name__ == "__main__":
    # Example usage
    extractor = VideoFeatureExtractor(max_frames=60)
    
    # Test on a single video
    test_video = "datasets/signVideos/Greetings/Hello/Hello_001.mp4"
    features = extractor.process_video(test_video)
    
    if features is not None:
        print(f"Extracted features shape: {features.shape}")
        print(f"Feature dimension per frame: {extractor.get_feature_dim()}")
    
    extractor.close()
