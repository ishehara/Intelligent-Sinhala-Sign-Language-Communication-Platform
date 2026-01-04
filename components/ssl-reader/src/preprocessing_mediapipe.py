"""
Video preprocessing using MediaPipe hand and pose landmarks.
Compatible with MediaPipe 0.10.31+ (tasks API).

Developer: IT22304674 – Liyanage M.L.I.S.
"""

import os
# Suppress MediaPipe warnings
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json
import requests
from tqdm import tqdm

# MediaPipe is optional - will fall back if not available
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    logging.warning("MediaPipe not available. Install with: pip install mediapipe")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MediaPipeFeatureExtractor:
    """Extracts hand and pose landmarks from sign language videos using MediaPipe."""
    
    # Model URLs for MediaPipe tasks
    HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker/float16/latest/pose_landmarker.task"
    
    def __init__(
        self,
        max_frames: int = 60,
        use_hands: bool = True,
        use_pose: bool = False,  # Disabled by default - pose model URL needs update
        model_dir: str = None
    ):
        """
        Initialize MediaPipe feature extractor.
        
        Args:
            max_frames: Maximum number of frames to extract per video
            use_hands: Whether to extract hand landmarks
            use_pose: Whether to extract pose landmarks
            model_dir: Directory to store/load model files
        """
        if not HAS_MEDIAPIPE:
            raise ImportError("MediaPipe is required. Install with: pip install mediapipe")
        
        self.max_frames = max_frames
        self.use_hands = use_hands
        self.use_pose = use_pose
        
        # Setup model directory
        if model_dir is None:
            model_dir = Path(__file__).parent.parent / "models" / "mediapipe"
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Download and initialize models
        self._setup_models()
    
    def _download_model(self, url: str, filename: str) -> Path:
        """Download a MediaPipe model file if it doesn't exist."""
        model_path = self.model_dir / filename
        
        if model_path.exists():
            logger.info(f"Model already exists: {model_path}")
            return model_path
        
        logger.info(f"Downloading {filename}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Downloaded: {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            raise
    
    def _setup_models(self):
        """Setup MediaPipe models."""
        # Download models
        if self.use_hands:
            hand_model_path = self._download_model(self.HAND_MODEL_URL, "hand_landmarker.task")
            
            # Create hand landmarker with GPU delegate
            base_options = python.BaseOptions(
                model_asset_path=str(hand_model_path),
                delegate=python.BaseOptions.Delegate.GPU
            )
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3
            )
            self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        else:
            self.hand_landmarker = None
        
        if self.use_pose:
            pose_model_path = self._download_model(self.POSE_MODEL_URL, "pose_landmarker.task")
            
            # Create pose landmarker
            base_options = python.BaseOptions(model_asset_path=str(pose_model_path))
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                min_pose_detection_confidence=0.3,
                min_pose_presence_confidence=0.3,
                min_tracking_confidence=0.3
            )
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
        else:
            self.pose_landmarker = None
    
    def extract_frame_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract MediaPipe landmarks from a single frame.
        
        Args:
            frame: BGR frame from video
            
        Returns:
            Feature vector containing hand and pose landmarks
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        features = []
        
        # Extract hand landmarks
        if self.hand_landmarker:
            try:
                hand_result = self.hand_landmarker.detect(mp_image)
                hand_features = self._process_hand_landmarks(hand_result)
                features.append(hand_features)
            except Exception as e:
                logger.warning(f"Hand detection failed: {e}")
                # Use zeros if detection fails
                features.append(np.zeros(21 * 2 * 3))  # 2 hands, 21 landmarks, 3 coords each
        
        # Extract pose landmarks
        if self.pose_landmarker:
            try:
                pose_result = self.pose_landmarker.detect(mp_image)
                pose_features = self._process_pose_landmarks(pose_result)
                features.append(pose_features)
            except Exception as e:
                logger.warning(f"Pose detection failed: {e}")
                # Use zeros if detection fails
                features.append(np.zeros(33 * 3))  # 33 landmarks, 3 coords each
        
        # Combine all features
        return np.concatenate(features) if features else np.zeros(1)
    
    def _process_hand_landmarks(self, result) -> np.ndarray:
        """Process hand landmark results into feature vector."""
        # Each hand has 21 landmarks with x, y, z coordinates
        hand_features = np.zeros(21 * 2 * 3)  # 2 hands max
        
        if result.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(result.hand_landmarks[:2]):  # Max 2 hands
                for lm_idx, landmark in enumerate(hand_landmarks):
                    base_idx = hand_idx * 21 * 3 + lm_idx * 3
                    hand_features[base_idx:base_idx + 3] = [landmark.x, landmark.y, landmark.z]
        
        return hand_features
    
    def _process_pose_landmarks(self, result) -> np.ndarray:
        """Process pose landmark results into feature vector."""
        # Pose has 33 landmarks with x, y, z coordinates
        pose_features = np.zeros(33 * 3)
        
        if result.pose_landmarks:
            pose_landmarks = result.pose_landmarks[0]  # First person
            for lm_idx, landmark in enumerate(pose_landmarks):
                base_idx = lm_idx * 3
                pose_features[base_idx:base_idx + 3] = [landmark.x, landmark.y, landmark.z]
        
        return pose_features
    
    def get_feature_dim(self) -> int:
        """Get the dimension of extracted features."""
        dim = 0
        if self.use_hands:
            dim += 21 * 2 * 3  # 2 hands, 21 landmarks, 3 coords = 126
        if self.use_pose:
            dim += 33 * 3  # 33 landmarks, 3 coords = 99
        return dim if dim > 0 else 1
    
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
                    # Extract features from this frame
                    features = self.extract_frame_features(frame)
                    frames_data.append(features)
                    next_sample_idx += 1
                
                current_frame += 1
            
            cap.release()
            
            if not frames_data:
                logger.warning(f"No frames extracted from {video_path}")
                return None
            
            # Pad or truncate to max_frames
            frames_data = np.array(frames_data)
            
            if len(frames_data) < self.max_frames:
                # Pad with zeros
                padding = np.zeros((self.max_frames - len(frames_data), frames_data.shape[1]))
                frames_data = np.vstack([frames_data, padding])
            elif len(frames_data) > self.max_frames:
                # Truncate
                frames_data = frames_data[:self.max_frames]
            
            return frames_data
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            return None


def create_dataset_splits(
    dataset_root: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_frames: int = 60
) -> Tuple[Dict[str, List], Dict[str, int]]:
    """
    Create train/val/test splits from video dataset.
    (Same implementation as preprocessing_simple.py)
    """
    from preprocessing_simple import create_dataset_splits as create_splits
    return create_splits(dataset_root, train_ratio, val_ratio, test_ratio, max_frames)
