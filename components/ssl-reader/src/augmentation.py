"""
Data augmentation for video sign language data.

Developer: IT22304674 – Liyanage M.L.I.S.
"""
import cv2
import numpy as np
import torch
import random
from typing import Tuple


class VideoAugmentation:
    """Apply augmentations to video frames."""
    
    @staticmethod
    def random_brightness(frame, delta=30):
        """Randomly adjust brightness."""
        value = random.randint(-delta, delta)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def random_rotation(frame, max_angle=10):
        """Randomly rotate frame."""
        angle = random.uniform(-max_angle, max_angle)
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(frame, M, (w, h))
    
    @staticmethod
    def random_scale(frame, scale_range=(0.9, 1.1)):
        """Randomly scale frame."""
        scale = random.uniform(*scale_range)
        h, w = frame.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Crop or pad to original size
        if scale > 1:
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            return resized[start_y:start_y+h, start_x:start_x+w]
        else:
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            return cv2.copyMakeBorder(resized, pad_y, h-new_h-pad_y, 
                                     pad_x, w-new_w-pad_x, 
                                     cv2.BORDER_CONSTANT, value=(0,0,0))
    
    @staticmethod
    def random_flip(frame, probability=0.5):
        """Randomly flip frame horizontally."""
        if random.random() < probability:
            return cv2.flip(frame, 1)
        return frame
    
    @staticmethod
    def temporal_crop(frames, crop_ratio=0.8):
        """Randomly crop temporal sequence."""
        n_frames = len(frames)
        crop_length = int(n_frames * crop_ratio)
        if crop_length < n_frames:
            start_idx = random.randint(0, n_frames - crop_length)
            return frames[start_idx:start_idx + crop_length]
        return frames
    
    @classmethod
    def augment_frame(cls, frame, training=True):
        """Apply random augmentations to a frame."""
        if not training:
            return frame
        
        # Apply augmentations with probability
        if random.random() < 0.5:
            frame = cls.random_brightness(frame)
        if random.random() < 0.3:
            frame = cls.random_rotation(frame)
        if random.random() < 0.3:
            frame = cls.random_scale(frame)
        # Don't flip for sign language - changes meaning!
        
        return frame


class SkeletonAugmenter:
    """
    Augmentation for skeleton landmark features to combat overfitting.
    
    Applies transformations to flattened (x, y, z) coordinates:
    - Hands: Indices 0-125 (2 hands × 21 landmarks × 3 coords)
    - Face: Indices 126-1581 (468 landmarks × 3 coords + 52 blendshapes)
    
    Developer: IT22304674 – Liyanage M.L.I.S.
    """
    
    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-5.0, 5.0),
        scale_range: Tuple[float, float] = (0.9, 1.1),
        noise_std: float = 0.002,
        temporal_shift_prob: float = 0.3,
        apply_prob: float = 0.8
    ):
        """
        Initialize augmenter.
        
        Args:
            rotation_range: Min/max rotation in degrees (default: -5 to +5)
            scale_range: Min/max scaling factor (default: 0.9 to 1.1)
            noise_std: Standard deviation for Gaussian noise (default: 0.002)
            temporal_shift_prob: Probability of temporal shifting (default: 0.3)
            apply_prob: Probability of applying augmentation (default: 0.8)
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.temporal_shift_prob = temporal_shift_prob
        self.apply_prob = apply_prob
        
        # Feature indices
        self.hand_start = 0
        self.hand_end = 126  # Exclusive
        self.face_start = 126
        self.face_end = 1582  # Exclusive (includes blendshapes)
    
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to feature tensor.
        
        Args:
            features: Tensor of shape (frames, feature_dim) with flattened coordinates
            
        Returns:
            Augmented feature tensor of same shape
        """
        if random.random() > self.apply_prob:
            return features
        
        # Clone to avoid modifying original
        augmented = features.clone()
        
        # Apply spatial augmentations (rotation, scaling, noise)
        augmented = self._apply_spatial_augmentations(augmented)
        
        # Apply temporal augmentations (frame shifting)
        if random.random() < self.temporal_shift_prob:
            augmented = self._apply_temporal_shift(augmented)
        
        return augmented
    
    def _apply_spatial_augmentations(self, features: torch.Tensor) -> torch.Tensor:
        """Apply rotation, scaling, and noise to spatial coordinates."""
        frames, feature_dim = features.shape
        
        # Process hands (indices 0-125)
        if self.hand_end <= feature_dim:
            hand_features = features[:, self.hand_start:self.hand_end]
            hand_features = self._augment_landmarks(hand_features, num_landmarks=42)
            features[:, self.hand_start:self.hand_end] = hand_features
        
        # Process face landmarks (indices 126-1529, excluding blendshapes 1530-1581)
        face_landmark_end = min(self.face_start + 468 * 3, feature_dim)
        if face_landmark_end <= feature_dim:
            face_features = features[:, self.face_start:face_landmark_end]
            face_features = self._augment_landmarks(face_features, num_landmarks=468)
            features[:, self.face_start:face_landmark_end] = face_features
        
        # Don't augment blendshapes (indices 1530-1581) - they're not spatial
        
        return features
    
    def _augment_landmarks(self, landmarks: torch.Tensor, num_landmarks: int) -> torch.Tensor:
        """
        Augment a set of landmarks with rotation, scaling, and noise.
        
        Args:
            landmarks: Tensor of shape (frames, num_landmarks*3)
            num_landmarks: Number of landmarks in the set
            
        Returns:
            Augmented landmarks tensor
        """
        frames = landmarks.shape[0]
        
        # Reshape to (frames, num_landmarks, 3)
        landmarks_3d = landmarks.view(frames, num_landmarks, 3)
        
        # Apply rotation to (x, y) coordinates
        if random.random() < 0.7:
            landmarks_3d = self._rotate_landmarks(landmarks_3d)
        
        # Apply scaling
        if random.random() < 0.7:
            landmarks_3d = self._scale_landmarks(landmarks_3d)
        
        # Add Gaussian noise
        if random.random() < 0.8:
            landmarks_3d = self._add_noise(landmarks_3d)
        
        # Flatten back
        return landmarks_3d.view(frames, num_landmarks * 3)
    
    def _rotate_landmarks(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Rotate (x, y) coordinates around center (0.5, 0.5).
        
        Args:
            landmarks: Tensor of shape (frames, num_landmarks, 3)
            
        Returns:
            Rotated landmarks
        """
        frames, num_landmarks, _ = landmarks.shape
        
        # Random rotation angle (in radians)
        angle_deg = random.uniform(*self.rotation_range)
        angle_rad = np.deg2rad(angle_deg)
        
        # Rotation matrix for 2D
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Center point
        cx, cy = 0.5, 0.5
        
        # Extract x, y coordinates
        x = landmarks[:, :, 0]  # (frames, num_landmarks)
        y = landmarks[:, :, 1]
        
        # Translate to origin
        x_centered = x - cx
        y_centered = y - cy
        
        # Apply rotation
        x_rotated = x_centered * cos_a - y_centered * sin_a
        y_rotated = x_centered * sin_a + y_centered * cos_a
        
        # Translate back
        landmarks[:, :, 0] = x_rotated + cx
        landmarks[:, :, 1] = y_rotated + cy
        
        # Keep z unchanged
        
        return landmarks
    
    def _scale_landmarks(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Scale landmarks by random factor to simulate distance variation.
        
        Args:
            landmarks: Tensor of shape (frames, num_landmarks, 3)
            
        Returns:
            Scaled landmarks
        """
        scale_factor = random.uniform(*self.scale_range)
        
        # Scale x, y, z coordinates around center (0.5, 0.5, 0.0)
        cx, cy, cz = 0.5, 0.5, 0.0
        
        landmarks[:, :, 0] = (landmarks[:, :, 0] - cx) * scale_factor + cx
        landmarks[:, :, 1] = (landmarks[:, :, 1] - cy) * scale_factor + cy
        landmarks[:, :, 2] = (landmarks[:, :, 2] - cz) * scale_factor + cz
        
        return landmarks
    
    def _add_noise(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to simulate hand tremors and detection uncertainty.
        
        Args:
            landmarks: Tensor of shape (frames, num_landmarks, 3)
            
        Returns:
            Noisy landmarks
        """
        noise = torch.randn_like(landmarks) * self.noise_std
        return landmarks + noise
    
    def _apply_temporal_shift(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal shifting by randomly skipping or duplicating frames.
        
        This simulates different signing speeds and helps model generalize.
        
        Args:
            features: Tensor of shape (frames, feature_dim)
            
        Returns:
            Temporally shifted features (same shape)
        """
        num_frames = features.shape[0]
        
        # Randomly skip or duplicate 1-3 frames
        num_shifts = random.randint(1, 3)
        
        if random.random() < 0.5:
            # Skip frames (speed up)
            indices = list(range(num_frames))
            # Randomly remove num_shifts frames
            for _ in range(min(num_shifts, num_frames - 10)):  # Keep at least 10 frames
                if len(indices) > 10:
                    idx_to_remove = random.randint(0, len(indices) - 1)
                    indices.pop(idx_to_remove)
            
            # Pad to original length by duplicating last frame
            while len(indices) < num_frames:
                indices.append(indices[-1])
            
            return features[indices]
        else:
            # Duplicate frames (slow down)
            indices = list(range(num_frames))
            # Randomly duplicate num_shifts frames
            for _ in range(num_shifts):
                idx_to_duplicate = random.randint(0, len(indices) - 1)
                indices.insert(idx_to_duplicate, indices[idx_to_duplicate])
            
            # Truncate to original length
            indices = indices[:num_frames]
            
            return features[indices]
