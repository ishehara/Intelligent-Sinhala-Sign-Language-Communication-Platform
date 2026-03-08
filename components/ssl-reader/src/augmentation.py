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
        time_warp_prob: float = 0.25,   # Reduced from 0.5 (was causing underfitting)
        frame_drop_prob: float = 0.2,   # Reduced from 0.4
        time_mask_prob: float = 0.15,   # Reduced from 0.3
        apply_prob: float = 0.7
    ):
        """
        Initialize augmenter.
        
        Args:
            rotation_range: Min/max rotation in degrees (default: -5 to +5)
            scale_range: Min/max scaling factor (default: 0.9 to 1.1)
            noise_std: Standard deviation for Gaussian noise (default: 0.002)
            temporal_shift_prob: Probability of temporal shifting (default: 0.3)
            time_warp_prob: Probability of time warping (speed change) (default: 0.5)
            frame_drop_prob: Probability of frame dropping (default: 0.4)
            time_mask_prob: Probability of time masking (occlusion) (default: 0.3)
            apply_prob: Probability of applying augmentation (default: 0.8)
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.temporal_shift_prob = temporal_shift_prob
        self.time_warp_prob = time_warp_prob
        self.frame_drop_prob = frame_drop_prob
        self.time_mask_prob = time_mask_prob
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
        
        # Apply temporal augmentations (CRITICAL for data multiplication)
        # These are applied independently with their own probabilities
        
        # Time Warping: Speed up or slow down the entire sequence (10-20% change)
        if random.random() < self.time_warp_prob:
            augmented = self._apply_time_warp(augmented)
        
        # Frame Dropping: Randomly remove ~5% of frames
        if random.random() < self.frame_drop_prob:
            augmented = self._apply_frame_drop(augmented)
        
        # Time Masking: Zero out 3-5 consecutive frames (simulates occlusion)
        if random.random() < self.time_mask_prob:
            augmented = self._apply_time_mask(augmented)
        
        # Legacy temporal shift (for backward compatibility)
        if random.random() < self.temporal_shift_prob:
            augmented = self._apply_temporal_shift(augmented)
        
        return augmented
    
    def _apply_spatial_augmentations(self, features: torch.Tensor) -> torch.Tensor:
        """Apply rotation, scaling, and noise to spatial coordinates."""
        frames, feature_dim = features.shape
        
        # Detect stream type by feature dimension
        if feature_dim == 126:
            # Hand-only stream (2 hands × 21 landmarks × 3)
            features = self._augment_landmarks(features, num_landmarks=42)
        elif feature_dim == 232:
            # Face-only stream (filtered: 60 landmarks × 3 + 52 blendshapes)
            # Only augment landmarks (first 180 dims), not blendshapes
            landmarks = features[:, :180]
            landmarks = self._augment_landmarks(landmarks, num_landmarks=60)
            features[:, :180] = landmarks
        elif feature_dim == 1456:
            # Face-only stream (full: 468 landmarks × 3 + 52 blendshapes)
            landmarks = features[:, :1404]
            landmarks = self._augment_landmarks(landmarks, num_landmarks=468)
            features[:, :1404] = landmarks
        elif feature_dim == 99:
            # Pose-only stream (33 landmarks × 3)
            features = self._augment_landmarks(features, num_landmarks=33)
        else:
            # Full concatenated stream - process each part
            if self.hand_end <= feature_dim:
                hand_features = features[:, self.hand_start:self.hand_end]
                hand_features = self._augment_landmarks(hand_features, num_landmarks=42)
                features[:, self.hand_start:self.hand_end] = hand_features
            
            # Process face landmarks (excluding blendshapes)
            if self.face_start < feature_dim:
                # Calculate actual face landmark count based on available dims
                available_face_dims = feature_dim - self.face_start
                if available_face_dims >= 180:  # Filtered face (60 landmarks × 3)
                    landmarks = features[:, self.face_start:self.face_start + 180]
                    landmarks = self._augment_landmarks(landmarks, num_landmarks=60)
                    features[:, self.face_start:self.face_start + 180] = landmarks
                elif available_face_dims >= 1404:  # Full face (468 landmarks × 3)
                    landmarks = features[:, self.face_start:self.face_start + 1404]
                    landmarks = self._augment_landmarks(landmarks, num_landmarks=468)
                    features[:, self.face_start:self.face_start + 1404] = landmarks
        
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
    
    def _apply_time_warp(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply time warping by speeding up or slowing down the sequence.
        
        This multiplies data by simulating different signing speeds.
        Speed change: 10-20% (e.g., 60 frames → 50-70 frames, then resampled back to 60)
        
        Args:
            features: Tensor of shape (frames, feature_dim)
            
        Returns:
            Time-warped features (same shape)
        """
        num_frames, feature_dim = features.shape
        
        # Random speed factor: 0.8-0.9 (faster) or 1.1-1.2 (slower)
        if random.random() < 0.5:
            speed_factor = random.uniform(0.8, 0.9)  # Speed up (fewer frames)
        else:
            speed_factor = random.uniform(1.1, 1.2)  # Slow down (more frames)
        
        # Calculate target number of frames
        target_frames = max(int(num_frames * speed_factor), 10)  # Minimum 10 frames
        
        # Create interpolation indices
        original_indices = torch.linspace(0, num_frames - 1, target_frames)
        
        # Interpolate features using linear interpolation
        warped_features = torch.zeros(target_frames, feature_dim, dtype=features.dtype, device=features.device)
        
        for i in range(target_frames):
            idx = original_indices[i]
            idx_floor = int(torch.floor(idx))
            idx_ceil = min(int(torch.ceil(idx)), num_frames - 1)
            
            if idx_floor == idx_ceil:
                warped_features[i] = features[idx_floor]
            else:
                # Linear interpolation
                weight = idx - idx_floor
                warped_features[i] = (1 - weight) * features[idx_floor] + weight * features[idx_ceil]
        
        # Resample back to original frame count
        if target_frames != num_frames:
            resample_indices = torch.linspace(0, target_frames - 1, num_frames)
            final_features = torch.zeros(num_frames, feature_dim, dtype=features.dtype, device=features.device)
            
            for i in range(num_frames):
                idx = resample_indices[i]
                idx_floor = int(torch.floor(idx))
                idx_ceil = min(int(torch.ceil(idx)), target_frames - 1)
                
                if idx_floor == idx_ceil:
                    final_features[i] = warped_features[idx_floor]
                else:
                    weight = idx - idx_floor
                    final_features[i] = (1 - weight) * warped_features[idx_floor] + weight * warped_features[idx_ceil]
            
            return final_features
        else:
            return warped_features
    
    def _apply_frame_drop(self, features: torch.Tensor) -> torch.Tensor:
        """
        Randomly drop ~5% of frames to prevent memorization.
        
        Forces the model to handle imperfect sequences (real-world scenario).
        
        Args:
            features: Tensor of shape (frames, feature_dim)
            
        Returns:
            Frame-dropped features (same shape, dropped frames are duplicated from neighbors)
        """
        num_frames = features.shape[0]
        
        # Drop 3-7% of frames
        drop_ratio = random.uniform(0.03, 0.07)
        num_drops = max(1, int(num_frames * drop_ratio))
        
        # Randomly select frames to drop (avoid first and last 3 frames)
        droppable_indices = list(range(3, num_frames - 3))
        if len(droppable_indices) < num_drops:
            return features  # Not enough frames to drop safely
        
        drop_indices = set(random.sample(droppable_indices, num_drops))
        
        # Create new sequence by skipping dropped frames
        keep_indices = [i for i in range(num_frames) if i not in drop_indices]
        
        # Pad back to original length by duplicating nearest neighbors
        features_dropped = features[keep_indices]
        
        # Interpolate to restore original length
        if len(keep_indices) < num_frames:
            scale_factor = len(keep_indices) / num_frames
            resample_indices = torch.linspace(0, len(keep_indices) - 1, num_frames)
            
            final_features = torch.zeros_like(features)
            for i in range(num_frames):
                idx = resample_indices[i]
                idx_floor = int(torch.floor(idx))
                idx_ceil = min(int(torch.ceil(idx)), len(keep_indices) - 1)
                
                if idx_floor == idx_ceil:
                    final_features[i] = features_dropped[idx_floor]
                else:
                    weight = idx - idx_floor
                    final_features[i] = (1 - weight) * features_dropped[idx_floor] + weight * features_dropped[idx_ceil]
            
            return final_features
        else:
            return features_dropped
    
    def _apply_time_mask(self, features: torch.Tensor) -> torch.Tensor:
        """
        Zero out a random block of 3-5 consecutive frames (time masking).
        
        Simulates occlusion/lag scenarios where hands briefly leave frame.
        
        Args:
            features: Tensor of shape (frames, feature_dim)
            
        Returns:
            Time-masked features (same shape)
        """
        num_frames = features.shape[0]
        
        # Mask length: 3-5 frames
        mask_length = random.randint(3, 5)
        
        # Avoid masking first/last 5 frames (start/end of sign is critical)
        if num_frames <= mask_length + 10:
            return features  # Sequence too short to mask safely
        
        # Random start position (leave 5 frames buffer on each end)
        mask_start = random.randint(5, num_frames - mask_length - 5)
        mask_end = mask_start + mask_length
        
        # Clone and zero out the masked region
        masked_features = features.clone()
        masked_features[mask_start:mask_end] = 0
        
        return masked_features


class StreamSpecificAugmenter:
    """
    Stream-specific augmentation tailored for each modality.
    
    Different augmentation strategies for:
    - Hand stream: Aggressive (fast, complex movements)
    - Face stream: Conservative (slow, contextual changes)
    - Pose stream: Minimal (smooth movements)
    
    Developer: IT22304674 – Liyanage M.L.I.S.
    """
    
    def __init__(
        self,
        hand_dim: int = 126,
        face_dim: int = 1456,
        pose_dim: int = 0,
        use_pose: bool = False
    ):
        """
        Initialize stream-specific augmenters.
        
        Args:
            hand_dim: Hand feature dimension (0-125 inclusive)
            face_dim: Face feature dimension (126-1581 inclusive)
            pose_dim: Pose feature dimension (optional)
            use_pose: Whether to use pose augmentation
        """
        self.hand_dim = hand_dim
        self.face_dim = face_dim
        self.pose_dim = pose_dim
        self.use_pose = use_pose
        
        # Hand augmentation (aggressive - fast movements need more variation)
        self.hand_augmenter = SkeletonAugmenter(
            rotation_range=(-8.0, 8.0),      # More rotation
            scale_range=(0.85, 1.15),        # More scaling
            noise_std=0.003,                 # More noise
            temporal_shift_prob=0.4,         # More temporal variation
            apply_prob=0.85                  # Apply frequently
        )
        
        # Face augmentation (conservative - expressions need stability)
        self.face_augmenter = SkeletonAugmenter(
            rotation_range=(-3.0, 3.0),      # Less rotation (head doesn't move much)
            scale_range=(0.95, 1.05),        # Less scaling
            noise_std=0.001,                 # Less noise (blendshapes sensitive)
            temporal_shift_prob=0.2,         # Less temporal (expressions are smooth)
            apply_prob=0.75                  # Apply moderately
        )
        
        # Pose augmentation (minimal - body movements are smooth)
        if use_pose and pose_dim > 0:
            self.pose_augmenter = SkeletonAugmenter(
                rotation_range=(-5.0, 5.0),
                scale_range=(0.9, 1.1),
                noise_std=0.002,
                temporal_shift_prob=0.3,
                apply_prob=0.7
            )
        else:
            self.pose_augmenter = None
    
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply stream-specific augmentation.
        
        Args:
            features: Tensor of shape (frames, total_feature_dim)
                     Features are concatenated: [hand, face, pose]
        
        Returns:
            Augmented features of same shape
        """
        # Clone to avoid modifying original
        augmented = features.clone()

        # Apply horizontal flip with 50% probability (mirrors sign & swaps hands)
        # Must happen before stream-specific augmentation so rotations are consistent
        if random.random() < 0.5:
            augmented = self._apply_horizontal_flip(augmented)

        # Augment hand stream (indices 0:126)
        if self.hand_dim > 0:
            hand_features = augmented[:, :self.hand_dim]
            hand_features = self._augment_hand_stream(hand_features)
            augmented[:, :self.hand_dim] = hand_features
        
        # Augment face stream (indices 126:1582)
        if self.face_dim > 0:
            face_start = self.hand_dim
            face_end = face_start + self.face_dim
            face_features = augmented[:, face_start:face_end]
            face_features = self._augment_face_stream(face_features)
            augmented[:, face_start:face_end] = face_features
        
        # Augment pose stream if enabled
        if self.use_pose and self.pose_dim > 0 and self.pose_augmenter is not None:
            pose_start = self.hand_dim + self.face_dim
            pose_end = pose_start + self.pose_dim
            pose_features = augmented[:, pose_start:pose_end]
            pose_features = self.pose_augmenter(pose_features)
            augmented[:, pose_start:pose_end] = pose_features
        
        return augmented
    
    def _apply_horizontal_flip(self, features: torch.Tensor) -> torch.Tensor:
        """
        Mirror the signer by flipping all X coordinates and swapping hand slots.
        Effectively doubles training data since mirrored signs are still valid.

        Layout: [hands(0:hand_dim), face(hand_dim:hand_dim+face_dim), pose(rest)]
        - Hands: flip X coords then swap hand-0 ↔ hand-1 buffers
        - Face: flip X coords of the 60 spatial landmarks (not blendshapes)
        - Pose: flip X coords of all 33 landmarks
        """
        frames = features.shape[0]
        flipped = features.clone()

        # --- Flip + swap hands ---
        if self.hand_dim == 126:
            hands = flipped[:, :126].view(frames, 2, 21, 3)
            hands[:, :, :, 0] = 1.0 - hands[:, :, :, 0]  # Mirror X
            hands = hands[:, [1, 0], :, :]                 # Swap left ↔ right slots
            flipped[:, :126] = hands.view(frames, 126)

        # --- Flip face landmark X (first 180 dims only, skip blendshapes) ---
        if self.face_dim >= 180:
            face_start = self.hand_dim
            face_lm = flipped[:, face_start:face_start + 180].view(frames, 60, 3)
            face_lm[:, :, 0] = 1.0 - face_lm[:, :, 0]
            flipped[:, face_start:face_start + 180] = face_lm.view(frames, 180)

        # --- Flip pose X ---
        if self.use_pose and self.pose_dim == 99:
            pose_start = self.hand_dim + self.face_dim
            pose = flipped[:, pose_start:pose_start + 99].view(frames, 33, 3)
            pose[:, :, 0] = 1.0 - pose[:, :, 0]
            flipped[:, pose_start:pose_start + 99] = pose.view(frames, 99)

        return flipped

    def _augment_hand_stream(self, hand_features: torch.Tensor) -> torch.Tensor:
        """Augment hand features with aggressive transformations."""
        frames = hand_features.shape[0]
        num_landmarks = 42  # 2 hands × 21 landmarks
        
        # Reshape to (frames, num_landmarks, 3)
        landmarks_3d = hand_features.view(frames, num_landmarks, 3)
        
        # Apply spatial augmentations with aggressive parameters
        if random.random() < 0.7:
            landmarks_3d = self._rotate_landmarks(
                landmarks_3d, 
                rotation_range=self.hand_augmenter.rotation_range
            )
        
        if random.random() < 0.7:
            landmarks_3d = self._scale_landmarks(
                landmarks_3d,
                scale_range=self.hand_augmenter.scale_range
            )
        
        if random.random() < 0.8:
            landmarks_3d = self._add_noise_to_landmarks(
                landmarks_3d,
                noise_std=self.hand_augmenter.noise_std
            )
        
        # Flatten back to (frames, hand_dim)
        hand_features = landmarks_3d.view(frames, num_landmarks * 3)
        
        # Apply temporal augmentation (more aggressive for hands)
        if random.random() < self.hand_augmenter.temporal_shift_prob:
            hand_features = self.hand_augmenter._apply_temporal_shift(hand_features)
        
        return hand_features
    
    def _augment_face_stream(self, face_features: torch.Tensor) -> torch.Tensor:
        """Augment face features with conservative transformations."""
        frames, face_dim = face_features.shape

        # --- 60 key-landmark filtered face (spatial only, no blendshapes) ---
        if face_dim == 180:
            landmarks_3d = face_features.view(frames, 60, 3)
            if random.random() < 0.65:
                landmarks_3d = self._rotate_landmarks(
                    landmarks_3d, self.face_augmenter.rotation_range)
            if random.random() < 0.65:
                landmarks_3d = self._scale_landmarks(
                    landmarks_3d, self.face_augmenter.scale_range)
            if random.random() < 0.75:
                landmarks_3d = self._add_noise_to_landmarks(
                    landmarks_3d, self.face_augmenter.noise_std)
            face_features = landmarks_3d.view(frames, 180)
            if random.random() < self.face_augmenter.temporal_shift_prob:
                face_features = self.face_augmenter._apply_temporal_shift(face_features)
            return face_features

        # --- 60 key-landmark filtered face + 52 blendshapes ---
        if face_dim == 232:
            landmark_part = face_features[:, :180].clone()
            blendshape_part = face_features[:, 180:]
            landmarks_3d = landmark_part.view(frames, 60, 3)
            if random.random() < 0.65:
                landmarks_3d = self._rotate_landmarks(
                    landmarks_3d, self.face_augmenter.rotation_range)
            if random.random() < 0.65:
                landmarks_3d = self._scale_landmarks(
                    landmarks_3d, self.face_augmenter.scale_range)
            if random.random() < 0.75:
                landmarks_3d = self._add_noise_to_landmarks(
                    landmarks_3d, self.face_augmenter.noise_std)
            face_features = torch.cat(
                [landmarks_3d.view(frames, 180), blendshape_part], dim=1)
            if random.random() < self.face_augmenter.temporal_shift_prob:
                face_features = self.face_augmenter._apply_temporal_shift(face_features)
            return face_features

        # --- Full 468-landmark face (legacy / full mode) ---
        # Split face landmarks and blendshapes
        # Landmarks: 0:1404 (468 × 3)
        # Blendshapes: 1404:1456 (52)
        landmark_dim = 468 * 3  # 1404
        num_landmarks = 468
        
        if face_dim >= landmark_dim:
            face_landmarks = face_features[:, :landmark_dim]
            face_blendshapes = face_features[:, landmark_dim:] if face_dim > landmark_dim else None
            
            # Reshape to (frames, num_landmarks, 3)
            landmarks_3d = face_landmarks.view(frames, num_landmarks, 3)
            
            # Apply spatial augmentations with conservative parameters
            if random.random() < 0.7:
                landmarks_3d = self._rotate_landmarks(
                    landmarks_3d,
                    rotation_range=self.face_augmenter.rotation_range
                )
            
            if random.random() < 0.7:
                landmarks_3d = self._scale_landmarks(
                    landmarks_3d,
                    scale_range=self.face_augmenter.scale_range
                )
            
            if random.random() < 0.8:
                landmarks_3d = self._add_noise_to_landmarks(
                    landmarks_3d,
                    noise_std=self.face_augmenter.noise_std
                )
            
            # Flatten back
            face_landmarks = landmarks_3d.view(frames, landmark_dim)
            
            # Apply very minimal temporal to landmarks (keep expressions smooth)
            if random.random() < self.face_augmenter.temporal_shift_prob:
                face_landmarks = self.face_augmenter._apply_temporal_shift(face_landmarks)
            
            # Recombine with blendshapes (not augmented)
            if face_blendshapes is not None:
                face_features = torch.cat([face_landmarks, face_blendshapes], dim=1)
            else:
                face_features = face_landmarks
        
        return face_features
    
    def _rotate_landmarks(self, landmarks: torch.Tensor, rotation_range: tuple) -> torch.Tensor:
        """
        Rotate (x, y) coordinates around center (0.5, 0.5).
        
        Args:
            landmarks: Tensor of shape (frames, num_landmarks, 3)
            rotation_range: Tuple of (min_degrees, max_degrees)
            
        Returns:
            Rotated landmarks
        """
        frames, num_landmarks, _ = landmarks.shape
        
        # Random rotation angle (in radians)
        angle_deg = random.uniform(*rotation_range)
        angle_rad = np.deg2rad(angle_deg)
        
        # Rotation matrix for 2D
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Center point
        cx, cy = 0.5, 0.5
        
        # Rotate x and y coordinates
        x = landmarks[:, :, 0].clone()
        y = landmarks[:, :, 1].clone()
        
        # Translate to origin, rotate, translate back
        x_centered = x - cx
        y_centered = y - cy
        
        landmarks[:, :, 0] = x_centered * cos_a - y_centered * sin_a + cx
        landmarks[:, :, 1] = x_centered * sin_a + y_centered * cos_a + cy
        
        return landmarks
    
    def _scale_landmarks(self, landmarks: torch.Tensor, scale_range: tuple) -> torch.Tensor:
        """
        Scale landmarks uniformly.
        
        Args:
            landmarks: Tensor of shape (frames, num_landmarks, 3)
            scale_range: Tuple of (min_scale, max_scale)
            
        Returns:
            Scaled landmarks
        """
        # Random scale factor
        scale = random.uniform(*scale_range)
        
        # Center point
        cx, cy, cz = 0.5, 0.5, 0.0
        
        # Scale around center
        landmarks[:, :, 0] = (landmarks[:, :, 0] - cx) * scale + cx
        landmarks[:, :, 1] = (landmarks[:, :, 1] - cy) * scale + cy
        landmarks[:, :, 2] = (landmarks[:, :, 2] - cz) * scale + cz
        
        return landmarks
    
    def _add_noise_to_landmarks(self, landmarks: torch.Tensor, noise_std: float) -> torch.Tensor:
        """
        Add Gaussian noise to landmarks.
        
        Args:
            landmarks: Tensor of shape (frames, num_landmarks, 3)
            noise_std: Standard deviation of Gaussian noise
            
        Returns:
            Noisy landmarks
        """
        noise = torch.randn_like(landmarks) * noise_std
        return landmarks + noise


# Create convenient factory function
# Create convenient factory function
def create_augmenter(
    augmentation_mode: str = 'unified',
    hand_dim: int = 126,
    face_dim: int = 1456,
    pose_dim: int = 0,
    use_pose: bool = False,
    **kwargs
):
    """
    Factory function to create augmenters.
    
    Args:
        augmentation_mode: 'unified' or 'stream_specific'
        hand_dim: Hand feature dimension
        face_dim: Face feature dimension
        pose_dim: Pose feature dimension
        use_pose: Whether to use pose
        **kwargs: Additional arguments for SkeletonAugmenter
    
    Returns:
        Augmenter instance
    """
    if augmentation_mode == 'stream_specific':
        return StreamSpecificAugmenter(
            hand_dim=hand_dim,
            face_dim=face_dim,
            pose_dim=pose_dim,
            use_pose=use_pose
        )
    else:
        return SkeletonAugmenter(**kwargs)
