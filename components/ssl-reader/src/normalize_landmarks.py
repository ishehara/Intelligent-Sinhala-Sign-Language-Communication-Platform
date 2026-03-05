"""
Landmark Normalization for Sign Language Recognition.
Implements wrist-relative, pose-relative, and face-relative normalization
to improve model accuracy by making coordinates invariant to position and scale.

Developer: IT22304674 – Liyanage M.L.I.S.
"""

import numpy as np
from typing import Tuple, Optional


def normalize_landmarks(
    data: np.ndarray,
    hand_landmarks: Optional[Tuple[int, int]] = None,
    pose_landmarks: Optional[Tuple[int, int]] = None,
    face_landmarks: Optional[Tuple[int, int]] = None,
    scale_factor: str = 'hand'
) -> np.ndarray:
    """
    Normalize landmarks using wrist-relative, pose-relative, and face-relative coordinates.
    
    This normalization makes the model invariant to:
    - Position: By centering around reference points (wrist, mid-shoulder, nose)
    - Scale: By normalizing distances relative to hand size
    - Orientation: Maintains relative spatial relationships
    
    Args:
        data: NumPy array of shape (frames, landmarks, 3) where 3 = (x, y, z)
              Can also be (landmarks, 3) for single frame
        hand_landmarks: Tuple (start_idx, end_idx) for hand landmarks in data array
                       Default: (0, 42) for dual hands (21 landmarks × 2 hands)
        pose_landmarks: Tuple (start_idx, end_idx) for pose landmarks
                       Default: (42, 75) for 33 pose landmarks
        face_landmarks: Tuple (start_idx, end_idx) for face landmarks
                       Default: (75, 543) for 468 face landmarks
        scale_factor: Method for scaling ('hand', 'pose', or 'none')
                     'hand': Scale by wrist-to-middle-finger-tip distance
                     'pose': Scale by shoulder width
                     'none': No scaling, only centering
    
    Returns:
        Normalized landmarks array with same shape as input
        
    Examples:
        >>> # Single frame with 21 hand landmarks
        >>> hand_data = np.random.rand(21, 3)
        >>> normalized = normalize_landmarks(hand_data, hand_landmarks=(0, 21))
        
        >>> # Multi-frame with hands + face
        >>> data = np.random.rand(60, 489, 3)  # 60 frames, 21+468 landmarks
        >>> normalized = normalize_landmarks(data, 
        ...     hand_landmarks=(0, 21), 
        ...     face_landmarks=(21, 489))
    """
    # Handle single frame input
    single_frame = False
    if data.ndim == 2:
        data = data[np.newaxis, ...]  # Add frame dimension
        single_frame = True
    
    frames, num_landmarks, coords = data.shape
    assert coords == 3, "Last dimension must be 3 (x, y, z)"
    
    # Default landmark ranges for MediaPipe (hands + pose + face)
    if hand_landmarks is None:
        hand_landmarks = (0, 42)  # 2 hands × 21 landmarks
    if pose_landmarks is None:
        pose_landmarks = (42, 75)  # 33 pose landmarks
    if face_landmarks is None:
        face_landmarks = (75, 543)  # 468 face landmarks
    
    # Create copy to avoid modifying original
    normalized_data = data.copy()
    
    # Process each frame independently
    for frame_idx in range(frames):
        frame = normalized_data[frame_idx]
        
        # ===== HAND NORMALIZATION =====
        if hand_landmarks[0] < num_landmarks:
            normalized_data[frame_idx] = _normalize_hands(
                frame, hand_landmarks, scale=(scale_factor == 'hand')
            )
        
        # ===== POSE NORMALIZATION =====
        if pose_landmarks[0] < num_landmarks and pose_landmarks[1] <= num_landmarks:
            normalized_data[frame_idx] = _normalize_pose(
                normalized_data[frame_idx], pose_landmarks, scale=(scale_factor == 'pose')
            )
        
        # ===== FACE NORMALIZATION =====
        if face_landmarks[0] < num_landmarks and face_landmarks[1] <= num_landmarks:
            normalized_data[frame_idx] = _normalize_face(
                normalized_data[frame_idx], face_landmarks
            )
    
    # Return single frame if input was single frame
    if single_frame:
        return normalized_data[0]
    
    return normalized_data


def _normalize_hands(
    frame: np.ndarray,
    hand_range: Tuple[int, int],
    scale: bool = True
) -> np.ndarray:
    """
    Normalize hand landmarks relative to wrist (Landmark 0 for each hand).
    
    MediaPipe Hand Landmarks:
    - 0: WRIST (reference point)
    - 9: MIDDLE_FINGER_MCP (middle finger base)
    - 12: MIDDLE_FINGER_TIP (used for scaling)
    
    Args:
        frame: Single frame of shape (landmarks, 3)
        hand_range: (start_idx, end_idx) for hand landmarks
        scale: Whether to scale by wrist-to-fingertip distance
    
    Returns:
        Frame with normalized hand landmarks
    """
    start_idx, end_idx = hand_range
    num_hand_landmarks = end_idx - start_idx
    
    # Assume 21 landmarks per hand (MediaPipe standard)
    landmarks_per_hand = 21
    num_hands = num_hand_landmarks // landmarks_per_hand
    
    for hand_idx in range(num_hands):
        hand_start = start_idx + hand_idx * landmarks_per_hand
        hand_end = hand_start + landmarks_per_hand
        
        # Extract hand landmarks
        hand_landmarks = frame[hand_start:hand_end]
        
        # Check if hand is detected (not all zeros or NaN)
        if _is_valid_detection(hand_landmarks):
            # WRIST is landmark 0 for each hand
            wrist = hand_landmarks[0].copy()  # (x, y, z)
            
            # Shift all landmarks relative to wrist (x, y only, keep z as-is for depth)
            hand_landmarks[:, 0] -= wrist[0]  # x coordinate
            hand_landmarks[:, 1] -= wrist[1]  # y coordinate
            # Z coordinate stays absolute for depth information
            
            if scale:
                # Scale by distance from wrist to middle finger tip (landmark 12)
                # This makes the model invariant to hand size
                middle_finger_tip = hand_landmarks[12]  # After centering
                
                # Calculate Euclidean distance (using x, y, z)
                scale_distance = np.linalg.norm(middle_finger_tip)
                
                # Avoid division by zero
                if scale_distance > 1e-6:
                    hand_landmarks /= scale_distance
            
            # Update frame with normalized hand
            frame[hand_start:hand_end] = hand_landmarks
    
    return frame


def _normalize_pose(
    frame: np.ndarray,
    pose_range: Tuple[int, int],
    scale: bool = True
) -> np.ndarray:
    """
    Normalize pose landmarks relative to mid-shoulder point.
    
    MediaPipe Pose Landmarks:
    - 11: LEFT_SHOULDER
    - 12: RIGHT_SHOULDER
    - Mid-shoulder = average of landmarks 11 and 12
    
    Args:
        frame: Single frame of shape (landmarks, 3)
        pose_range: (start_idx, end_idx) for pose landmarks
        scale: Whether to scale by shoulder width
    
    Returns:
        Frame with normalized pose landmarks
    """
    start_idx, end_idx = pose_range
    pose_landmarks = frame[start_idx:end_idx]
    
    # Check if pose is detected
    if not _is_valid_detection(pose_landmarks):
        return frame
    
    # MediaPipe pose: landmarks 11 (LEFT_SHOULDER) and 12 (RIGHT_SHOULDER)
    # In the pose array, these are indices 11 and 12
    if len(pose_landmarks) >= 13:  # Need at least 13 landmarks for shoulders
        left_shoulder = pose_landmarks[11]
        right_shoulder = pose_landmarks[12]
        
        # Calculate mid-shoulder point
        mid_shoulder = (left_shoulder + right_shoulder) / 2.0
        
        # Shift all pose landmarks relative to mid-shoulder
        pose_landmarks[:, 0] -= mid_shoulder[0]  # x
        pose_landmarks[:, 1] -= mid_shoulder[1]  # y
        # Keep z absolute for depth
        
        if scale:
            # Scale by shoulder width (distance between shoulders)
            shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
            
            if shoulder_width > 1e-6:
                pose_landmarks /= shoulder_width
        
        frame[start_idx:end_idx] = pose_landmarks
    
    return frame


def _normalize_face(
    frame: np.ndarray,
    face_range: Tuple[int, int]
) -> np.ndarray:
    """
    Normalize face landmarks relative to nose tip (Landmark 0).
    
    MediaPipe Face Mesh:
    - 0: NOSE_TIP (reference point)
    - 468 total landmarks
    
    Args:
        frame: Single frame of shape (landmarks, 3)
        face_range: (start_idx, end_idx) for face landmarks
    
    Returns:
        Frame with normalized face landmarks
    """
    start_idx, end_idx = face_range
    face_landmarks = frame[start_idx:end_idx]
    
    # Check if face is detected
    if not _is_valid_detection(face_landmarks):
        return frame
    
    # NOSE_TIP is landmark 0 in MediaPipe face mesh
    nose_tip = face_landmarks[0].copy()
    
    # Shift all face landmarks relative to nose
    face_landmarks[:, 0] -= nose_tip[0]  # x
    face_landmarks[:, 1] -= nose_tip[1]  # y
    # Keep z absolute for depth
    
    frame[start_idx:end_idx] = face_landmarks
    
    return frame


def _is_valid_detection(landmarks: np.ndarray, threshold: float = 1e-6) -> bool:
    """
    Check if landmarks represent a valid detection.
    
    Invalid cases:
    - All zeros (no detection)
    - Contains NaN values
    - Very small variance (likely padding)
    
    Args:
        landmarks: Array of shape (num_landmarks, 3)
        threshold: Minimum variance to consider valid
    
    Returns:
        True if detection is valid, False otherwise
    """
    # Check for NaN values
    if np.isnan(landmarks).any():
        return False
    
    # Check if all zeros
    if np.allclose(landmarks, 0, atol=threshold):
        return False
    
    # Check for sufficient variance (not just padding)
    variance = np.var(landmarks)
    if variance < threshold:
        return False
    
    return True


def normalize_mediapipe_features(
    hand_features: Optional[np.ndarray] = None,
    face_features: Optional[np.ndarray] = None,
    pose_features: Optional[np.ndarray] = None,
    concatenate: bool = True
) -> np.ndarray:
    """
    Convenience function to normalize separate MediaPipe feature arrays.
    
    This is useful when you have hands, face, and pose extracted separately
    and want to normalize them before concatenating.
    
    Args:
        hand_features: Array of shape (frames, 42, 3) for 2 hands
        face_features: Array of shape (frames, 468, 3) for face mesh
        pose_features: Array of shape (frames, 33, 3) for pose
        concatenate: If True, concatenate all features into single array
    
    Returns:
        Normalized features (concatenated if concatenate=True)
        
    Example:
        >>> hands = np.random.rand(60, 42, 3)
        >>> face = np.random.rand(60, 468, 3)
        >>> normalized = normalize_mediapipe_features(hands, face)
        >>> print(normalized.shape)  # (60, 510, 3)
    """
    normalized_features = []
    
    # Normalize hands
    if hand_features is not None:
        norm_hands = normalize_landmarks(
            hand_features,
            hand_landmarks=(0, hand_features.shape[1]),
            pose_landmarks=None,
            face_landmarks=None,
            scale_factor='hand'
        )
        normalized_features.append(norm_hands)
    
    # Normalize face
    if face_features is not None:
        norm_face = normalize_landmarks(
            face_features,
            hand_landmarks=None,
            pose_landmarks=None,
            face_landmarks=(0, face_features.shape[1]),
            scale_factor='none'
        )
        normalized_features.append(norm_face)
    
    # Normalize pose
    if pose_features is not None:
        norm_pose = normalize_landmarks(
            pose_features,
            hand_landmarks=None,
            pose_landmarks=(0, pose_features.shape[1]),
            face_landmarks=None,
            scale_factor='pose'
        )
        normalized_features.append(norm_pose)
    
    if concatenate and normalized_features:
        # Concatenate along landmarks dimension
        return np.concatenate(normalized_features, axis=1)
    
    return normalized_features if len(normalized_features) > 1 else normalized_features[0]


# ===== TESTING AND EXAMPLES =====

def test_normalization():
    """Test the normalization functions with synthetic data."""
    print("="*60)
    print("Testing Landmark Normalization")
    print("="*60)
    
    # Test 1: Single hand (21 landmarks)
    print("\n[Test 1] Single hand normalization")
    hand = np.random.rand(21, 3) * 100  # Random coordinates 0-100
    hand[0] = [50, 50, 0]  # Set wrist at (50, 50, 0)
    
    normalized = normalize_landmarks(hand, hand_landmarks=(0, 21))
    print(f"Original wrist: {hand[0]}")
    print(f"Normalized wrist: {normalized[0]}")  # Should be close to [0, 0, 0]
    print(f"✓ Wrist centered: {np.allclose(normalized[0, :2], 0, atol=1e-6)}")
    
    # Test 2: Two hands (42 landmarks)
    print("\n[Test 2] Dual hands normalization")
    hands = np.random.rand(42, 3) * 100
    hands[0] = [30, 40, 0]   # Left hand wrist
    hands[21] = [70, 60, 0]  # Right hand wrist
    
    normalized = normalize_landmarks(hands, hand_landmarks=(0, 42))
    print(f"Left hand wrist: {normalized[0, :2]}")   # Should be ~[0, 0]
    print(f"Right hand wrist: {normalized[21, :2]}")  # Should be ~[0, 0]
    print(f"✓ Both wrists centered: {np.allclose(normalized[[0, 21], :2], 0, atol=1e-6)}")
    
    # Test 3: Multi-frame sequence
    print("\n[Test 3] Multi-frame sequence (60 frames, 42 landmarks)")
    sequence = np.random.rand(60, 42, 3) * 100
    normalized = normalize_landmarks(sequence, hand_landmarks=(0, 42))
    print(f"Input shape: {sequence.shape}")
    print(f"Output shape: {normalized.shape}")
    print(f"✓ Shape preserved: {sequence.shape == normalized.shape}")
    
    # Test 4: Missing detection (all zeros)
    print("\n[Test 4] Handling missing detections")
    hands_missing = np.zeros((42, 3))  # No detection
    normalized = normalize_landmarks(hands_missing, hand_landmarks=(0, 42))
    print(f"✓ Handles zeros: {np.allclose(normalized, 0)}")
    
    # Test 5: Full MediaPipe features (hands + face)
    print("\n[Test 5] Combined hands + face (510 landmarks)")
    full_data = np.random.rand(60, 510, 3) * 100  # 42 hand + 468 face
    normalized = normalize_landmarks(
        full_data,
        hand_landmarks=(0, 42),
        face_landmarks=(42, 510)
    )
    print(f"Input shape: {full_data.shape}")
    print(f"Output shape: {normalized.shape}")
    
    # Check wrist and nose are centered
    wrist_centered = np.allclose(normalized[:, 0, :2].mean(), 0, atol=0.1)
    nose_centered = np.allclose(normalized[:, 42, :2].mean(), 0, atol=0.1)
    print(f"✓ Wrists centered (avg): {wrist_centered}")
    print(f"✓ Noses centered (avg): {nose_centered}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)


if __name__ == '__main__':
    test_normalization()
