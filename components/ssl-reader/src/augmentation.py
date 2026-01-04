"""
Data augmentation for video sign language data.
"""
import cv2
import numpy as np
import random


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
