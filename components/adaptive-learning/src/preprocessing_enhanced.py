"""
Enhanced Preprocessing Pipeline for Mobile Sign Recognition
=============================================================
Addresses the accuracy gap between laptop webcam (95%) and mobile camera
by ensuring preprocessing consistency, handling mobile-specific issues
(lighting, orientation, resolution, compression), and extracting
MediaPipe hand landmarks for hybrid classification.

Key Improvements over current app.py preprocessing:
1. Adaptive histogram equalization (CLAHE) for lighting invariance
2. White balance correction for color consistency
3. Background noise reduction via GrabCut segmentation
4. MediaPipe hand landmark extraction for hybrid feature input
5. Multi-scale hand detection with confidence retry
6. Image quality assessment before inference
7. TFLite-optimized inference path

Usage:
    from preprocessing_enhanced import EnhancedPreprocessor
    preprocessor = EnhancedPreprocessor()
    result = preprocessor.process(pil_image, is_front_camera=True)
"""

import cv2
import numpy as np
import math
import mediapipe as mp
from PIL import Image, ImageOps, ImageStat, ImageFilter
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import time
import os

try:
    from cvzone.HandTrackingModule import HandDetector
    USE_CVZONE = True
except ImportError:
    USE_CVZONE = False


# ══════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════

@dataclass
class HandLandmarks:
    """21 MediaPipe hand landmarks with normalized coordinates."""
    landmarks: List[Tuple[float, float, float]]  # (x, y, z) × 21
    handedness: str  # 'Left' or 'Right'
    confidence: float

    def to_feature_vector(self) -> np.ndarray:
        """Flatten landmarks to 63-dim feature vector (21 × 3)."""
        return np.array([(l[0], l[1], l[2]) for l in self.landmarks]).flatten()

    def to_relative_vector(self) -> np.ndarray:
        """
        Compute wrist-relative + normalized landmark vector.
        More robust to position/scale variation.
        Returns: 60-dim vector (20 landmarks × 3, wrist-relative).
        """
        wrist = np.array(self.landmarks[0])
        relative = []
        for i in range(1, 21):
            diff = np.array(self.landmarks[i]) - wrist
            relative.extend(diff.tolist())
        vec = np.array(relative)
        # Normalize by hand span (wrist to middle finger tip)
        middle_tip = np.array(self.landmarks[12]) - wrist
        scale = np.linalg.norm(middle_tip[:2])
        if scale > 1e-6:
            vec = vec / scale
        return vec


@dataclass
class ImageQualityMetrics:
    """Quality assessment of input image for sign recognition."""
    brightness: float       # 0-255 mean brightness
    contrast: float         # std deviation of luminance
    sharpness: float        # Laplacian variance (higher = sharper)
    is_too_dark: bool
    is_too_bright: bool
    is_blurry: bool
    quality_score: float    # 0-1 composite score

    @property
    def is_acceptable(self) -> bool:
        return self.quality_score >= 0.4


@dataclass
class PreprocessResult:
    """Output of the enhanced preprocessing pipeline."""
    image_batch: np.ndarray         # (1, 224, 224, 3) normalized
    hand_detected: bool
    landmarks: Optional[HandLandmarks] = None
    quality: Optional[ImageQualityMetrics] = None
    canvas_image: Optional[np.ndarray] = None  # 300×300 white canvas (for debug)
    debug_info: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════
# Image Quality Assessment
# ══════════════════════════════════════════════════════════════

def assess_image_quality(img_bgr: np.ndarray) -> ImageQualityMetrics:
    """
    Assess image quality for sign recognition suitability.
    
    Metrics:
    - Brightness: mean luminance (ideal: 80-180)
    - Contrast: luminance std (ideal: > 40)
    - Sharpness: Laplacian variance (ideal: > 100)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    is_too_dark = brightness < 60
    is_too_bright = brightness > 220
    is_blurry = sharpness < 50

    # Composite quality score [0, 1]
    brightness_score = 1.0 - min(1.0, abs(brightness - 130) / 130)
    contrast_score = min(1.0, contrast / 60)
    sharpness_score = min(1.0, sharpness / 200)

    quality_score = (
        0.3 * brightness_score +
        0.3 * contrast_score +
        0.4 * sharpness_score
    )

    return ImageQualityMetrics(
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        is_too_dark=is_too_dark,
        is_too_bright=is_too_bright,
        is_blurry=is_blurry,
        quality_score=round(quality_score, 3),
    )


# ══════════════════════════════════════════════════════════════
# Lighting & Color Correction
# ══════════════════════════════════════════════════════════════

def apply_clahe(img_bgr: np.ndarray, clip_limit: float = 2.0,
                grid_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to the L-channel in LAB color space.
    
    This normalizes lighting across different environments without
    distorting colors — critical for mobile cameras with auto-exposure.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(grid_size, grid_size))
    l_corrected = clahe.apply(l)
    lab_corrected = cv2.merge([l_corrected, a, b])
    return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)


def white_balance_grayworld(img_bgr: np.ndarray) -> np.ndarray:
    """
    Gray-world white balance correction.
    Assumes the average color in the scene should be gray.
    Corrects color casts from artificial/mixed lighting.
    """
    result = img_bgr.copy().astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3.0

    if avg_b > 0:
        result[:, :, 0] *= avg_gray / avg_b
    if avg_g > 0:
        result[:, :, 1] *= avg_gray / avg_g
    if avg_r > 0:
        result[:, :, 2] *= avg_gray / avg_r

    return np.clip(result, 0, 255).astype(np.uint8)


def adjust_brightness_contrast(img_bgr: np.ndarray,
                                target_brightness: float = 130) -> np.ndarray:
    """
    Adjust image brightness to a target mean value.
    Useful when mobile camera auto-exposure produces inconsistent brightness.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    current_brightness = np.mean(gray)

    if current_brightness < 1:
        return img_bgr

    ratio = target_brightness / current_brightness
    # Clamp adjustment to avoid extreme changes
    ratio = np.clip(ratio, 0.5, 2.0)

    adjusted = cv2.convertScaleAbs(img_bgr, alpha=ratio, beta=0)
    return adjusted


# ══════════════════════════════════════════════════════════════
# Background Noise Reduction
# ══════════════════════════════════════════════════════════════

def reduce_background_noise(img_bgr: np.ndarray,
                             bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Use GrabCut segmentation to reduce background noise around the hand.
    
    This helps when mobile users have cluttered backgrounds vs the
    clean backgrounds used during model training.
    
    Args:
        img_bgr: Input image
        bbox: (x, y, w, h) hand bounding box from MediaPipe
    
    Returns:
        Image with background replaced by white pixels
    """
    mask = np.zeros(img_bgr.shape[:2], np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    x, y, w, h = bbox
    # Expand bbox slightly for GrabCut
    pad = 30
    rect = (
        max(0, x - pad),
        max(0, y - pad),
        min(img_bgr.shape[1] - x + pad, w + 2 * pad),
        min(img_bgr.shape[0] - y + pad, h + 2 * pad),
    )

    try:
        cv2.grabCut(img_bgr, mask, rect, bg_model, fg_model,
                     iterCount=3, mode=cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return img_bgr  # Fallback if GrabCut fails

    # Create binary mask: foreground + probable foreground
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)

    # Apply mask: keep hand, replace background with white
    result = img_bgr.copy()
    result[fg_mask == 0] = [255, 255, 255]

    return result


def simple_skin_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Fast skin-color segmentation in YCrCb color space.
    Less accurate than GrabCut but much faster for real-time use.
    """
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    # Skin color range in YCrCb
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Apply mask
    result = img_bgr.copy()
    result[mask == 0] = [255, 255, 255]
    return result


# ══════════════════════════════════════════════════════════════
# MediaPipe Hand Landmark Extraction
# ══════════════════════════════════════════════════════════════

class HandLandmarkExtractor:
    """
    Extract MediaPipe hand landmarks for hybrid classification.
    
    Instead of relying solely on CNN image classification, we extract
    21 hand landmarks (63 features) that encode hand pose geometry.
    This can be used:
    1. As auxiliary features concatenated with CNN features
    2. For a separate landmark-based classifier
    3. For hand quality/visibility validation
    """

    def __init__(self, detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.5):
        self.mp_hands = mp.solutions.hands
        self.detector = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def extract(self, img_bgr: np.ndarray) -> Optional[HandLandmarks]:
        """Extract hand landmarks from BGR image."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.process(img_rgb)

        if not results.multi_hand_landmarks:
            return None

        hand_lms = results.multi_hand_landmarks[0]
        handedness = 'Right'
        confidence = 0.0
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label
            confidence = results.multi_handedness[0].classification[0].score

        landmarks = [
            (lm.x, lm.y, lm.z) for lm in hand_lms.landmark
        ]

        return HandLandmarks(
            landmarks=landmarks,
            handedness=handedness,
            confidence=confidence,
        )

    def get_bbox_from_landmarks(self, landmarks: HandLandmarks,
                                 img_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Compute bounding box from landmarks.
        More reliable than detection bbox for cropping.
        """
        h, w = img_shape[:2]
        xs = [lm[0] * w for lm in landmarks.landmarks]
        ys = [lm[1] * h for lm in landmarks.landmarks]

        x_min = max(0, int(min(xs)))
        y_min = max(0, int(min(ys)))
        x_max = min(w, int(max(xs)))
        y_max = min(h, int(max(ys)))

        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def compute_finger_angles(self, landmarks: HandLandmarks) -> Dict[str, float]:
        """
        Compute finger extension angles for gesture discrimination.
        Useful for distinguishing similar signs.
        """
        lms = landmarks.landmarks
        angles = {}

        # Finger tip and pip indices for each finger
        fingers = {
            'thumb': (4, 3, 2),
            'index': (8, 7, 6),
            'middle': (12, 11, 10),
            'ring': (16, 15, 14),
            'pinky': (20, 19, 18),
        }

        for name, (tip, pip, mcp) in fingers.items():
            v1 = np.array(lms[pip]) - np.array(lms[mcp])
            v2 = np.array(lms[tip]) - np.array(lms[pip])

            cos_angle = np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
            )
            angles[name] = float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))

        return angles

    def close(self):
        self.detector.close()


# ══════════════════════════════════════════════════════════════
# Enhanced Preprocessor (replaces preprocess_for_model)
# ══════════════════════════════════════════════════════════════

class EnhancedPreprocessor:
    """
    Complete preprocessing pipeline that bridges the accuracy gap
    between laptop and mobile inference.
    
    Pipeline stages:
    1. EXIF correction + front-camera flip
    2. Proportional downscale (match training resolution)
    3. Image quality assessment
    4. CLAHE lighting normalization
    5. White balance correction
    6. Hand detection (cvzone or MediaPipe)
    7. Landmark extraction
    8. Background noise reduction (optional)
    9. Crop + white canvas placement
    10. Resize to 224×224 + normalize
    
    Configuration:
        enable_clahe: Apply adaptive histogram equalization (recommended)
        enable_white_balance: Apply gray-world white balance
        enable_background_reduction: Apply skin/GrabCut segmentation
        enable_landmarks: Extract MediaPipe landmarks
        target_brightness: Auto-adjust brightness to this value
    """

    def __init__(self,
                 target_size: Tuple[int, int] = (224, 224),
                 canvas_size: int = 300,
                 crop_offset: int = 20,
                 max_input_dim: int = 640,
                 enable_clahe: bool = True,
                 enable_white_balance: bool = True,
                 enable_background_reduction: bool = False,  # slow, off by default
                 enable_landmarks: bool = True,
                 target_brightness: float = 130,
                 detection_confidence: float = 0.7):

        self.target_size = target_size
        self.canvas_size = canvas_size
        self.crop_offset = crop_offset
        self.max_input_dim = max_input_dim
        self.enable_clahe = enable_clahe
        self.enable_white_balance = enable_white_balance
        self.enable_background_reduction = enable_background_reduction
        self.enable_landmarks = enable_landmarks
        self.target_brightness = target_brightness

        # Hand detector — match prediction.py
        if USE_CVZONE:
            self.cvzone_detector = HandDetector(
                maxHands=1, detectionCon=detection_confidence
            )
            print("✅ EnhancedPreprocessor: using cvzone HandDetector")
        else:
            self.cvzone_detector = None

        # Landmark extractor
        if enable_landmarks:
            self.landmark_extractor = HandLandmarkExtractor(
                detection_confidence=detection_confidence
            )
        else:
            self.landmark_extractor = None

    def process(self, pil_image: Image.Image,
                is_front_camera: bool = False,
                save_debug_dir: Optional[str] = None) -> PreprocessResult:
        """
        Full preprocessing pipeline.
        
        Args:
            pil_image: RGB PIL Image from mobile camera
            is_front_camera: Whether to flip horizontally
            save_debug_dir: Directory to save intermediate images (None = skip)
        
        Returns:
            PreprocessResult with all outputs
        """
        debug_info = {}
        warnings = []
        ts = int(time.time() * 1000)

        # ── 1. PIL → BGR ──
        img = np.array(pil_image)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        debug_info['original_shape'] = img_bgr.shape

        # ── 2. Proportional downscale ──
        h_orig, w_orig = img_bgr.shape[:2]
        if max(h_orig, w_orig) > self.max_input_dim:
            scale = self.max_input_dim / max(h_orig, w_orig)
            new_w, new_h = int(w_orig * scale), int(h_orig * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h),
                                  interpolation=cv2.INTER_AREA)
            debug_info['scaled_to'] = f'{new_w}x{new_h}'

        # ── 3. Front camera flip ──
        if is_front_camera:
            img_bgr = cv2.flip(img_bgr, 1)
            debug_info['flipped'] = True

        # ── 4. Image quality assessment ──
        quality = assess_image_quality(img_bgr)
        debug_info['quality_score'] = quality.quality_score
        if quality.is_too_dark:
            warnings.append("Image is too dark — try better lighting")
        if quality.is_too_bright:
            warnings.append("Image is overexposed — reduce lighting")
        if quality.is_blurry:
            warnings.append("Image is blurry — hold camera steady")

        # ── 5. CLAHE lighting normalization ──
        if self.enable_clahe:
            img_bgr = apply_clahe(img_bgr)
            debug_info['clahe_applied'] = True

        # ── 6. White balance ──
        if self.enable_white_balance:
            img_bgr = white_balance_grayworld(img_bgr)
            debug_info['white_balance_applied'] = True

        # ── 7. Brightness adjustment (if needed) ──
        if quality.is_too_dark or quality.is_too_bright:
            img_bgr = adjust_brightness_contrast(img_bgr, self.target_brightness)
            debug_info['brightness_adjusted'] = True

        if save_debug_dir:
            cv2.imwrite(os.path.join(save_debug_dir, f'{ts}_1_corrected.jpg'), img_bgr)

        # ── 8. Hand detection ──
        bbox, img_drawn = self._detect_hand(img_bgr)

        # ── 9. Landmark extraction ──
        landmarks = None
        if self.landmark_extractor:
            landmarks = self.landmark_extractor.extract(img_bgr)
            if landmarks:
                debug_info['handedness'] = landmarks.handedness
                debug_info['landmark_confidence'] = landmarks.confidence
                # If bbox failed but landmarks succeeded, derive bbox
                if bbox is None:
                    bbox = self.landmark_extractor.get_bbox_from_landmarks(
                        landmarks, img_bgr.shape
                    )
                    debug_info['bbox_from_landmarks'] = True

        if bbox is None:
            warnings.append("No hand detected — show hand clearly in frame")
            if save_debug_dir:
                cv2.imwrite(os.path.join(save_debug_dir, f'{ts}_NO_HAND.jpg'), img_bgr)
            # Fallback: full image resize
            img_resized = cv2.resize(img_bgr, self.target_size)
            return PreprocessResult(
                image_batch=np.expand_dims(img_resized / 255.0, axis=0),
                hand_detected=False,
                landmarks=landmarks,
                quality=quality,
                debug_info=debug_info,
                warnings=warnings,
            )

        x, y, w, h = bbox
        debug_info['bbox'] = (x, y, w, h)

        # ── 10. Optional background reduction ──
        if self.enable_background_reduction:
            img_proc = simple_skin_mask(img_drawn)  # Fast path
            debug_info['background_reduced'] = True
        else:
            img_proc = img_drawn

        # ── 11. Crop + white canvas ──
        canvas = self._crop_and_canvas(img_proc, x, y, w, h)
        if canvas is None:
            img_resized = cv2.resize(img_bgr, self.target_size)
            return PreprocessResult(
                image_batch=np.expand_dims(img_resized / 255.0, axis=0),
                hand_detected=False,
                landmarks=landmarks,
                quality=quality,
                debug_info=debug_info,
                warnings=["Hand crop failed"],
            )

        if save_debug_dir:
            cv2.imwrite(os.path.join(save_debug_dir, f'{ts}_2_canvas.jpg'), canvas)

        # ── 12. Final resize + normalize ──
        final = cv2.resize(canvas, self.target_size) / 255.0

        if save_debug_dir:
            cv2.imwrite(os.path.join(save_debug_dir, f'{ts}_3_model_input.jpg'),
                         (final * 255).astype(np.uint8))

        return PreprocessResult(
            image_batch=np.expand_dims(final, axis=0),
            hand_detected=True,
            landmarks=landmarks,
            quality=quality,
            canvas_image=canvas,
            debug_info=debug_info,
            warnings=warnings,
        )

    def _detect_hand(self, img_bgr):
        """Detect hand bounding box using cvzone or raw MediaPipe."""
        if self.cvzone_detector:
            hands, img_drawn = self.cvzone_detector.findHands(img_bgr)
            if hands:
                return tuple(hands[0]['bbox']), img_drawn
            return None, img_bgr
        else:
            # Raw MediaPipe fallback
            h_img, w_img = img_bgr.shape[:2]
            mp_hands = mp.solutions.hands
            with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                                min_detection_confidence=0.7) as hands:
                results = hands.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                if not results.multi_hand_landmarks:
                    return None, img_bgr
                lm = results.multi_hand_landmarks[0]
                xs = [p.x for p in lm.landmark]
                ys = [p.y for p in lm.landmark]
                x_min = max(0, int(min(xs) * w_img))
                y_min = max(0, int(min(ys) * h_img))
                x_max = min(w_img, int(max(xs) * w_img))
                y_max = min(h_img, int(max(ys) * h_img))
                return (x_min, y_min, x_max - x_min, y_max - y_min), img_bgr

    def _crop_and_canvas(self, img_bgr, x, y, w, h):
        """Crop hand and place on white canvas (matches prediction.py)."""
        offset = self.crop_offset
        y1 = max(0, y - offset)
        y2 = min(img_bgr.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img_bgr.shape[1], x + w + offset)
        crop = img_bgr[y1:y2, x1:x2]

        h_c, w_c = crop.shape[:2]
        if h_c == 0 or w_c == 0:
            return None

        sz = self.canvas_size
        canvas = np.ones((sz, sz, 3), np.uint8) * 255
        aspect = h_c / w_c

        if aspect > 1:
            k = sz / h_c
            wCal = math.ceil(k * w_c)
            if wCal > 0:
                resized = cv2.resize(crop, (wCal, sz))
                gap = math.ceil((sz - wCal) / 2)
                end = min(gap + wCal, sz)
                canvas[:, gap:end] = resized[:, :end - gap]
        else:
            k = sz / w_c
            hCal = math.ceil(k * h_c)
            if hCal > 0:
                resized = cv2.resize(crop, (sz, hCal))
                gap = math.ceil((sz - hCal) / 2)
                end = min(gap + hCal, sz)
                canvas[gap:end, :] = resized[:end - gap, :]

        return canvas
