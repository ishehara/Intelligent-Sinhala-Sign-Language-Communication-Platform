"""
Video Sign Language Predictor
==============================
Handles dynamic (video-based) sign language detection.
The mobile app sends individual frames; this module accumulates
hand-landmark sequences per user session and runs the LSTM model
when enough frames have been collected.

Model: sign_language_model.h5  (sequence-based, 30 frames × 42 features)
Labels: label_encoder.npy  (v1-v22 → Sinhala letters)
"""

import numpy as np
import tensorflow as tf
import threading
import os
from collections import deque, Counter
from datetime import datetime

try:
    from cvzone.HandTrackingModule import HandDetector
    _USE_CVZONE = True
except ImportError:
    import mediapipe as mp
    _USE_CVZONE = False

# ── Mapping from class labels to Sinhala letters ──────────────
CLASS_TO_SINHALA = {
    'v1': 'ඈ', 'v2': 'ඊ', 'v3': 'ඌ', 'v4': 'ඒ', 'v5': 'ඔ',
    'v6': 'ඕ', 'v7': 'ජ', 'v8': 'ණ', 'v9': 'ළ', 'v10': 'ඟ',
    'v11': 'ඦ', 'v12': 'ඳ', 'v13': 'ඹ', 'v14': 'ඛ', 'v15': 'ඬ',
    'v16': 'ඵ', 'v17': 'ධ', 'v18': 'ඨ', 'v19': 'ඡ', 'v20': 'ක්\u200dය',
    'v21': 'භ', 'v22': 'ථ',
}

# All dynamic sign labels (Sinhala letters)
VIDEO_SIGN_LABELS = list(CLASS_TO_SINHALA.values())

# ── Constants ─────────────────────────────────────────────────
MAX_SEQUENCE_LENGTH = 30
FEATURE_DIM = 42          # 21 landmarks × 2 (x, y)
PREDICTION_THRESHOLD = 0.7


class VideoSignPredictor:
    """Manages per-session frame buffers and runs the video sign model."""

    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(__file__), '..', 'model', 'VedioModel'
            )

        model_path = os.path.join(model_dir, 'sign_language_model.h5')
        label_path = os.path.join(model_dir, 'label_encoder.npy')

        print(f"📹 Loading video sign model from {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        self.label_classes = np.load(label_path, allow_pickle=True)
        print(f"📹 Video model loaded. Classes: {self.label_classes}")

        # Per-session frame buffers  { session_id: deque(...) }
        self._buffers = {}
        self._lock = threading.Lock()

        # Hand detector for extracting landmarks from frames
        if _USE_CVZONE:
            self._detector = HandDetector(maxHands=1, detectionCon=0.8)
        else:
            _mp_hands = mp.solutions.hands
            self._detector = _mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.8,
            )

    # ── Public API ────────────────────────────────────────────

    def process_frame(self, img_bgr, session_id="default"):
        """
        Process a single BGR frame: extract landmarks, append to buffer,
        and predict if enough frames are available.

        Returns dict with prediction info.
        """
        landmarks = self._extract_landmarks(img_bgr)
        hand_detected = landmarks is not None

        with self._lock:
            buf = self._buffers.setdefault(session_id, deque(maxlen=MAX_SEQUENCE_LENGTH))
            if hand_detected:
                normalized = self._normalize_landmarks(landmarks)
                buf.append(normalized)

            buffer_size = len(buf)

        # Need at least 10 frames to attempt prediction
        if buffer_size < 10 or not hand_detected:
            return {
                "prediction": None,
                "confidence": 0.0,
                "hand_detected": hand_detected,
                "buffer_size": buffer_size,
                "status": "buffering",
            }

        # Run prediction
        with self._lock:
            seq = list(buf)
        predicted_label, confidence = self._predict_sequence(seq)

        if predicted_label is None:
            return {
                "prediction": None,
                "confidence": float(confidence),
                "hand_detected": True,
                "buffer_size": buffer_size,
                "status": "low_confidence",
            }

        return {
            "prediction": predicted_label,
            "confidence": float(confidence),
            "hand_detected": True,
            "buffer_size": buffer_size,
            "status": "predicted",
        }

    def clear_buffer(self, session_id="default"):
        """Clear the frame buffer for a session."""
        with self._lock:
            if session_id in self._buffers:
                self._buffers[session_id].clear()

    def get_labels(self):
        """Return the list of dynamic sign labels (Sinhala letters)."""
        return VIDEO_SIGN_LABELS

    # ── Internal Methods ──────────────────────────────────────

    def _extract_landmarks(self, img_bgr):
        """Extract 21 hand landmarks (x, y) from a BGR image. Returns flat array or None."""
        if _USE_CVZONE:
            hands, _ = self._detector.findHands(img_bgr.copy(), draw=False)
            if not hands:
                return None
            lmList = hands[0]['lmList']
            return np.array([[lm[0], lm[1]] for lm in lmList]).flatten()
        else:
            import cv2
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = self._detector.process(img_rgb)
            if not results.multi_hand_landmarks:
                return None
            lm = results.multi_hand_landmarks[0]
            h, w = img_bgr.shape[:2]
            coords = []
            for p in lm.landmark:
                coords.append(p.x * w)
                coords.append(p.y * h)
            return np.array(coords)

    @staticmethod
    def _normalize_landmarks(landmarks):
        """Normalize landmarks relative to wrist position."""
        landmarks_2d = landmarks.reshape(-1, 2)
        wrist = landmarks_2d[0]
        normalized = landmarks_2d - wrist
        middle_tip = landmarks_2d[12]
        hand_size = np.linalg.norm(middle_tip - wrist)
        if hand_size > 0:
            normalized = normalized / hand_size
        return normalized.flatten()

    def _predict_sequence(self, sequence):
        """Predict sign from a list of landmark arrays."""
        seq_array = np.array(sequence)
        if len(seq_array) < MAX_SEQUENCE_LENGTH:
            padding = np.zeros((MAX_SEQUENCE_LENGTH - len(seq_array), FEATURE_DIM))
            seq_array = np.vstack([seq_array, padding])

        seq_array = seq_array.reshape(1, MAX_SEQUENCE_LENGTH, FEATURE_DIM)
        prediction = self.model.predict(seq_array, verbose=0)

        predicted_idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        if confidence < PREDICTION_THRESHOLD:
            return None, confidence

        class_label = self.label_classes[predicted_idx]
        sinhala_letter = CLASS_TO_SINHALA.get(str(class_label), str(class_label))
        return sinhala_letter, confidence
