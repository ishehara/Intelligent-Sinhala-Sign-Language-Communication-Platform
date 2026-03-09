"""
Real-time Sign Language Recognition — MultiStreamFusionModel (457-dim).
Uses webcam to capture signs and display Sinhala + English predictions.

Developer: IT22304674 – Liyanage M.L.I.S.
"""

import torch
import cv2
import numpy as np
import json
from pathlib import Path
import argparse
from collections import deque
import logging
import time

from PIL import Image, ImageDraw, ImageFont
from preprocessing_mediapipe import MediaPipeFeatureExtractor
from models import MultiStreamFusionModel

# ── Sinhala-capable font (Nirmala UI ships with Windows 10/11) ────────────────
_SINHALA_FONT_PATH = "C:/Windows/Fonts/Nirmala.ttc"
try:
    _FONT_LG = ImageFont.truetype(_SINHALA_FONT_PATH, 42)   # Sinhala label
    _FONT_MD = ImageFont.truetype(_SINHALA_FONT_PATH, 26)   # English / confidence
    _FONT_SM = ImageFont.truetype(_SINHALA_FONT_PATH, 20)   # Top-3 list
    _FONT_XS = ImageFont.truetype(_SINHALA_FONT_PATH, 16)   # Status bar
except OSError:
    _FONT_LG = _FONT_MD = _FONT_SM = _FONT_XS = ImageFont.load_default()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealtimeSignLanguageDetector:
    """Real-time sign language detection using MultiStreamFusionModel (457-dim)."""

    def __init__(
        self,
        model_path: str,
        label_map_path: str = None,
        device: str = 'cuda',
        max_frames: int = 60,
        confidence_threshold: float = 0.3,
        buffer_size: int = 60
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.max_frames = max_frames
        self.confidence_threshold = confidence_threshold
        self.buffer_size = buffer_size

        # ── Feature extractor: must match training (457-dim) ──────────────
        logger.info("Initializing MediaPipe (457-dim: hands + filtered face + blendshapes + pose)...")
        self.feature_extractor = MediaPipeFeatureExtractor(
            max_frames=max_frames,
            use_hands=True,
            use_face=True,
            use_filtered_face=True,
            use_blendshapes=True,
            use_pose=True
        )
        logger.info(f"Feature dimension: {self.feature_extractor.get_feature_dim()}")

        # ── Label mapping ─────────────────────────────────────────────────
        checkpoint = torch.load(model_path, map_location=self.device)

        if label_map_path and Path(label_map_path).exists():
            with open(label_map_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
        else:
            raw = checkpoint.get('label_to_idx', {})

        self.label_to_idx = raw.get('label_to_idx', raw) if isinstance(raw, dict) else raw
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        # Sinhala translations (idx string → sinhala string)
        self.idx_to_sinhala = raw.get('idx_to_sinhala', {}) if isinstance(raw, dict) else {}
        num_classes = len(self.label_to_idx)
        logger.info(f"Classes: {num_classes}  |  Sinhala translations: {len(self.idx_to_sinhala)}")

        # ── Build MultiStreamFusionModel matching training config ─────────
        logger.info(f"Loading MultiStreamFusionModel from {model_path}")
        self.model = MultiStreamFusionModel(
            hand_dim=126,
            face_dim=232,   # 60 filtered landmarks × 3 + 52 blendshapes
            pose_dim=99,    # 33 landmarks × 3
            num_classes=num_classes,
            hand_hidden=128,
            face_hidden=256,
            pose_hidden=128,
            fusion_dim=512,
            dropout=0.0,    # no dropout at inference time
            use_pose=True
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model ready on {self.device}")

        # Frame buffer
        self.frame_buffer = deque(maxlen=buffer_size)

        # Feature split dims (must match training config)
        self.hand_dim = 126
        self.face_dim = 232   # 60 landmarks×3 + 52 blendshapes
        self.pose_dim = 99    # 33 landmarks×3

    def _apply_root_relative_norm(self, features: torch.Tensor) -> torch.Tensor:
        """
        Mirror of dataset._apply_root_relative_norm — MUST match exactly.
        Operates on (frames, 457) tensor in-place on a clone.
        """
        features = features.clone()
        frames = features.shape[0]

        # Hands: wrist-relative (landmark 0 of each hand)
        hands = features[:, :126].view(frames, 2, 21, 3)
        wrist = hands[:, :, 0:1, :]
        detected = (wrist.abs().sum(dim=-1, keepdim=True) > 1e-6)
        hands = hands - wrist * detected.float()
        features[:, :126] = hands.view(frames, 126)

        # Face: nose-relative (index 36 of 60 key landmarks)
        face_start = 126
        face_lm = features[:, face_start:face_start + 180].view(frames, 60, 3)
        nose = face_lm[:, 36:37, :]
        detected = (nose.abs().sum(dim=-1, keepdim=True) > 1e-6)
        face_lm = face_lm - nose * detected.float()
        features[:, face_start:face_start + 180] = face_lm.view(frames, 180)
        # Blendshapes (dims 306:358) are left unchanged

        # Pose: mid-shoulder-relative (landmarks 11 + 12)
        pose_start = 126 + 232   # = 358
        pose = features[:, pose_start:pose_start + 99].view(frames, 33, 3)
        mid_shoulder = (pose[:, 11:12, :] + pose[:, 12:13, :]) / 2.0
        detected = (mid_shoulder.abs().sum(dim=-1, keepdim=True) > 1e-6)
        pose = pose - mid_shoulder * detected.float()
        features[:, pose_start:pose_start + 99] = pose.view(frames, 99)

        return features
    
    def predict_from_buffer(self):
        """Make prediction from buffered frames. Returns (label, sinhala, confidence, top3)."""
        if len(self.frame_buffer) < 10:
            return None, None, 0.0, []

        frames = list(self.frame_buffer)

        # Pad or truncate to max_frames
        if len(frames) < self.max_frames:
            padding = [np.zeros_like(frames[0])] * (self.max_frames - len(frames))
            frames = frames + padding
        else:
            frames = frames[:self.max_frames]

        features = torch.FloatTensor(np.array(frames)).unsqueeze(0).to(self.device)

        # Apply root-relative normalization — MUST match dataset.py preprocessing
        features = self._apply_root_relative_norm(features.squeeze(0)).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(features)
            # MultiStreamFusionModel returns (logits, attention_weights)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_idx = probabilities.max(1)

        idx = predicted_idx.item()
        predicted_label = self.idx_to_label.get(idx, 'Unknown')
        sinhala_label = self.idx_to_sinhala.get(str(idx), predicted_label.split('/')[-1])
        confidence_score = confidence.item()

        # Top-3
        top3_probs, top3_idxs = torch.topk(probabilities, min(3, probabilities.shape[1]), dim=1)
        top3 = [
            (self.idx_to_label.get(i.item(), '?'),
             self.idx_to_sinhala.get(str(i.item()), self.idx_to_label.get(i.item(), '?').split('/')[-1]),
             float(p.item()))
            for p, i in zip(top3_probs[0], top3_idxs[0])
        ]

        return predicted_label, sinhala_label, confidence_score, top3
    
    def draw_results(self, frame, label, sinhala, confidence, top3, fps):
        """Draw prediction results using PIL so Sinhala Unicode renders correctly."""
        h, w = frame.shape[:2]

        # Convert BGR (OpenCV) → RGB (PIL)
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")

        # Semi-transparent dark background panel
        draw.rectangle([(10, 10), (w - 10, 210)], fill=(0, 0, 0, 140))

        if label and confidence > self.confidence_threshold:
            color = (0, 255, 80) if confidence > 0.7 else (0, 220, 220)
            # Sinhala label — large
            draw.text((20, 18), sinhala or label, font=_FONT_LG, fill=color)
            # English label
            draw.text((20, 68), label, font=_FONT_MD, fill=color)
            # Confidence bar
            bar_total = w - 40
            bar_fill = int(bar_total * confidence)
            draw.rectangle([(20, 100), (w - 20, 114)], fill=(60, 60, 60, 200))
            draw.rectangle([(20, 100), (20 + bar_fill, 114)], fill=(*color, 220))
            draw.text((20, 118), f"{confidence:.1%}", font=_FONT_SM, fill=color)
        else:
            draw.text((20, 30), "Performing sign...", font=_FONT_MD, fill=(160, 160, 160))

        # Top-3 predictions
        draw.text((20, 148), "Top predictions:", font=_FONT_XS, fill=(200, 200, 200))
        y = 166
        for i, (eng, sin, prob) in enumerate(top3):
            txt = f"  {i+1}. {sin}  {prob:.1%}"
            clr = (160, 255, 160) if i == 0 else (150, 150, 150)
            draw.text((20, y), txt, font=_FONT_XS, fill=clr)
            y += 18

        # Buffer bar + FPS at bottom
        draw.rectangle([(10, h - 28), (w - 10, h - 10)], fill=(0, 0, 0, 120))
        buf_w = int((w - 40) * min(len(self.frame_buffer) / 30, 1.0))
        draw.rectangle([(20, h - 24), (w - 20, h - 14)], fill=(40, 40, 40, 200))
        draw.rectangle([(20, h - 24), (20 + buf_w, h - 14)], fill=(0, 150, 255, 220))
        draw.text((20, h - 34), f"Buffer {len(self.frame_buffer)}/30  FPS {fps:.1f}  [q]uit  [c]lear",
                  font=_FONT_XS, fill=(200, 200, 200))

        # Convert back to BGR (OpenCV)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def run_webcam(self, camera_id: int = 0):
        """Run real-time detection from webcam."""
        logger.info(f"Starting webcam {camera_id}...")
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_id}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        logger.info("Webcam ready. Press 'q' to quit, 'c' to clear buffer.")

        label, sinhala, confidence, top3 = None, None, 0.0, []
        fps = 0.0
        prev_time = time.time()
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break

                # Mirror effect
                frame = cv2.flip(frame, 1)

                # Extract 457-dim features from this frame
                try:
                    features = self.feature_extractor.extract_frame_features(frame)
                except Exception:
                    features = None

                if features is not None and not np.all(features == 0):
                    self.frame_buffer.append(features)

                    # Predict every 10 frames once buffer has ≥30 frames
                    if len(self.frame_buffer) >= 30 and frame_count % 10 == 0:
                        label, sinhala, confidence, top3 = self.predict_from_buffer()

                # FPS
                frame_count += 1
                if frame_count % 10 == 0:
                    curr_time = time.time()
                    fps = 10 / (curr_time - prev_time + 1e-9)
                    prev_time = curr_time

                display = self.draw_results(frame, label, sinhala, confidence, top3, fps)
                cv2.imshow('Sinhala Sign Language Detection', display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.frame_buffer.clear()
                    label, sinhala, confidence, top3 = None, None, 0.0, []
                    logger.info("Buffer cleared")

        except KeyboardInterrupt:
            logger.info("Stopped by user (Ctrl+C)")
        finally:
            try:
                cap.release()
                cv2.destroyAllWindows()
            except Exception:
                pass
            logger.info("Webcam stopped")


def main():
    parser = argparse.ArgumentParser(
        description='Real-time Sinhala Sign Language Detection'
    )
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained checkpoint (.pth)')
    parser.add_argument('--label_map', type=str, default=None,
                        help='Path to label_mapping.json (auto-detected if omitted)')
    parser.add_argument('--camera_id', type=int, default=0,
                        help='Camera device ID (default: 0)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--confidence', type=float, default=0.3,
                        help='Minimum confidence threshold (default: 0.3)')
    parser.add_argument('--buffer_size', type=int, default=60,
                        help='Frame buffer size (default: 60)')

    args = parser.parse_args()

    detector = RealtimeSignLanguageDetector(
        model_path=args.model_path,
        label_map_path=args.label_map,
        device=args.device,
        confidence_threshold=args.confidence,
        buffer_size=args.buffer_size
    )

    detector.run_webcam(camera_id=args.camera_id)


if __name__ == '__main__':
    main()
