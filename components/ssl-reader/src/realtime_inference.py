"""
Real-time Sign Language Recognition with Emotion Detection.
Uses webcam to capture signs and display predictions with emotional context.

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

from preprocessing_mediapipe import MediaPipeFeatureExtractor
from models import MultimodalLSTMModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealtimeSignLanguageDetector:
    """Real-time sign language detection with emotion recognition."""
    
    def __init__(
        self,
        model_path: str,
        label_map_path: str = None,
        device: str = 'cuda',
        max_frames: int = 60,
        confidence_threshold: float = 0.3,
        buffer_size: int = 60
    ):
        """
        Initialize the real-time detector.
        
        Args:
            model_path: Path to trained model checkpoint (.pth)
            label_map_path: Path to label mapping JSON file
            device: Device to run inference on ('cuda' or 'cpu')
            max_frames: Maximum frames to buffer
            confidence_threshold: Minimum confidence for predictions
            buffer_size: Number of frames to buffer before prediction
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.max_frames = max_frames
        self.confidence_threshold = confidence_threshold
        self.buffer_size = buffer_size
        
        # Initialize MediaPipe feature extractor
        logger.info("Initializing MediaPipe...")
        self.feature_extractor = MediaPipeFeatureExtractor(
            max_frames=max_frames,
            use_hands=True,
            use_face=True,
            use_pose=False
        )
        
        feature_dim = self.feature_extractor.get_feature_dim()
        logger.info(f"Feature dimension: {feature_dim}")
        
        # Load label mapping
        if label_map_path and Path(label_map_path).exists():
            with open(label_map_path, 'r') as f:
                label_data = json.load(f)
                # Handle nested structure {"label_to_idx": {...}}
                if isinstance(label_data, dict) and 'label_to_idx' in label_data:
                    self.label_to_idx = label_data['label_to_idx']
                else:
                    self.label_to_idx = label_data
        else:
            # Try to load from checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            label_data = checkpoint.get('label_to_idx', {})
            # Handle nested structure
            if isinstance(label_data, dict) and 'label_to_idx' in label_data:
                self.label_to_idx = label_data['label_to_idx']
            else:
                self.label_to_idx = label_data
        
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        num_classes = len(self.label_to_idx)
        
        logger.info(f"Number of sign classes: {num_classes}")
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model config from checkpoint or use defaults
        hidden_dim = checkpoint.get('hidden_dim', 512)
        num_layers = checkpoint.get('num_layers', 3)
        
        self.model = MultimodalLSTMModel(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            bidirectional=True
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully on {self.device}")
        
        # Frame buffer for temporal sequences
        self.frame_buffer = deque(maxlen=buffer_size)
        
        # Emotion labels (based on facial blendshapes)
        self.emotion_labels = ['Neutral', 'Happy', 'Sad', 'Angry', 'Surprised', 'Fear']
    
    def extract_emotion_from_blendshapes(self, features):
        """Extract dominant emotion from facial blendshapes."""
        # Blendshapes are the last 52 features
        blendshapes = features[-52:]
        
        # Simple heuristic based on key blendshapes
        # These indices are approximate - adjust based on MediaPipe's blendshape order
        smile_score = blendshapes[0:10].mean() if len(blendshapes) > 10 else 0
        frown_score = blendshapes[10:20].mean() if len(blendshapes) > 20 else 0
        
        if smile_score > 0.3:
            return 'Happy'
        elif frown_score > 0.3:
            return 'Sad'
        else:
            return 'Neutral'
    
    def predict_from_buffer(self):
        """Make prediction from buffered frames."""
        if len(self.frame_buffer) < 10:  # Need minimum frames
            return None, 0.0, 'Neutral'
        
        # Stack frames into sequence
        frames = list(self.frame_buffer)
        
        # Pad or truncate to max_frames
        if len(frames) < self.max_frames:
            # Pad with zeros
            padding = [np.zeros_like(frames[0])] * (self.max_frames - len(frames))
            frames = frames + padding
        else:
            frames = frames[:self.max_frames]
        
        # Convert to tensor
        features = torch.FloatTensor(np.array(frames)).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = probabilities.max(1)
            
            predicted_label = self.idx_to_label.get(predicted_idx.item(), 'Unknown')
            confidence_score = confidence.item()
            
            # Extract emotion from last frame's facial features
            last_frame = frames[-1]
            emotion = self.extract_emotion_from_blendshapes(last_frame)
        
        return predicted_label, confidence_score, emotion
    
    def draw_results(self, frame, prediction, confidence, emotion, fps):
        """Draw prediction results on frame."""
        h, w = frame.shape[:2]
        
        # Create overlay
        overlay = frame.copy()
        
        # Draw semi-transparent background for text
        cv2.rectangle(overlay, (10, 10), (w - 10, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Draw prediction
        if prediction and confidence > self.confidence_threshold:
            text = f"Sign: {prediction}"
            conf_text = f"Confidence: {confidence:.2%}"
            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
        else:
            text = "No sign detected"
            conf_text = "Show a sign..."
            color = (128, 128, 128)
        
        cv2.putText(frame, text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, conf_text, (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw emotion
        emotion_text = f"Emotion: {emotion}"
        emotion_color = (255, 200, 0)
        cv2.putText(frame, emotion_text, (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion_color, 2)
        
        # Draw FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (20, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw instructions
        cv2.putText(frame, "Press 'q' to quit, 'c' to clear buffer", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run_webcam(self, camera_id: int = 0, show_landmarks: bool = True):
        """
        Run real-time detection from webcam.
        
        Args:
            camera_id: Webcam device ID
            show_landmarks: Whether to draw hand/face landmarks
        """
        logger.info(f"Starting webcam {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("Webcam opened successfully")
        logger.info("Press 'q' to quit, 'c' to clear buffer")
        
        prediction = None
        confidence = 0.0
        emotion = 'Neutral'
        fps = 0
        
        # FPS calculation
        prev_time = time.time()
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Extract features from current frame
                features = self.feature_extractor.extract_frame_features(frame)
                
                if features is not None and not np.all(features == 0):
                    # Add to buffer
                    self.frame_buffer.append(features)
                    
                    # Make prediction every few frames
                    if len(self.frame_buffer) >= 30 and frame_count % 10 == 0:
                        prediction, confidence, emotion = self.predict_from_buffer()
                
                # Calculate FPS
                frame_count += 1
                if frame_count % 10 == 0:
                    curr_time = time.time()
                    fps = 10 / (curr_time - prev_time)
                    prev_time = curr_time
                
                # Draw results
                display_frame = self.draw_results(frame, prediction, confidence, emotion, fps)
                
                # Show frame
                cv2.imshow('Sign Language Detection with Emotion Recognition', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.frame_buffer.clear()
                    prediction = None
                    confidence = 0.0
                    emotion = 'Neutral'
                    logger.info("Buffer cleared")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Webcam stopped")


def main():
    parser = argparse.ArgumentParser(
        description='Real-time Sign Language Detection with Emotion Recognition'
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--label_map', type=str, default=None,
                       help='Path to label mapping JSON file')
    parser.add_argument('--camera_id', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Minimum confidence threshold (default: 0.3)')
    parser.add_argument('--buffer_size', type=int, default=60,
                       help='Frame buffer size (default: 60)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = RealtimeSignLanguageDetector(
        model_path=args.model_path,
        label_map_path=args.label_map,
        device=args.device,
        confidence_threshold=args.confidence,
        buffer_size=args.buffer_size
    )
    
    # Run webcam detection
    detector.run_webcam(camera_id=args.camera_id)


if __name__ == '__main__':
    main()
