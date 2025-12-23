"""
Inference script for Sinhala Sign Language Recognition.
Supports real-time video and webcam inference.

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

from preprocessing import VideoFeatureExtractor
from models import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignLanguageInference:
    """Real-time inference for sign language recognition."""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        max_frames: int = 60,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on
            max_frames: Maximum frames to process
            confidence_threshold: Minimum confidence for predictions
        """
        self.device = device
        self.max_frames = max_frames
        self.confidence_threshold = confidence_threshold
        
        # Load checkpoint
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Load label mapping
        self.label_to_idx = checkpoint.get('label_to_idx', {})
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        num_classes = len(self.label_to_idx)
        
        # Initialize feature extractor
        self.feature_extractor = VideoFeatureExtractor(max_frames=max_frames)
        input_dim = self.feature_extractor.get_feature_dim()
        
        # Create and load model
        # Try to infer model type from checkpoint or default to LSTM
        model_config = checkpoint.get('model_config', {
            'model_type': 'lstm',
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.3
        })
        
        self.model = create_model(
            model_type=model_config.get('model_type', 'lstm'),
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=model_config.get('hidden_dim', 256),
            num_layers=model_config.get('num_layers', 2),
            dropout=model_config.get('dropout', 0.3)
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully")
        logger.info(f"Number of classes: {num_classes}")
        logger.info(f"Device: {device}")
    
    def predict_video(self, video_path: str) -> dict:
        """
        Predict sign language from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with prediction results
        """
        # Extract features
        features = self.feature_extractor.process_video(video_path)
        
        if features is None:
            return {
                'success': False,
                'error': 'Failed to extract features from video'
            }
        
        # Convert to tensor and add batch dimension
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
        
        predicted_label = self.idx_to_label[predicted_idx.item()]
        confidence_score = confidence.item()
        
        # Get top-5 predictions
        top5_probs, top5_indices = torch.topk(probabilities, min(5, len(self.idx_to_label)), dim=1)
        top5_predictions = [
            {
                'label': self.idx_to_label[idx.item()],
                'confidence': prob.item()
            }
            for prob, idx in zip(top5_probs[0], top5_indices[0])
        ]
        
        return {
            'success': True,
            'predicted_label': predicted_label,
            'confidence': confidence_score,
            'top5_predictions': top5_predictions,
            'is_confident': confidence_score >= self.confidence_threshold
        }
    
    def predict_webcam(self, camera_id: int = 0, window_size: int = 60):
        """
        Real-time prediction from webcam.
        
        Args:
            camera_id: Camera device ID
            window_size: Number of frames to collect before prediction
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_id}")
            return
        
        logger.info("Starting webcam inference. Press 'q' to quit, 'r' to reset buffer")
        
        frame_buffer = deque(maxlen=window_size)
        is_recording = False
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Display frame
            display_frame = frame.copy()
            
            # Add recording indicator
            if is_recording:
                cv2.putText(display_frame, f"Recording: {len(frame_buffer)}/{window_size}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "Press SPACE to start recording",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Process frame if recording
            if is_recording:
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = self.feature_extractor.holistic.process(frame_rgb)
                
                # Extract landmarks
                landmarks = self.feature_extractor.extract_landmarks(results)
                
                # Concatenate features
                frame_features = np.concatenate([
                    landmarks['left_hand'],
                    landmarks['right_hand'],
                    landmarks['face'],
                    landmarks['pose']
                ])
                
                frame_buffer.append(frame_features)
                
                # Predict when buffer is full
                if len(frame_buffer) == window_size:
                    features_array = np.array(list(frame_buffer))
                    features_tensor = torch.FloatTensor(features_array).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(features_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        confidence, predicted_idx = torch.max(probabilities, dim=1)
                    
                    predicted_label = self.idx_to_label[predicted_idx.item()]
                    confidence_score = confidence.item()
                    
                    # Display prediction
                    color = (0, 255, 0) if confidence_score >= self.confidence_threshold else (0, 165, 255)
                    cv2.putText(display_frame, f"Prediction: {predicted_label}",
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(display_frame, f"Confidence: {confidence_score:.2f}",
                               (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Auto-reset after prediction
                    is_recording = False
                    frame_buffer.clear()
            
            cv2.imshow('Sign Language Recognition', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space to start/stop recording
                if not is_recording:
                    is_recording = True
                    frame_buffer.clear()
                else:
                    is_recording = False
                    frame_buffer.clear()
            elif key == ord('r'):  # Reset buffer
                frame_buffer.clear()
                is_recording = False
        
        cap.release()
        cv2.destroyAllWindows()
        self.feature_extractor.close()
    
    def batch_predict(self, video_dir: str, output_file: str = None):
        """
        Predict on a batch of videos.
        
        Args:
            video_dir: Directory containing video files
            output_file: Optional file to save results
        """
        video_dir = Path(video_dir)
        video_files = list(video_dir.glob('*.mp4'))
        
        logger.info(f"Found {len(video_files)} videos in {video_dir}")
        
        results = []
        
        for video_file in video_files:
            logger.info(f"Processing {video_file.name}")
            result = self.predict_video(str(video_file))
            result['video_file'] = video_file.name
            results.append(result)
            
            if result['success']:
                logger.info(f"  Predicted: {result['predicted_label']} "
                           f"(confidence: {result['confidence']:.3f})")
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Sinhala Sign Language Recognition Inference')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--mode', type=str, default='video',
                       choices=['video', 'webcam', 'batch'],
                       help='Inference mode')
    parser.add_argument('--video_path', type=str,
                       help='Path to input video (for video mode)')
    parser.add_argument('--video_dir', type=str,
                       help='Directory of videos (for batch mode)')
    parser.add_argument('--output_file', type=str,
                       help='Output file for batch predictions')
    parser.add_argument('--camera_id', type=int, default=0,
                       help='Camera device ID (for webcam mode)')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for inference')
    parser.add_argument('--max_frames', type=int, default=60,
                       help='Maximum frames to process')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for predictions')
    
    args = parser.parse_args()
    
    # Create inference engine
    inference = SignLanguageInference(
        model_path=args.model_path,
        device=args.device,
        max_frames=args.max_frames,
        confidence_threshold=args.confidence_threshold
    )
    
    # Run inference based on mode
    if args.mode == 'video':
        if not args.video_path:
            parser.error("--video_path is required for video mode")
        
        result = inference.predict_video(args.video_path)
        
        if result['success']:
            print(f"\n{'='*50}")
            print(f"Predicted Sign: {result['predicted_label']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"\nTop 5 Predictions:")
            for i, pred in enumerate(result['top5_predictions'], 1):
                print(f"  {i}. {pred['label']}: {pred['confidence']:.3f}")
            print(f"{'='*50}\n")
        else:
            print(f"Error: {result['error']}")
    
    elif args.mode == 'webcam':
        inference.predict_webcam(camera_id=args.camera_id)
    
    elif args.mode == 'batch':
        if not args.video_dir:
            parser.error("--video_dir is required for batch mode")
        
        results = inference.batch_predict(args.video_dir, args.output_file)
        
        # Print summary
        successful = sum(1 for r in results if r['success'])
        print(f"\nProcessed {len(results)} videos ({successful} successful)")


if __name__ == "__main__":
    main()
