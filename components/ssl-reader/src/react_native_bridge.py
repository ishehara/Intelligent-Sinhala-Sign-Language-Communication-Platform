"""
React Native Bridge API Server for Sinhala Sign Language Recognition.
Provides real-time inference API for React Native mobile app.

Developer: IT22304674 – Liyanage M.L.I.S.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import base64
import cv2
import logging
from pathlib import Path
import json
from threading import Lock

from preprocessing import VideoFeatureExtractor
from models import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React Native

# Global variables
model = None
feature_extractor = None
label_to_idx = None
idx_to_label = None
model_lock = Lock()
frame_buffer = []
MAX_FRAMES = 60


def load_model(model_path: str):
    """Load the trained model."""
    global model, feature_extractor, label_to_idx, idx_to_label
    
    logger.info(f"Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    label_to_idx = checkpoint.get('label_to_idx', {})
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    
    # Initialize feature extractor
    feature_extractor = VideoFeatureExtractor(max_frames=MAX_FRAMES)
    input_dim = feature_extractor.get_feature_dim()
    num_classes = len(label_to_idx)
    
    # Create model
    model_config = checkpoint.get('model_config', {
        'model_type': 'lstm',
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.0
    })
    
    model = create_model(
        model_type=model_config.get('model_type', 'lstm'),
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=model_config.get('hidden_dim', 256),
        num_layers=model_config.get('num_layers', 2),
        dropout=0.0
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Model loaded: {num_classes} classes")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'num_classes': len(label_to_idx) if label_to_idx else 0
    })


@app.route('/labels', methods=['GET'])
def get_labels():
    """Get all available labels."""
    if label_to_idx is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'labels': list(label_to_idx.keys()),
        'count': len(label_to_idx)
    })


@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    """
    Process a single frame from React Native.
    Accumulates frames and returns prediction when buffer is full.
    """
    global frame_buffer
    
    try:
        data = request.get_json()
        
        # Get base64 encoded frame
        frame_base64 = data.get('frame')
        if not frame_base64:
            return jsonify({'error': 'No frame provided'}), 400
        
        # Decode frame
        frame_bytes = base64.b64decode(frame_base64)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid frame data'}), 400
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract features using MediaPipe
        results = feature_extractor.holistic.process(frame_rgb)
        landmarks = feature_extractor.extract_landmarks(results)
        
        # Concatenate features
        frame_features = np.concatenate([
            landmarks['left_hand'],
            landmarks['right_hand'],
            landmarks['face'],
            landmarks['pose']
        ])
        
        # Add to buffer
        frame_buffer.append(frame_features)
        
        # Check if buffer is full
        if len(frame_buffer) >= MAX_FRAMES:
            # Prepare features
            features_array = np.array(frame_buffer[-MAX_FRAMES:])
            features_tensor = torch.FloatTensor(features_array).unsqueeze(0)
            
            # Predict
            with model_lock:
                with torch.no_grad():
                    outputs = model(features_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, dim=1)
            
            predicted_label = idx_to_label[predicted_idx.item()]
            confidence_score = confidence.item()
            
            # Get top 5 predictions
            top5_probs, top5_indices = torch.topk(probabilities, min(5, len(idx_to_label)), dim=1)
            top5_predictions = [
                {
                    'label': idx_to_label[idx.item()],
                    'confidence': float(prob.item())
                }
                for prob, idx in zip(top5_probs[0], top5_indices[0])
            ]
            
            # Clear buffer
            frame_buffer = []
            
            return jsonify({
                'success': True,
                'predicted_label': predicted_label,
                'confidence': float(confidence_score),
                'top5_predictions': top5_predictions,
                'buffer_full': True
            })
        
        else:
            # Buffer not full yet
            return jsonify({
                'success': True,
                'buffer_count': len(frame_buffer),
                'buffer_max': MAX_FRAMES,
                'buffer_full': False
            })
    
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict_video', methods=['POST'])
def predict_video():
    """
    Predict sign language from uploaded video.
    For batch processing from React Native.
    """
    try:
        data = request.get_json()
        video_base64 = data.get('video')
        
        if not video_base64:
            return jsonify({'error': 'No video provided'}), 400
        
        # Decode video
        video_bytes = base64.b64decode(video_base64)
        
        # Save temporarily
        temp_path = '/tmp/temp_video.mp4'
        with open(temp_path, 'wb') as f:
            f.write(video_bytes)
        
        # Extract features
        features = feature_extractor.process_video(temp_path)
        
        if features is None:
            return jsonify({'error': 'Failed to process video'}), 500
        
        # Predict
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        with model_lock:
            with torch.no_grad():
                outputs = model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, dim=1)
        
        predicted_label = idx_to_label[predicted_idx.item()]
        confidence_score = confidence.item()
        
        # Get top 5
        top5_probs, top5_indices = torch.topk(probabilities, min(5, len(idx_to_label)), dim=1)
        top5_predictions = [
            {
                'label': idx_to_label[idx.item()],
                'confidence': float(prob.item())
            }
            for prob, idx in zip(top5_probs[0], top5_indices[0])
        ]
        
        # Clean up
        Path(temp_path).unlink(missing_ok=True)
        
        return jsonify({
            'success': True,
            'predicted_label': predicted_label,
            'confidence': float(confidence_score),
            'top5_predictions': top5_predictions
        })
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/reset_buffer', methods=['POST'])
def reset_buffer():
    """Reset the frame buffer."""
    global frame_buffer
    frame_buffer = []
    return jsonify({'success': True, 'message': 'Buffer reset'})


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='React Native Bridge API Server')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to bind to')
    
    args = parser.parse_args()
    
    # Load model
    load_model(args.model_path)
    
    # Start server
    logger.info(f"Starting API server on {args.host}:{args.port}")
    logger.info("Ready for React Native connections!")
    
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
