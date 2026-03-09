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

import tempfile
from preprocessing_mediapipe import MediaPipeFeatureExtractor
from models import MultiStreamFusionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React Native

# Global variables
model = None
feature_extractor = None
label_to_idx = None
idx_to_label = None
idx_to_sinhala = None
model_lock = Lock()
frame_buffer = []
MAX_FRAMES = 60


def load_model(model_path: str, label_map_path: str = None):
    """Load the trained model and label mapping."""
    global model, feature_extractor, label_to_idx, idx_to_label, idx_to_sinhala

    logger.info(f"Loading model from {model_path}")

    # Resolve label mapping: checkpoint doesn't embed it, so load from JSON
    if label_map_path is None:
        # Auto-detect: datasets/label_mapping.json relative to workspace root
        model_dir = Path(model_path).resolve().parent
        candidate = model_dir.parent.parent.parent / 'datasets' / 'label_mapping.json'
        if candidate.exists():
            label_map_path = str(candidate)
        else:
            raise FileNotFoundError(
                f"label_mapping.json not found near {model_path}. "
                "Pass --label_map explicitly."
            )

    with open(label_map_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    # Support both flat {"label": idx} and nested {"label_to_idx": {"label": idx}}
    label_to_idx = raw.get('label_to_idx', raw)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    idx_to_sinhala = raw.get('idx_to_sinhala', {})
    num_classes = len(label_to_idx)
    logger.info(f"Loaded {num_classes} labels from {label_map_path}")

    # Feature extractor: 457-dim (hands 126 + face 232 + pose 99)
    feature_extractor = MediaPipeFeatureExtractor(
        max_frames=MAX_FRAMES,
        use_hands=True,
        use_pose=True,
        use_face=True,
        use_filtered_face=True,
        use_blendshapes=True
    )

    # Model: MultiStreamFusionModel matching train_mediapipe.py defaults
    checkpoint = torch.load(model_path, map_location='cpu')
    model = MultiStreamFusionModel(
        hand_dim=126,
        face_dim=232,
        pose_dim=99,
        num_classes=num_classes,
        hand_hidden=128,
        face_hidden=256,
        pose_hidden=128,
        fusion_dim=512,
        dropout=0.0,
        use_pose=True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Model loaded: {num_classes} classes, 457-dim features")


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

        # Extract features using the Tasks API extractor (handles BGR→RGB internally)
        frame_features = feature_extractor.extract_frame_features(frame)
        
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
                    # MultiStreamFusionModel returns (logits, attention_weights)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    probabilities = torch.softmax(logits, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, dim=1)

            predicted_label = idx_to_label[predicted_idx.item()]
            confidence_score = confidence.item()
            sinhala_label = idx_to_sinhala.get(str(predicted_idx.item()), predicted_label.split('/')[-1])

            # Get top 5 predictions
            top5_probs, top5_indices = torch.topk(probabilities, min(5, len(idx_to_label)), dim=1)
            top5_predictions = [
                {
                    'label': idx_to_label[idx.item()],
                    'sinhala': idx_to_sinhala.get(str(idx.item()), idx_to_label[idx.item()].split('/')[-1]),
                    'confidence': float(prob.item())
                }
                for prob, idx in zip(top5_probs[0], top5_indices[0])
            ]

            # Clear buffer
            frame_buffer = []

            return jsonify({
                'success': True,
                'predicted_label': predicted_label,
                'sinhala_label': sinhala_label,
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
        
        # Decode video and save to a temporary file (cross-platform)
        video_bytes = base64.b64decode(video_base64)
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(video_bytes)
            temp_path = tmp.name

        # Extract features
        features = feature_extractor.process_video(temp_path)
        
        if features is None:
            return jsonify({'error': 'Failed to process video'}), 500
        
        # Predict
        features_tensor = torch.FloatTensor(features).unsqueeze(0)

        with model_lock:
            with torch.no_grad():
                outputs = model(features_tensor)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                probabilities = torch.softmax(logits, dim=1)
                confidence, predicted_idx = torch.max(probabilities, dim=1)
        
        predicted_label = idx_to_label[predicted_idx.item()]
        confidence_score = confidence.item()
        sinhala_label = idx_to_sinhala.get(str(predicted_idx.item()), predicted_label.split('/')[-1])

        # Get top 5
        top5_probs, top5_indices = torch.topk(probabilities, min(5, len(idx_to_label)), dim=1)
        top5_predictions = [
            {
                'label': idx_to_label[idx.item()],
                'sinhala': idx_to_sinhala.get(str(idx.item()), idx_to_label[idx.item()].split('/')[-1]),
                'confidence': float(prob.item())
            }
            for prob, idx in zip(top5_probs[0], top5_indices[0])
        ]

        # Clean up
        Path(temp_path).unlink(missing_ok=True)

        return jsonify({
            'success': True,
            'predicted_label': predicted_label,
            'sinhala_label': sinhala_label,
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
                       help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--label_map', type=str, default=None,
                       help='Path to label_mapping.json (auto-detected if omitted)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to bind to')
    
    args = parser.parse_args()
    
    # Load model
    load_model(args.model_path, args.label_map)
    
    # Start server
    logger.info(f"Starting API server on {args.host}:{args.port}")
    logger.info("Ready for React Native connections!")
    
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
