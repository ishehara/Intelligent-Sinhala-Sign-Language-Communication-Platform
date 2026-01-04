# Smart Sinhala Sign Language Reader with Emotion Recognition

**Developer**: IT22304674 – Liyanage M.L.I.S.

An intelligent SSL recognition system that integrates hand gestures, facial emotions, and body postures with full on-device processing.

## 🎯 Overview

Most SSL recognition systems focus solely on hand gestures, neglecting critical non-manual features like facial expressions and body postures that add meaning and emotional context to communication. This component provides **holistic sign recognition** with **emotion detection**, operating entirely on-device for privacy and real-time performance.

## 🚨 Problem Statement

**How can we design a smart SSL reader that:**
1. Accurately decodes signs by integrating hand gestures, facial emotions, and body postures
2. Detects emotional context to provide richer understanding of sign communication
3. Operates fully on-device for privacy, real-time performance, and culturally relevant communication

## ✨ Key Features

### 1. Multimodal Sign Detection
- **Hand Gesture Recognition**: 21-point hand landmark tracking per hand
- **Dual Hand Support**: Tracks both hands simultaneously (42 landmarks total)
- **Feature Vector**: 126 dimensions (2 hands × 21 landmarks × 3 coordinates)
- **Temporal Analysis**: LSTM and Hybrid models for dynamic sign sequences
- **227 Sign Classes**: Comprehensive Sinhala sign vocabulary
- **Facial Expression Analysis**: 468 facial landmarks for emotion detection
- **Body Pose Detection**: 33 body keypoints for posture and movement
- **Context Integration**: Combines all modalities for accurate interpretation

### 2. Real-Time Emotion Recognition
- **Facial Emotion Detection**: Happiness, sadness, anger, surprise, fear, neutral
- **Expression Intensity**: Measures degree of emotional expression
- **Contextual Emotions**: Urgency, emphasis, questioning
- **Cultural Adaptation**: SSL-specific emotional expressions
- **Confidence Scoring**: Provides reliability metrics for predictions

### 3. On-Device Processing
- **Full Privacy**: No data sent to cloud servers
- **Low Latency**: <500ms from sign to speech
- **Offline Capable**: Works without internet connection
- **Edge AI Models**: Optimized for mobile devices
- **Battery Efficient**: Power-optimized inference

## 🔬 Technical Approach

### Recognition Pipeline

```
Camera Input (30 fps)
         ↓
Multi-Modal Feature Extraction
├─ Hand Landmarks (MediaPipe Hands)
├─ Face Mesh (MediaPipe Face)
└─ Pose Detection (MediaPipe Pose)
         ↓
Feature Fusion & Normalization
         ↓
Temporal Modeling (LSTM/Transformer)
├─ LSTM: 512 hidden units, 3 layers
└─ Hybrid: LSTM + CNN features
         ↓
Sign Classification + Emotion Recognition
         ↓
Sign Output + Emotion Context
```

### Machine Learning Models

1. **Hand Gesture Model**
   - **Base**: MediaPipe Hands (21 landmarks × 2 hands = 126 features)
   - **Architecture Options**:
     - **LSTM Model**: 512 hidden units, 3 layers, dropout 0.3
     - **Hybrid Model**: LSTM + 1D CNN for spatial-temporal features
   - **Output**: 227 sign classes + confidence scores
   - **Features**: Normalized 3D hand coordinates, temporal sequences
   - **Training**: PyTorch 2.7.1 with CUDA 11.8 on NVIDIA GPU
   - **Model Size**: ~4.1M parameters (Hybrid), ~2.9M (LSTM)
   - **Deployment**: TensorFlow Lite for on-device inference

2. **Facial Emotion Model**
   - Base: MediaPipe Face Mesh (468 landmarks)
   - Classifier: CNN or FaceNet-based
   - Output: Emotion class + intensity
   - Features: Key facial regions (eyes, mouth, eyebrows)

3. **Body Pose Model**
   - Base: MediaPipe Pose (33 keypoints)
   - Purpose: Context and sign disambiguation
   - Features: Upper body orientation, shoulder position

4. **Temporal Fusion Model**
   - Architecture: Multi-stream LSTM or Transformer
   - Inputs: Hand features, face features, pose features
   - Output: Final sign prediction + emotional context


## 🏗️ System Architecture

```
┌─────────────────────────────────────────┐
│         Mobile Application UI           │
├─────────────────────────────────────────┤
│  Camera View  │  Sign Display  │ Emotion│
└─────────────────┬───────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│      Multimodal Recognition Engine      │
├─────────────────────────────────────────┤
│  Hand Tracking  │  Face Analysis        │
│  Pose Detection │  Temporal Fusion      │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│        Emotion Analysis Module          │
├─────────────────────────────────────────┤
│  Expression Detector │ Intensity Est.   │
│  Context Classifier  │ Emotion Mapper   │
└─────────────────────────────────────────┘
```

## 📊 Dataset Requirements

### Sign Language Video Dataset
- **Total Classes**: 227+ unique signs
- **Total Videos**: 2,623+ samples
- **Dataset Splits**:
  - Training: 1,729 videos (65.9%)
  - Validation: 293 videos (11.2%)
  - Test: 601 videos (22.9%)
- **Sign Categories**:
  - Adjectives, Adverbs, Colors, Conjunctions
  - Days, Determiners, Greetings, Interjections
  - Months, Nouns, Numbers, People, Places
  - Prepositions, Vehicles, Verbs
- **Static Signs**: 500+ unique signs × 20 samples each
- **Dynamic Signs**: 300+ sequences (phrases, sentences)
- **Annotations**: Hand keypoints, emotion labels, text translations
- **Feature Format**: Cached MediaPipe landmarks (126-dim hands + face + pose)
- **Storage**: Preprocessed features in `.pkl` format

### Emotion-Labeled Data
- **Facial Expressions**: 1000+ samples per emotion category
- **In-Context Emotions**: SSL-specific emotional expressions
- **Intensity Labels**: Low, medium, high emotion intensity


## 🚀 Installation and Setup

### Prerequisites
```bash
Python 3.8+
PyTorch 2.7.1+cu118 (with CUDA 11.8 support)
MediaPipe 0.10.31
TensorFlow >= 2.20.0 (MediaPipe dependency)
TensorFlow Lite 2.x (for mobile deployment)
OpenCV >= 4.8.0
numpy, tqdm, scikit-learn
React Native (for mobile app)
NVIDIA GPU with CUDA 11.8 (recommended for training)
```

### Installation Steps

1. **Clone the repository**
```bash
cd components/ssl-reader
```

2. **Create Python virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install PyTorch with CUDA support**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Verify GPU availability**
```bash
python check_gpu.py
```

6. **Download MediaPipe hand model** (automatic on first run)
```bash
python src/preprocessing_mediapipe.py --help
```

### Configuration

Edit `config/settings.json`:
```json
{
  "camera": {
    "resolution": [1280, 720],
    "fps": 30
  },
  "recognition": {
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.5,
    "temporal_window": 30
  },
  "emotion": {
    "detection_frequency": 5,
    "intensity_threshold": 0.6
  }
}
```

**Training Settings** (command-line arguments):
```bash
--model_type lstm        # or 'hybrid'
--hidden_dim 512         # LSTM hidden units
--num_layers 3           # LSTM layers
--batch_size 16          # Training batch size
--num_epochs 200         # Maximum epochs
--patience 30            # Early stopping patience
--learning_rate 0.001    # Initial learning rate
--device cuda            # Use GPU
```

## 📈 Evaluation Metrics

### Recognition Performance
- **Sign Accuracy**: Correct sign identification rate
- **Emotion Accuracy**: Correct emotion classification rate
- **Temporal Accuracy**: Sequence-level correctness
- **Multi-Modal Fusion**: Improvement from combining modalities


### System Performance
- **Latency**: End-to-end processing time
- **FPS**: Camera processing frame rate
- **Battery Usage**: Power consumption per hour
- **Model Size**: On-device storage requirements

### User Experience
- **Communication Effectiveness**: Success in conveying meaning
- **User Satisfaction**: Overall system rating
- **Ease of Use**: Learning curve and usability
- **Cultural Appropriateness**: SSL and Sinhala cultural fit


## 🐛 Known Issues and Limitations

- Requires good lighting conditions for accurate tracking
- May struggle with rapid or complex finger movements
- Emotion detection less accurate with partial face visibility
- Limited to pre-trained sign vocabulary
- Performance varies across device capabilities

## 📚 References

### Computer Vision
- MediaPipe Hands, Face, and Pose
- LSTM and Transformer architectures
- Multi-modal learning techniques

### Emotion Recognition
- Facial Action Coding System (FACS)
- Deep learning for emotion detection
- Affective computing principles

### Sign Language
- Sinhala Sign Language linguistics
- Non-manual features in sign languages
- Cultural aspects of SSL

## 🤝 Contributing

Contributions welcome in:
- Sign language dataset expansion
- Emotion recognition improvements
- Model optimization for mobile
- Cultural validation and testing



## 👤 Developer

**Liyanage M.L.I.S.** (IT22304674)
- Email: [mlisliyanage@gmail.com]
- Focus: Computer Vision, Emotion Recognition

---

## 📂 Project Structure

```
ssl-reader/
├── src/
│   ├── preprocessing_mediapipe.py  # MediaPipe feature extraction
│   ├── preprocess_mediapipe.py     # Batch preprocessing script
│   ├── train_mediapipe.py          # Training script
│   ├── dataset.py                  # PyTorch dataset loader
│   ├── models.py                   # LSTM and Hybrid models
│   └── preprocessing_simple.py     # Dataset splitting
├── data/
│   └── processed/
│       └── mediapipe_full/         # Cached features (.pkl)
├── models/
│   └── mediapipe/
│       ├── hand_landmarker.task    # MediaPipe model
│       ├── checkpoint_best.pth     # Best trained model
│       └── checkpoint_latest.pth   # Latest checkpoint
├── logs/                           # Training logs
├── datasets/                       # Video datasets
├── requirements.txt                # Python dependencies
├── check_gpu.py                    # GPU verification
└── README.md                       # This file
```

---

**Component Status**: Active Development | **Version**: 1.0.0 | **Last Updated**: January 2026
