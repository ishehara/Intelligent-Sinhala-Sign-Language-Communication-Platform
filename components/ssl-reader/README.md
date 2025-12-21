# Smart Sinhala Sign Language Reader with Emotion Recognition

**Developer**: IT22304674 – Liyanage M.L.I.S.

An intelligent SSL recognition system that integrates hand gestures, facial emotions, and body postures to provide natural, emotion-aware Sinhala speech synthesis with full on-device processing.

## 🎯 Overview

Most SSL recognition systems focus solely on hand gestures, neglecting critical non-manual features like facial expressions and body postures that add meaning and emotional context to communication. This component provides **holistic sign recognition** with **emotion-aware speech synthesis**, operating entirely on-device for privacy and real-time performance.

## 🚨 Problem Statement

**How can we design a smart SSL reader that:**
1. Accurately decodes signs by integrating hand gestures, facial emotions, and body postures
2. Synthesizes natural, emotion-aware Sinhala speech reflecting the signer's affective state
3. Operates fully on-device for privacy, real-time performance, and culturally relevant communication

## ✨ Key Features

### 1. Multimodal Sign Detection
- **Hand Gesture Recognition**: 21-point hand landmark tracking per hand
- **Facial Expression Analysis**: 468 facial landmarks for emotion detection
- **Body Pose Detection**: 33 body keypoints for posture and movement
- **Temporal Analysis**: LSTM/Transformer for dynamic sign sequences
- **Context Integration**: Combines all modalities for accurate interpretation

### 2. Real-Time Emotion Recognition
- **Facial Emotion Detection**: Happiness, sadness, anger, surprise, fear, neutral
- **Expression Intensity**: Measures degree of emotional expression
- **Contextual Emotions**: Urgency, emphasis, questioning
- **Cultural Adaptation**: Considers SSL-specific emotional expressions
- **Confidence Scoring**: Provides reliability metrics for predictions

### 3. Emotion-Aware Speech Synthesis
- **Expressive TTS**: Modulates voice tone, pitch, and speed based on emotion
- **Natural Sinhala Voice**: High-quality, culturally appropriate speech
- **Emotion Mapping**: Happy→cheerful voice, Sad→softer tone, Urgent→faster pace
- **Prosody Control**: Stress, intonation, and rhythm adaptation
- **Non-Robotic Output**: Reduces mechanical speech patterns

### 4. On-Device Processing
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
Feature Normalization & Preprocessing
         ↓
Temporal Modeling (LSTM/Transformer)
         ↓
Sign Classification + Emotion Recognition
         ↓
Emotion-Aware TTS Generation
         ↓
Sinhala Speech Output
```

### Machine Learning Models

1. **Hand Gesture Model**
   - Base: MediaPipe Hands (21 landmarks × 2 hands)
   - Classifier: LSTM + Attention or Transformer
   - Output: Sign class + confidence
   - Features: Normalized coordinates, hand shape, movement

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

5. **Emotion-to-Speech Mapper**
   - Input: Emotion class, intensity, sign text
   - Output: SSML (Speech Synthesis Markup Language)
   - Controls: Pitch, rate, volume, emphasis

## 🏗️ System Architecture

```
┌─────────────────────────────────────────┐
│         Mobile Application UI           │
├─────────────────────────────────────────┤
│  Camera View  │  Text Display  │ Audio  │
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
└─────────────────┬───────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│      Emotion-Aware TTS Engine           │
├─────────────────────────────────────────┤
│  Text Generation  │  SSML Generator     │
│  Voice Synthesis  │  Prosody Control    │
└─────────────────────────────────────────┘
```

## 📊 Dataset Requirements

### Sign Language Video Dataset
- **Static Signs**: 500+ unique signs × 20 samples each
- **Dynamic Signs**: 300+ sequences (phrases, sentences)
- **Annotations**: Hand keypoints, emotion labels, text translations
- **Variations**: Different signers, lighting, backgrounds

### Emotion-Labeled Data
- **Facial Expressions**: 1000+ samples per emotion category
- **In-Context Emotions**: SSL-specific emotional expressions
- **Intensity Labels**: Low, medium, high emotion intensity

### Speech Synthesis Data
- **Sinhala Voice Corpus**: Native speaker recordings
- **Emotional Speech**: Samples with different emotional tones
- **Prosody Annotations**: Pitch, duration, emphasis markers

## 🚀 Installation and Setup

### Prerequisites
```bash
Python 3.8+
TensorFlow Lite 2.x
MediaPipe 0.10+
Pyttsx3 or Google TTS
React Native
OpenCV
```

### Installation Steps

1. **Clone the component**
```bash
cd components/ssl-reader
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install MediaPipe**
```bash
pip install mediapipe
```

4. **Install mobile dependencies**
```bash
npm install
react-native link react-native-camera
react-native link react-native-tts
```

5. **Download models**
```bash
python scripts/download_models.py
```

6. **Set up TTS engine**
```bash
# For Android
npm install react-native-tts

# For iOS (additional setup in Xcode)
cd ios && pod install
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
  },
  "tts": {
    "language": "si-LK",
    "speech_rate": 1.0,
    "pitch": 1.0
  }
}
```

## 💻 Usage

### Basic Sign Recognition

```python
from src.recognizer import MultimodalSignRecognizer
from src.emotion import EmotionDetector
from src.tts import EmotionAwareTTS

# Initialize components
recognizer = MultimodalSignRecognizer(
    model_path='models/sign_classifier.tflite'
)
emotion_detector = EmotionDetector(
    model_path='models/emotion_classifier.tflite'
)
tts_engine = EmotionAwareTTS(language='si-LK')

# Process video frame
result = recognizer.process_frame(frame)

if result['confidence'] > 0.7:
    # Detect emotion
    emotion = emotion_detector.detect(result['face_landmarks'])
    
    # Generate speech with emotion
    text = result['sign_text']
    speech_params = tts_engine.generate_params(
        text=text,
        emotion=emotion['class'],
        intensity=emotion['intensity']
    )
    
    # Speak with emotion
    tts_engine.speak(text, params=speech_params)
    
    print(f"Sign: {text}, Emotion: {emotion['class']}")
```

### Real-Time Video Processing

```python
from src.video_processor import RealtimeSSLProcessor

processor = RealtimeSSLProcessor(
    camera_index=0,
    display_landmarks=True
)

# Start processing
processor.start()

while processor.is_running():
    # Get current recognition result
    result = processor.get_latest_result()
    
    if result:
        print(f"Recognized: {result['text']}")
        print(f"Emotion: {result['emotion']}")
        print(f"Confidence: {result['confidence']:.2f}")
```

### Training Custom Models

```python
from src.training import SignClassifierTrainer
from src.data_loader import SSLVideoDataset

# Load dataset
dataset = SSLVideoDataset(
    video_dir='datasets/ssl_videos',
    annotations='datasets/annotations.json'
)

# Initialize trainer
trainer = SignClassifierTrainer(
    model_architecture='lstm_attention',
    num_classes=500
)

# Train model
history = trainer.train(
    dataset=dataset,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)

# Export to TFLite for mobile
trainer.export_tflite('models/sign_classifier.tflite')
```

## 📱 Mobile Integration

### React Native Example

```javascript
import { SSLReader } from './src/native-modules';

const SSLReaderComponent = () => {
  const [recognizedText, setRecognizedText] = useState('');
  const [emotion, setEmotion] = useState('neutral');

  useEffect(() => {
    SSLReader.initialize({
      enableEmotionDetection: true,
      enableOnDeviceProcessing: true
    });

    SSLReader.addEventListener('signRecognized', (event) => {
      setRecognizedText(event.text);
      setEmotion(event.emotion);
      
      // Trigger TTS with emotion
      SSLReader.speakWithEmotion(
        event.text,
        event.emotion,
        event.intensity
      );
    });

    return () => SSLReader.cleanup();
  }, []);

  return (
    <View>
      <CameraView />
      <Text>Recognized: {recognizedText}</Text>
      <EmotionIndicator emotion={emotion} />
    </View>
  );
};
```

## 📈 Evaluation Metrics

### Recognition Performance
- **Sign Accuracy**: Correct sign identification rate
- **Emotion Accuracy**: Correct emotion classification rate
- **Temporal Accuracy**: Sequence-level correctness
- **Multi-Modal Fusion**: Improvement from combining modalities

### Speech Quality
- **Naturalness**: Mean Opinion Score (MOS)
- **Emotion Appropriateness**: User ratings on emotion-speech match
- **Intelligibility**: Word recognition rate
- **Prosody Quality**: Pitch and rhythm naturalness

### System Performance
- **Latency**: Time from sign to speech output
- **FPS**: Camera processing frame rate
- **Battery Usage**: Power consumption per hour
- **Model Size**: On-device storage requirements

### User Experience
- **Communication Effectiveness**: Success in conveying meaning
- **User Satisfaction**: Overall system rating
- **Ease of Use**: Learning curve and usability
- **Cultural Appropriateness**: SSL and Sinhala cultural fit

## 🎨 Emotion-to-Speech Mapping

### Speech Parameter Adjustments

| Emotion | Pitch | Rate | Volume | Emphasis | Example |
|---------|-------|------|--------|----------|---------|
| Happy | +20% | +10% | +5% | High | Cheerful, bright |
| Sad | -20% | -15% | -10% | Low | Soft, gentle |
| Angry | +15% | +5% | +15% | High | Intense, sharp |
| Surprise | +25% | +20% | +10% | High | Excited, quick |
| Fear | +10% | +15% | 0% | Medium | Tense, hurried |
| Urgent | +10% | +25% | +10% | High | Fast, pressing |
| Neutral | 0% | 0% | 0% | Medium | Normal speech |

## 🔮 Future Enhancements

- **Gesture Vocabulary Expansion**: Support for 1000+ signs
- **Conversational Context**: Multi-turn dialogue understanding
- **Person-Specific Adaptation**: Learn individual signing styles
- **3D Hand Pose**: More accurate spatial understanding
- **AR Visualization**: Overlay sign guidance in real-time
- **Multi-Person Detection**: Recognize multiple signers
- **Custom Voice Models**: Personalized TTS voices

## 🐛 Known Issues and Limitations

- Requires good lighting conditions for accurate tracking
- May struggle with rapid or complex finger movements
- Emotion detection less accurate with partial face visibility
- Limited to pre-trained sign vocabulary
- TTS quality varies with emotion intensity extremes
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

### Speech Synthesis
- SSML (Speech Synthesis Markup Language)
- Prosody modeling in TTS
- Emotional speech synthesis

### Sign Language
- Sinhala Sign Language linguistics
- Non-manual features in sign languages
- Cultural aspects of SSL

## 🤝 Contributing

Contributions welcome in:
- Sign language dataset expansion
- Emotion recognition improvements
- TTS quality enhancements
- Model optimization for mobile
- Cultural validation and testing

## 📝 License

MIT License - See main project LICENSE file

## 👤 Developer

**Liyanage M.L.I.S.** (IT22304674)
- Email: [developer-email]
- Focus: Computer Vision, Emotion Recognition, Speech Synthesis

---

**Component Status**: Active Development | **Version**: 0.1.0 | **Last Updated**: December 2025
