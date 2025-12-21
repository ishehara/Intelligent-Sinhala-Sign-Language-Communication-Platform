# Environmental Sound Alert Module

**Developer**: IT22325464 – Kodithuwakku M.A.S.S.H.

A real-time environmental sound detection and classification system that provides context-aware alerts for Deaf and hard-of-hearing users, with specialized vehicle horn classification for enhanced situational awareness.

## 🎯 Overview

Existing environmental sound alert systems identify basic sounds (alarms, horns) but lack granular classification. Understanding whether a horn belongs to a car, bus, train, or other vehicle significantly enhances context-awareness and safety. This component provides **detailed vehicle horn classification** combined with continuous ambient sound monitoring.

## 🚨 Problem Statement

**How can we design a mobile environmental sound alert system that:**
1. Accurately detects horn sounds and classifies the type of vehicle producing them
2. Continuously listens to critical ambient sounds (fire alarms, sirens)
3. Prioritizes alerts based on urgency for timely notifications
4. Provides real-time, context-rich alerts that improve user awareness and decision-making

## ✨ Key Features

### 1. Vehicle Horn Recognition & Classification
- **Multi-Vehicle Detection**: Car, bus, train, motorcycle, truck horns
- **Audio Pattern Analysis**: Frequency, amplitude, and temporal feature extraction
- **High Accuracy**: ML-based classification with >90% accuracy
- **Real-time Processing**: Low-latency detection (<500ms)

### 2. Continuous Critical Sound Monitoring
- **Fire Alarms**: Multiple alarm types and patterns
- **Emergency Sirens**: Ambulance, police, fire truck
- **Loudspeakers**: Public announcements and warnings
- **Doorbell/Knocking**: Home security and visitor alerts

### 3. Urgency-Based Prioritization
- **Smart Scoring System**: Assigns urgency levels to different sounds
- **Distance Estimation**: Approximate proximity of sound source
- **Context Awareness**: Time of day, location, user activity
- **Alert Filtering**: Prevents notification fatigue

### 4. Multi-Modal Alert System
- **Vibration Patterns**: Different intensities for different sounds
- **Screen Flash**: Attention-grabbing visual indicator
- **On-Screen Banners**: Detailed sound information
- **Emoji Animations**: Quick visual representation of sound type
- **Haptic Feedback**: Tactile patterns for urgent alerts

## 🔬 Technical Approach

### Sound Classification Pipeline

```
Audio Input (Microphone)
         ↓
Pre-processing (Noise Reduction, Normalization)
         ↓
Feature Extraction (MFCC, Spectrogram, Mel-scale)
         ↓
Classification Model (CNN/RNN)
         ↓
Post-processing (Confidence Filtering, Smoothing)
         ↓
Urgency Assessment
         ↓
Multi-Modal Alert Dispatch
```

### Machine Learning Models

1. **Horn Classification Model**
   - Architecture: 1D-CNN + LSTM or 2D-CNN (Spectrogram-based)
   - Input: Audio waveform or Mel-spectrogram
   - Output: Vehicle type + confidence score
   - Classes: Car, Bus, Train, Motorcycle, Truck, Other

2. **General Sound Detection Model**
   - Architecture: VGGish-based or ResNet audio classifier
   - Input: Short audio clips (1-3 seconds)
   - Output: Sound category + probability
   - Classes: Fire alarm, Siren, Loudspeaker, Doorbell, etc.

3. **Urgency Prediction Model**
   - Features: Sound type, volume, proximity, context
   - Output: Urgency score (0-100)
   - Algorithm: Decision tree or rule-based system

## 🏗️ System Architecture

```
┌─────────────────────────────────────────┐
│         Mobile Application UI           │
├─────────────────────────────────────────┤
│  Alert Display  │  Settings  │  History │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│        Alert Management Layer           │
├─────────────────────────────────────────┤
│  Prioritization  │  Notification        │
│  Context Engine  │  Alert Dispatcher    │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│       Audio Processing Pipeline         │
├─────────────────────────────────────────┤
│  Audio Capture  │  Feature Extraction   │
│  Preprocessing  │  Model Inference      │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│         Classification Models           │
├─────────────────────────────────────────┤
│  Horn Classifier  │  Sound Detector     │
│  Urgency Model    │  Distance Estimator │
└─────────────────────────────────────────┘
```

## 📊 Dataset Requirements

### Vehicle Horn Dataset
- **Car Horns**: 1000+ samples (various models, distances)
- **Bus Horns**: 500+ samples (city buses, long-distance)
- **Train Horns**: 300+ samples (passenger, freight, crossing signals)
- **Motorcycle Horns**: 500+ samples (various engine sizes)
- **Truck Horns**: 400+ samples (light, heavy trucks)

### Critical Sound Dataset
- **Fire Alarms**: 500+ samples (various types, buildings)
- **Sirens**: 800+ samples (ambulance, police, fire)
- **Loudspeakers**: 400+ samples (announcements, warnings)
- **Doorbells**: 300+ samples

### Data Augmentation
- Background noise injection
- Volume variation
- Speed variation
- Pitch shifting
- Time stretching

## 🚀 Installation and Setup

### Prerequisites
```bash
Python 3.8+
TensorFlow 2.x / PyTorch 1.x
Librosa
NumPy, SciPy
React Native
```

### Installation Steps

1. **Clone the component**
```bash
cd components/sound-alert
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install mobile dependencies**
```bash
npm install
react-native link react-native-audio
```

4. **Download pre-trained models**
```bash
python scripts/download_models.py
```

5. **Set up audio permissions** (Android: AndroidManifest.xml, iOS: Info.plist)

### Configuration

Edit `config/settings.json`:
```json
{
  "audio": {
    "sample_rate": 44100,
    "chunk_duration": 2.0,
    "overlap": 0.5
  },
  "detection": {
    "confidence_threshold": 0.75,
    "horn_threshold": 0.80,
    "alert_cooldown": 5
  },
  "alerts": {
    "vibration_enabled": true,
    "flash_enabled": true,
    "sound_history_limit": 100
  }
}
```

## 💻 Usage

### Real-Time Sound Detection

```python
from src.audio_processor import AudioProcessor
from src.models import HornClassifier, SoundDetector

# Initialize processors
audio_proc = AudioProcessor(sample_rate=44100)
horn_classifier = HornClassifier('models/horn_classifier.h5')
sound_detector = SoundDetector('models/sound_detector.h5')

# Start continuous monitoring
audio_proc.start_stream()

while True:
    # Capture audio chunk
    audio_chunk = audio_proc.get_next_chunk()
    
    # Extract features
    features = audio_proc.extract_features(audio_chunk)
    
    # Classify
    horn_result = horn_classifier.predict(features)
    sound_result = sound_detector.predict(features)
    
    # Process results
    if horn_result['confidence'] > 0.80:
        print(f"Horn detected: {horn_result['vehicle_type']}")
        alert_dispatcher.send_alert(horn_result)
```

### Training Custom Models

```python
from src.training import ModelTrainer
from src.data_loader import AudioDataLoader

# Load training data
data_loader = AudioDataLoader('datasets/vehicle_horns')
train_data, val_data = data_loader.load_and_split(test_size=0.2)

# Initialize trainer
trainer = ModelTrainer(
    model_type='horn_classifier',
    architecture='cnn_lstm'
)

# Train model
history = trainer.train(
    train_data=train_data,
    val_data=val_data,
    epochs=50,
    batch_size=32
)

# Save model
trainer.save_model('models/horn_classifier_v2.h5')
```

### Urgency Assessment

```python
from src.urgency_engine import UrgencyAssessor

urgency = UrgencyAssessor()

# Assess sound urgency
urgency_score = urgency.calculate(
    sound_type='fire_alarm',
    volume=85,  # dB
    distance=10,  # meters (estimated)
    time_of_day='night',
    location='indoor'
)

print(f"Urgency Score: {urgency_score}/100")
```

## 📱 Mobile Integration

### React Native Example

```javascript
import { SoundAlertModule } from './src/native-modules';

// Start monitoring
SoundAlertModule.startMonitoring({
  enableHornDetection: true,
  enableCriticalSounds: true,
  confidenceThreshold: 0.75
});

// Listen for alerts
SoundAlertModule.addEventListener('soundDetected', (event) => {
  const { soundType, vehicleType, urgency, timestamp } = event;
  
  // Display alert
  showAlert({
    type: soundType,
    vehicle: vehicleType,
    urgency: urgency,
    time: timestamp
  });
  
  // Trigger vibration
  if (urgency > 70) {
    Vibration.vibrate([0, 500, 200, 500]);
  }
});
```

## 📈 Evaluation Metrics

### Classification Performance
- **Accuracy**: Overall correct classification rate
- **Precision/Recall**: Per-class metrics
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error analysis

### Detection Performance
- **True Positive Rate**: Percentage of correctly detected sounds
- **False Positive Rate**: Incorrect alert frequency
- **Detection Latency**: Time from sound occurrence to alert
- **Distance Accuracy**: Proximity estimation error

### User Experience
- **Alert Relevance**: User-reported usefulness
- **Response Time**: User reaction speed
- **Battery Impact**: Power consumption metrics
- **User Satisfaction**: Feedback and ratings

## 🎯 Alert Prioritization Logic

### Urgency Scoring

| Sound Type | Base Score | Distance Factor | Time Factor | Final Range |
|------------|------------|-----------------|-------------|-------------|
| Fire Alarm | 95 | -2/meter | +10 (night) | 85-100 |
| Ambulance Siren | 90 | -3/meter | +5 (night) | 75-95 |
| Train Horn | 85 | -1/meter | 0 | 70-90 |
| Truck Horn | 75 | -2/meter | 0 | 60-80 |
| Bus Horn | 70 | -2/meter | 0 | 55-75 |
| Car Horn | 65 | -2/meter | 0 | 50-70 |
| Doorbell | 60 | 0 | -10 (away) | 40-60 |

## 🔮 Future Enhancements

- **Direction Detection**: Indicate where sound is coming from
- **Multi-Sound Handling**: Simultaneous detection of multiple sounds
- **Personalized Alerts**: Learn user preferences over time
- **Indoor/Outdoor Classification**: Context-aware detection
- **Smart Watch Integration**: Wearable device support
- **Cloud Sync**: Share alert history across devices
- **Offline Mode**: On-device processing without internet

## 🐛 Known Issues and Limitations

- Background noise can affect accuracy in very noisy environments
- Battery drain with continuous monitoring (optimizations in progress)
- May have difficulty with very distant or muffled sounds
- Limited to pre-trained sound categories
- False positives with similar-sounding non-target sounds

## 📚 References

### Audio Processing
- Librosa: Python library for audio analysis
- MFCC (Mel-Frequency Cepstral Coefficients)
- Mel-Spectrogram analysis

### Machine Learning
- CNN for audio classification
- LSTM for temporal pattern recognition
- Transfer learning with VGGish

### Datasets
- UrbanSound8K
- ESC-50 (Environmental Sound Classification)
- FSDKaggle2018

## 🤝 Contributing

Contributions welcome in:
- New sound categories and datasets
- Model architecture improvements
- Alert system enhancements
- Battery optimization
- UI/UX improvements

## 📝 License

MIT License - See main project LICENSE file

## 👤 Developer

**Kodithuwakku M.A.S.S.H.** (IT22325464)
- Email: [developer-email]
- Focus: Audio Classification, Real-time Processing, Mobile Development

---

**Component Status**: Active Development | **Version**: 0.1.0 | **Last Updated**: December 2025
