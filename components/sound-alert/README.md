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
- **High Accuracy**: ML-based classification evaluated using standard metrics
- **Real-time Processing**: Low-latency detection (<500ms)

### 2. Continuous Critical Sound Monitoring
- **Fire Alarms**: Multiple alarm types and patterns
- **Emergency Sirens**: Ambulance, police, fire truck

### 3. Urgency-Based Prioritization
- **Smart Scoring System**: Assigns urgency levels to different sounds
 - **Context Awareness**: Time of day, sound duration
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
   - Architecture: CNN-based audio classification models
   - Input: Audio waveform or Mel-spectrogram
   - Output: Vehicle type + confidence score
   - Classes: Car, Bus, Train, Motorcycle, Truck

2. **General Sound Detection Model**
   - Architecture: Pre-trained or CNN-based audio classifier
   - Input: Short audio clips (1-3 seconds)
   - Output: Sound category + probability
   - Classes: Fire alarm, Siren etc.

3. **Urgency Prediction Model**
   - Features: Sound type, volume, temporal context
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
│  Urgency Model    │  Context Analyzer   │
└─────────────────────────────────────────┘
```

## 📊 Dataset Requirements
### Vehicle Horn Dataset
- **Car Horns**: various models and recording conditions
- **Bus Horns**: city buses, long-distance
- **Train Horns**: passenger, freight, crossing signals
- **Motorcycle Horns**: various engine sizes
- **Truck Horns**: light, heavy trucks

### Critical Sound Dataset
- **Fire Alarms**: samples various types, buildings
- **Sirens**: ambulance, police, fire

### Data Augmentation
- Background noise injection
- Volume variation
- Speed variation
- Pitch shifting
- Time stretching

```
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
 
### User Experience
- **Alert Relevance**: User-reported usefulness
- **Response Time**: User reaction speed
- **Battery Impact**: Power consumption metrics
- **User Satisfaction**: Feedback and ratings

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

## 👤 Developer

**Kodithuwakku M.A.S.S.H.** (IT22325464)

---

**Component Status**: Active Development | **Version**: 0.1.0 | **Last Updated**: January 2025
