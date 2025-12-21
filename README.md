# Inclusive Communication Platform for Sinhala Sign Language

A comprehensive mobile platform that leverages AI and machine learning to bridge communication gaps for the Deaf and hard-of-hearing community in Sri Lanka through adaptive learning, environmental awareness, intelligent translation, and family-centered communication.

## 🎯 Project Overview

This research project addresses critical gaps in Sinhala Sign Language (SSL) education and communication by developing four interconnected components that work together to create an accessible, intelligent, and culturally relevant communication ecosystem.

### Key Innovations

- **AI-Powered Adaptive Learning**: First SSL learning system using Reinforcement Learning for personalized education
- **Context-Aware Environmental Alerts**: Vehicle-specific horn classification and priority-based sound notifications
- **Emotion-Integrated Recognition**: Holistic SSL reader combining gestures, emotions, and expressive speech synthesis
- **Family-Centered Communication**: Two-way visual communication system designed for children and parents

## 📦 Components

### 1. Adaptive Sinhala Sign Language Learning System
**Developer**: IT22315878 – Silva H.T.P.

An intelligent mobile learning platform that adapts to individual learner needs using Reinforcement Learning.

**Key Features**:
- Personalized lesson paths based on learner history
- Dynamic difficulty adjustment
- AI-guided spaced repetition for long-term retention
- Adaptive feedback mechanisms

**Novelty**: First Sinhala sign language app using RL for personalized adaptive learning

[📁 Component Documentation]([./components/adaptive-learning/README.md](https://github.com/ishehara/Intelligent-Sinhala-Sign-Language-Communication-Platform/blob/main/components/adaptive-learning/README.md?plain=1))

---

### 2. Environmental Sound Alert Module
**Developer**: IT22325464 – Kodithuwakku M.A.S.S.H.

A real-time sound detection and classification system that provides context-rich environmental awareness.

**Key Features**:
- Vehicle horn classification (car, bus, train, motorcycle, truck)
- Continuous critical sound monitoring (fire alarms, sirens, loudspeakers)
- Urgency-based alert prioritization
- Multi-modal notifications (vibration, screen flash, banners, emojis)

**Novelty**: Granular vehicle horn classification for enhanced situational awareness

[📁 Component Documentation](./components/sound-alert/README.md)

---

### 3. Smart Sinhala Sign Language Reader with Emotion Recognition
**Developer**: IT22304674 – Liyanage M.L.I.S.

An intelligent SSL reader that integrates gestures, emotions, and context for natural communication.

**Key Features**:
- Multimodal sign detection (hands, face, body posture)
- Real-time emotion recognition from facial cues
- Emotion-aware Sinhala speech synthesis
- Fully on-device processing for privacy and low latency

**Novelty**: First SSL system integrating emotional context with expressive speech output

[📁 Component Documentation](./components/ssl-reader/README.md)

---

### 4. Two-Way Communication System for Children
**Developer**: IT22308870 – De Silva U.P.A.N.

An HCI-based visual communication platform enabling natural interaction between children with speaking disabilities and their families.

**Key Features**:
- Text/Touch-to-Sign conversion with animations
- Emotion and feedback layer with visual cues
- Offline functionality for rural accessibility
- Icon-based communication for daily needs
- Culturally adapted interaction methods

**Novelty**: First two-way SSL communication system designed for family-centered daily interaction

[📁 Component Documentation](./components/two-way-comm/README.md)

---

## 🏗️ Project Structure

```
├── components/
│   ├── adaptive-learning/     # RL-based adaptive learning system
│   │   ├── src/
│   │   ├── models/
│   │   ├── datasets/
│   │   └── README.md
│   │
│   ├── sound-alert/          # Environmental sound classification
│   │   ├── src/
│   │   ├── models/
│   │   ├── datasets/
│   │   └── README.md
│   │
│   ├── ssl-reader/           # Emotion-aware SSL recognition
│   │   ├── src/
│   │   ├── models/
│   │   ├── datasets/
│   │   └── README.md
│   │
│   └── two-way-comm/         # 
│       ├── src/
│       ├── assets/
│       ├── animations/
│       └── README.md
│
├── docs/                     # Project documentation
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Node.js 14+ (for mobile development)
- TensorFlow/PyTorch
- React Native (for mobile app)
- Android Studio / Xcode

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-org/ssl-communication-platform.git
cd ssl-communication-platform
```

2. **Set up Python environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Install mobile dependencies**
```bash
npm install
cd android && ./gradlew clean  # For Android
cd ios && pod install          # For iOS
```

4. **Set up individual components**

Each component has its own setup requirements. Refer to individual component READMEs for detailed instructions.

## 🔬 Research Methodology

### Machine Learning Approaches

- **Reinforcement Learning**: Q-learning/DQN for adaptive lesson selection
- **Deep Learning**: CNN/LSTM for gesture and sign recognition
- **Audio Classification**: Spectrogram analysis for sound detection
- **Emotion Recognition**: Facial landmark detection and expression analysis
- **Speech Synthesis**: Emotion-aware TTS for Sinhala language

### Data Collection

- Sinhala Sign Language video datasets
- Environmental sound recordings (vehicle horns, alarms, sirens)
- Facial expression and emotion datasets
- User interaction and learning progress data

### Evaluation Metrics

- Recognition accuracy (signs, sounds, emotions)
- Learning effectiveness (retention rates, progress speed)
- System usability (SUS scores, user feedback)
- Real-time performance (latency, response time)
- Accessibility metrics (rural vs urban usage)

## 🎯 Target Audience

- **Deaf and Hard-of-Hearing Community**: Primary users for learning, communication, and environmental awareness
- **Children with Speaking Disabilities**: Users of the two-way communication system
- **Families and Caregivers**: Supporting communication and learning
- **Educators and Trainers**: SSL teaching and curriculum development
- **General Public**: Promoting SSL awareness and inclusive communication

## 🌟 Impact and Significance

### Social Impact
- Breaks down communication barriers for marginalized communities
- Provides accessible education regardless of location or resources
- Empowers families to communicate naturally with children
- Enhances safety and independence through environmental awareness

### Technical Contributions
- First RL-based adaptive learning system for SSL
- Novel vehicle-specific sound classification for accessibility
- Integration of emotion recognition with sign language
- Privacy-preserving on-device processing
- Offline-first design for rural accessibility

### Cultural Relevance
- Tailored specifically for Sinhala Sign Language
- Considers Sri Lankan cultural context and expressions
- Addresses local infrastructure challenges
- Designed for low-resource environments

## 🤝 Contributing

We welcome contributions from researchers, developers, and community members! Please read our contributing guidelines for details on:

- Code style and standards
- Dataset contribution protocols
- Testing requirements
- Documentation standards
- Pull request process

## 👥 Team

- **Silva H.T.P.** (IT22315878) - Adaptive Learning System
- **Kodithuwakku M.A.S.S.H.** (IT22325464) - Environmental Sound Alert
- **Liyanage M.L.I.S.** (IT22304674) - SSL Reader with Emotion Recognition
- **De Silva U.P.A.N.** (IT22308870) - Two-Way Communication System


## 🙏 Acknowledgments

- Sri Lankan Deaf community for insights and testing
- Sign language experts and linguists
- Research advisors and mentors
- Funding organizations and institutions

