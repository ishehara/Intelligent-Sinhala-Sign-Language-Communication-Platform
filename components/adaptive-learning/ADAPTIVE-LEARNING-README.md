# Adaptive Sinhala Sign Language Learning System

**Developer**: IT22315878 – Silva H.T.P.

An intelligent, AI-powered mobile learning platform that adapts to individual learners' needs using Reinforcement Learning to provide personalized Sinhala Sign Language education.

## 🎯 Overview

Traditional SSL learning systems provide static lessons without adapting to individual learning pace, leading to frustration (too difficult) or disengagement (too easy). This component introduces the **first Reinforcement Learning-based adaptive learning system** for Sinhala Sign Language that dynamically adjusts to each learner's strengths, weaknesses, and progress.

## 🚨 Problem Statement

**How can we design a mobile learning platform for Sinhala Sign Language that:**
1. Adapts lessons to each learner's strengths and weaknesses
2. Provides real-time corrective feedback using AI-driven recognition
3. Ensures long-term sign retention through AI-guided review scheduling

## ✨ Key Features

### 1. Personalized Lesson Path
- **RL-Based Selection**: Dynamically selects signs and phrases based on learner history
- **Difficulty Adjustment**: Automatically increases or decreases complexity
- **Progress Tracking**: Monitors mastery levels for each sign and concept
- **Custom Learning Pace**: Adapts speed to individual learning style

### 2. Adaptive Feedback Mechanism
- **Granular Corrections**: Detailed feedback (e.g., "hand tilted left") for struggling learners
- **Encouragement Mode**: Positive reinforcement for confident learners
- **Real-time Recognition**: AI-driven sign verification using camera input
- **Contextual Hints**: Provides tips based on common mistakes

### 3. AI-Guided Spaced Repetition
- **Forgetting Curve Prediction**: Identifies when learner is likely to forget
- **Smart Review Scheduling**: Revisits old signs at optimal intervals
- **Long-term Retention**: Ensures signs move from short-term to long-term memory
- **Priority Ranking**: Focuses on signs most at risk of being forgotten

### 4. Holistic Sign Recognition
- **Multi-modal Input**: Captures hands, body posture, and facial expressions
- **Cultural Accuracy**: Recognizes SSL-specific movements and expressions
- **Context Understanding**: Distinguishes between similar signs based on context
- **Real-time Processing**: Provides immediate feedback during practice

## 🔬 Technical Approach

### Reinforcement Learning Architecture

```
State Space:
- Current learner proficiency (per sign)
- Historical performance data
- Time since last practice
- Difficulty level preferences
- Error patterns

Action Space:
- Select next sign to teach
- Adjust difficulty level
- Modify feedback detail level
- Schedule review sessions

Reward Function:
- Success rate on assessments
- Retention over time
- Engagement metrics
- Progress speed
```

### Machine Learning Models

1. **Sign Recognition Model**
   - Architecture: MediaPipe + LSTM/Transformer
   - Input: Video frames (hands, face, body)
   - Output: Predicted sign + confidence score

2. **RL Agent**
   - Algorithm: Deep Q-Network (DQN) / Proximal Policy Optimization (PPO)
   - Purpose: Lesson sequencing and difficulty adjustment
   - Training: User interaction data and learning outcomes

3. **Spaced Repetition Model**
   - Algorithm: Modified Leitner System with ML predictions
   - Features: Time elapsed, past performance, sign complexity
   - Output: Review priority scores

## 🏗️ System Architecture

```
┌─────────────────────────────────────────┐
│        Mobile Application Layer         │
├─────────────────────────────────────────┤
│  UI Components  │  Camera  │  Storage   │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│         RL Learning Engine              │
├─────────────────────────────────────────┤
│  State Manager  │  Action Selector     │
│  Reward Calculator │ Policy Network    │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│       Sign Recognition Module           │
├─────────────────────────────────────────┤
│  Gesture Detection  │  Emotion Analysis │
│  Body Pose  │  Validation Engine        │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│        Data Management Layer            │
├─────────────────────────────────────────┤
│  User Profiles  │  Progress DB          │
│  Sign Library  │  Performance Metrics   │
└─────────────────────────────────────────┘
```

## 📊 Dataset Requirements

### Sign Language Data
- **SSL Video Database**: 500+ signs with multiple variations
- **Annotations**: Hand keypoints, facial landmarks, body pose
- **Metadata**: Difficulty level, common errors, cultural context

### User Data (Collected during usage)
- Learning progress per sign
- Time spent on each lesson
- Error patterns and corrections
- Assessment scores
- Retention test results

## 🚀 Installation and Setup

### Prerequisites
```bash
Python 3.8+
TensorFlow 2.x or PyTorch 1.x
OpenCV
MediaPipe
React Native
```

### Installation Steps

1. **Clone the component**
```bash
cd components/adaptive-learning
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install mobile dependencies**
```bash
npm install
```

4. **Download pre-trained models**
```bash
python scripts/download_models.py
```

5. **Set up database**
```bash
python scripts/init_database.py
```

### Configuration

Edit `config/settings.json`:
```json
{
  "rl_agent": {
    "algorithm": "DQN",
    "learning_rate": 0.001,
    "epsilon": 0.1
  },
  "recognition": {
    "confidence_threshold": 0.85,
    "fps": 30
  },
  "spaced_repetition": {
    "initial_interval": 1,
    "multiplier": 2.5
  }
}
```

## 💻 Usage

### Training the RL Agent

```python
from src.rl_engine import AdaptiveLearningAgent
from src.data_loader import UserDataLoader

# Initialize agent
agent = AdaptiveLearningAgent(
    state_size=128,
    action_size=100,  # Number of possible signs
    learning_rate=0.001
)

# Load user data
data_loader = UserDataLoader('data/user_profiles')

# Train agent
agent.train(
    episodes=1000,
    user_data=data_loader.get_training_data()
)

# Save trained model
agent.save('models/rl_agent.h5')
```

### Running Sign Recognition

```python
from src.recognition import SignRecognizer

recognizer = SignRecognizer(
    model_path='models/sign_recognition.h5'
)

# Process video input
result = recognizer.recognize_sign(video_frame)
print(f"Detected: {result['sign']}, Confidence: {result['confidence']}")
```

### Implementing Adaptive Lesson

```python
from src.lesson_manager import AdaptiveLessonManager

lesson_mgr = AdaptiveLessonManager(
    user_id='user123',
    rl_agent=agent
)

# Get next recommended sign
next_sign = lesson_mgr.get_next_lesson()

# Record practice session
lesson_mgr.record_practice(
    sign_id=next_sign,
    success=True,
    time_taken=45.5,
    errors=['hand_angle']
)

# Check review schedule
reviews = lesson_mgr.get_review_signs()
```

## 📈 Evaluation Metrics

### Learning Effectiveness
- **Retention Rate**: Percentage of signs remembered after 1 week, 1 month
- **Learning Speed**: Time to mastery per sign
- **Error Reduction**: Decrease in mistakes over time

### System Performance
- **Recognition Accuracy**: Precision, recall, F1-score
- **RL Convergence**: Reward progression over episodes
- **Adaptation Quality**: Learner satisfaction scores

### User Experience
- **Engagement**: Session duration, frequency
- **Satisfaction**: SUS (System Usability Scale) scores
- **Perceived Difficulty**: Self-reported challenge level

## 🔮 Future Enhancements

- **Multi-user Learning**: Compare progress with peers
- **Gamification**: Badges, leaderboards, challenges
- **Advanced RL**: Multi-agent systems for collaborative learning
- **AR Integration**: Augmented reality for immersive practice
- **Voice Feedback**: Audio instructions and corrections
- **Parent/Teacher Dashboard**: Monitor learner progress

## 🐛 Known Issues and Limitations

- Requires good lighting for camera-based recognition
- Initial cold-start problem for new users (minimal history)
- Limited to static signs (dynamic signs in development)
- Performance varies with device camera quality

## 📚 References

### Related Work
- SignAll (2021). Ace ASL - AI-based ASL learning platform
- Paudyal et al. (2020). SignGuru - Automated sign language tutor
- Reinforcement Learning for Education: Review and applications

### Technical Documentation
- MediaPipe Hands Documentation
- Deep Q-Network (DQN) Algorithm
- Spaced Repetition Systems

## 🤝 Contributing

Contributions are welcome! Focus areas:
- New sign language datasets
- RL algorithm improvements
- Recognition model optimization
- UX/UI enhancements
- Performance optimization

## 📝 License

MIT License - See main project LICENSE file

## 👤 Developer

**Silva H.T.P.** (IT22315878)
- Email: [developer-email]
- Focus: Reinforcement Learning, Adaptive Systems, Sign Language Recognition

---

**Component Status**: Active Development | **Version**: 0.1.0 | **Last Updated**: December 2025
