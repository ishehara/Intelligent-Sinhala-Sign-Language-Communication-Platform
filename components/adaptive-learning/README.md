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
- **Sign Input**: Captures hands
- **Real-time Processing**: Provides immediate feedback during practice

## 🔬 Technical Approach

### Reinforcement Learning Architecture

```
State Space:
- Current learner proficiency (per sign)
- Historical performance data
- Time since last practice
- Difficulty level preferences

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
│  Gesture Detection | Validation Engine  |                         
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
- **SSL Database**: Static and dyanamic sign data
- **Annotations**: Hand keypoints, facial landmarks, body pose
- **Metadata**: Difficulty level, common errors

### User Data (Collected during usage)
- Learning progress per sign
- Time spent on each lesson
- Error patterns and corrections
- Assessment scores
- Retention test results


## 🔮 Future Enhancements

- **Multi-user Learning**: Compare progress with peers
- **Gamification**: Badges, leaderboards, challenges
- **Advanced RL**: Multi-agent systems for collaborative learning
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
