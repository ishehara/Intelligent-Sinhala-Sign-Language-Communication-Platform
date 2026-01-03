# Two-Way Communication System for Children with Speaking Disabilities

**Developer**: IT22308870 – De Silva U.P.A.N.

An HCI-based visual communication platform that enables natural, silent, and emotion-aware two-way interaction between children with speaking disabilities and their families, addressing the challenges of learning and using Sinhala Sign Language in daily life.

## 🎯 Overview

In Sri Lanka, children with speaking disabilities face significant communication barriers. While Sinhala Sign Language exists, many children cannot learn it due to limited training facilities, financial constraints, and family members' unfamiliarity with SSL. This component provides a **practical, family-centered communication solution** that requires no prior SSL knowledge, using intuitive visual interfaces for daily interaction.

## 🚨 Problem Statement

**How can we design a Sinhala two-way communication system for children with speaking disabilities that:**
1. Enables them to express daily needs and emotions through sign gestures, icons, or visual selections
2. Allows parents to respond through simple text, sign, or symbol-based feedback instead of voice
3. Strengthens mutual understanding through accessible, emotion-aware, and culturally adapted interaction methods

## ✨ Key Features

### 1. Text/Touch-to-Sign Conversion
- **Parent Input Options**:
  - Type Sinhala text using keyboard
  - Select pre-defined phrases (common words)
  - Choose from categorized icon library (food, activities, emotions)
- **Sign Output**:
  - 2D hand drawn SSL signs
  - Smooth, natural sign animations
  - Loop and replay options
  - Adjustable animation speed

### 2. Child Expression Interface
- **Multiple Input Methods**:
  - Icon/picture selection interface
  - Emotion selector (visual faces)
  - Quick needs buttons (eat, drink, toilet, play, hurt)
- **Visual Feedback**:
  - Selected item highlights
  - Confirmation animations

### 3. Emotion and Feedback Layer
- **Emotional Expression**:
  - Emoji-style facial icons (😊😢😡😨😫)
  - Color-coded emotional states (green=happy, blue=sad, red=angry)
  - Intensity indicators (slightly, very)
  - Contextual emotion suggestions


### 4. Daily Communication Templates
- **Pre-Built Scenarios**:
  - Mealtime (hunger, thirst, food preferences)
  - School/Homework (subjects, difficulties, achievements)
  - Health (pain location, symptoms, feelings)
  - Play/Activities (toys, games, friends)
  - Bedtime (tired, scared, story request)
  - Emotions (happy, sad, angry, scared, tired)

### 5. Offline Functionality
- **No Internet Required**:
  - All sign images stored locally
  - Offline icon library
  - Local database of phrases
  - Works in rural areas


## 🔬 Technical Approach

### Communication Flow

```
Parent Side:
Text Input / Icon Selection
         ↓
Text-to-Sign Mapping
         ↓
Sign image
         ↓
Display to Child

Child Side:
Sign/Icon/Emotion Selection
         ↓
Visual Recognition / Direct Input
         ↓
Text Generation
         ↓
Display to Parent
```

### System Components

1. **Text-to-Sign Engine**
   - Input: Sinhala Unicode text
   - Processing: Word segmentation, grammar parsing
   - Output: Sequence of sign animation IDs
   - Special handling: Compound words, fingerspelling

2. **2D Sign images System**
   - Format: 2D hand drawn images

3. **Sign Recognition Module** (Optional)
   - Model: Lightweight CNN for mobile
   - Input: Camera frames
   - Output: Recognized sign + confidence
   - Fallback: Icon selection if low confidence

4. **Icon Library System**
   - Categories: 5+ domains (food, school, emotions, etc.)
   - Icons: 40+ culturally appropriate images
   - Search: Visual search and text search
   - Customization: Add family-specific icons

5. **Conversation Manager**
   - History: Saves conversations locally
   - Context: Understands previous messages
   - Suggestions: Predicts likely responses
   - Analytics: Tracks communication patterns

## 🏗️ System Architecture

```
┌─────────────────────────────────────────┐
│         Dual-View Interface             │
├──────────────────┬──────────────────────┤
│   Parent View    │    Child View        │
│  Text/Icons In   │  Signs/Icons Out     │
│  Signs Out       │  Icons/Emotions In   │
└──────────────────┴──────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│      Communication Bridge Layer         │
├─────────────────────────────────────────┤
│  Message Queue  │  Translation Engine   │
│  Sync Manager   │  Emotion Processor    │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│         Content Processing              │
├─────────────────────────────────────────┤
│  Text-to-Sign  │  Sign Recognition      │
│  Icon Mapping  │  Animation Controller  │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│          Local Storage                  │
├─────────────────────────────────────────┤
│  Sign Library  │  Icon Database         │
│  Conversations │  User Preferences      │
└─────────────────────────────────────────┘
```

## 📊 Content Requirements

### Sign Animation Library
- **Basic Signs**: 50+ common words and phrases
- **Daily Vocabulary**: Food, clothing, activities, emotions
- **Question Signs**: What, where, when, who, why, how
- **Social Signs**: Please, thank you, sorry, help
- **Fingerspelling**: Sinhala alphabet

### Icon Library
- **Food & Drink**: 100+ items (rice, curry, fruits, beverages)
- **Activities**: 80+ actions (play, study, sleep, bathe)
- **Emotions**: 50+ expressions (happy, sad, angry, scared, tired)
- **People**: 40+ (family members, friends, teachers)
- **Places**: 40+ (home, school, hospital, park)
- **Objects**: 200+ (toys, clothes, books, tools)

### Phrase Templates
- **Greetings**: Good morning, good night, hello
- **Needs**: I'm hungry, I'm tired, I need help
- **Questions**: Where is mom? Can I play? What's for dinner?
- **Responses**: Yes, no, okay, thank you, I don't know

## 🚀 Installation and Setup

### Prerequisites
```bash
React Native 0.70+
Node.js 16+
Unity 3D (for animation rendering) or Three.js
SQLite for local storage
```

### Installation Steps

1. **Clone the component**
```bash
cd components/two-way-comm
```

2. **Install dependencies**
```bash
npm install
```

3. **Download sign animation assets**
```bash
npm run download-assets
```

4. **Set up local database**
```bash
npm run init-database
```

5. **Build the app**
```bash
# For Android
npm run android

# For iOS
cd ios && pod install && cd ..
npm run ios
```

### Configuration

Edit `config/app-settings.json`:
```json
{
  "interface": {
    "childView": {
      "iconSize": "large",
      "categoriesVisible": 6,
      "quickAccessButtons": 8
    },
    "parentView": {
      "keyboardType": "sinhala",
      "phraseCategories": true,
      "animationSpeed": 1.0
    }
  },
  "communication": {
    "autoPlaySigns": true,
    "enableEmotions": true,
    "saveHistory": true,
    "historyLimit": 500
  },
  "accessibility": {
    "highContrast": false,
    "largeText": false,
    "hapticFeedback": true
  }
}
```

## 💻 Usage Examples

### Parent Sends Message

```javascript
import { CommunicationBridge } from './src/communication';

const bridge = new CommunicationBridge();

// Type text message
bridge.sendFromParent({
  type: 'text',
  content: 'කන්න කැමති කුමක්ද?',  // What do you want to eat?
  emotion: 'neutral'
});

// Or select icon
bridge.sendFromParent({
  type: 'icon',
  iconId: 'food-rice',
  question: true
});

// System converts to sign animation for child
```

### Child Responds

```javascript
// Child selects icon
bridge.sendFromChild({
  type: 'icon',
  iconId: 'food-rice-and-curry',
  emotion: 'happy'
});

// System displays text + icon to parent
```

### Emotion Expression

```javascript
// Child expresses emotion
bridge.sendEmotion({
  emotion: 'sad',
  intensity: 'very',
  reason: 'toy-broken'
});

// Parent sees: "Very sad - toy broken" with visual cues
```

## 📱 User Interface Design

### Parent View Layout
```
┌─────────────────────────────────────┐
│  [<]  Two-Way Communication  [⚙️]  │
├─────────────────────────────────────┤
│                                     │
│  [Child's Messages]                 │
│   😊 "I want to play" [12:30 PM]   │
│                                     │
│  [Your Messages]                    │
│   "Let's play after lunch" [12:31]  │
│   [Sign Animation Playing...]       │
│                                     │
├─────────────────────────────────────┤
│  Type message: [_______________]    │
│  [Quick Phrases ▼] [Icons 🎨]      │
└─────────────────────────────────────┘
```

### Child View Layout
```
┌─────────────────────────────────────┐
│         📱 Tell Mom & Dad           │
├─────────────────────────────────────┤
│  Quick Needs:                       │
│  [🍽️ Eat] [💧 Drink] [🚽 Toilet]   │
│  [🎮 Play] [😴 Sleep] [🤕 Hurt]     │
├─────────────────────────────────────┤
│  Categories:                        │
│  [🍔 Food] [🏀 Play] [📚 School]    │
│  [❤️ Feelings] [👨‍👩‍👧 Family] [More...]│
├─────────────────────────────────────┤
│  How do you feel?                   │
│  😊 😢 😡 😨 😫 😐                  │
├─────────────────────────────────────┤
│  Parent's Message:                  │
│  [Sign Animation Display]           │
└─────────────────────────────────────┘
```

## 📈 Evaluation Metrics

### Communication Effectiveness
- **Message Clarity**: Successful message understanding rate
- **Response Time**: Time to compose and send message
- **Conversation Flow**: Natural back-and-forth patterns
- **Error Rate**: Misunderstandings or corrections needed

### User Experience
- **Ease of Use**: Learning curve for parents and children
- **Satisfaction**: User feedback and ratings
- **Engagement**: Daily usage frequency
- **Accessibility**: Success across different age groups

### System Performance
- **Animation Quality**: Smoothness and clarity of signs
- **Response Speed**: Time from input to output
- **Storage Efficiency**: Local database size
- **Battery Usage**: Power consumption per hour

### Social Impact
- **Family Communication**: Improvement in daily interactions
- **Child Expression**: Increase in communication attempts
- **Parent Understanding**: Better comprehension of child's needs
- **Emotional Well-being**: Reduced frustration for both parties

## 🎨 Cultural Adaptation

### Sinhala Language Support
- **Unicode Input**: Full Sinhala keyboard support
- **Grammar Handling**: Proper word order for SSL
- **Common Phrases**: Culturally appropriate expressions
- **Respectful Forms**: Proper terms for family members

### Icon Customization
- **Sri Lankan Context**: Local foods, clothing, activities
- **Family Photos**: Option to add personal family images
- **Custom Signs**: Record family-specific gestures
- **Regional Variations**: Different signing styles supported

## 🔮 Future Enhancements

- **AI-Powered Suggestions**: Predict likely responses based on context
- **Multi-Child Support**: Different profiles for multiple children
- **Progress Tracking**: Monitor communication development over time
- **Educational Mode**: Gradually introduce SSL learning
- **Voice Input**: Optional speech-to-text for parents
- **Smart Watch Integration**: Quick alerts and responses
- **Cloud Backup**: Sync across devices (optional)
- **Community Features**: Share custom signs and icons

## 🐛 Known Issues and Limitations

- Animation rendering may be slow on older devices
- Limited sign vocabulary (expanding continuously)
- Sign recognition requires good camera and lighting
- Some complex concepts difficult to express with icons
- Initial setup requires significant storage space

## 📚 References

### HCI and Accessibility
- Universal Design principles
- Child-Computer Interaction (CCI)
- Accessibility guidelines for disabilities

### Sign Language
- Sinhala Sign Language grammar
- Visual-spatial communication
- Non-manual features in SSL

### Mobile Development
- React Native best practices
- 3D animation on mobile devices
- Offline-first application design

## 🤝 Contributing

We welcome contributions in:
- New sign animations
- Icon library expansion
- Translation improvements
- UI/UX enhancements
- Testing with families

## 📝 License

MIT License - See main project LICENSE file

## 👤 Developer

**De Silva U.P.A.N.** (IT22308870)
- Email: asanganavodidesilva@gmail.com
- Focus: HCI, Accessibility, Mobile Development, Visual Communication

## 🙏 Special Thanks

- Sri Lankan Deaf community
- Families with children with speaking disabilities
- Hashila, Indumini and their parents
- Ms. Pradeepa
- Special education teachers
- Sign language interpreters
- UX/UI designers

---

**Component Status**: Active Development | **Version**: 0.1.0 | **Last Updated**: January 2025

## 📞 Support and Feedback

For support, feature requests, or to share your experience:
- Email: support@ssl-communication.org
- Community Forum: [Link to forum]
- Issue Tracker: [GitHub Issues]

---

**Note for Families**: This system is designed to complement, not replace, sign language education. We encourage families to learn SSL together while using this tool for daily communication.
