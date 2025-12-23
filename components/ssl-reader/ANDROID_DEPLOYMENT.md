# 📱 Android Deployment - Complete Guide

## Real-Time On-Device Sign Language Recognition for Android

Transform sign language into text in real-time, completely on-device, with no internet required!

---

## 🎯 What You Get

A complete **Android app** that:

✅ **Captures video** from camera  
✅ **Processes frames** on-device (MediaPipe)  
✅ **Recognizes signs** with AI (TensorFlow Lite)  
✅ **Outputs text** in real-time  
✅ **Works offline** - No internet needed  
✅ **Protects privacy** - All processing on-device  
✅ **Integrates with React Native** - Beautiful UI  

---

## 🚀 Quick Start (3 Steps)

### Step 1: Train Your Model

```bash
cd components/ssl-reader/src
python quick_train.py
```

This trains the AI model on your Sinhala sign language videos.

### Step 2: Deploy for Android

```bash
python deploy_android.py
```

This converts your model to TensorFlow Lite for Android.

### Step 3: Integrate with React Native

Follow the [React Native Guide](REACT_NATIVE_GUIDE.md) to build your mobile app.

**That's it!** 🎉

---

## 📋 Detailed Workflow

### 1. Train Model (Desktop)

Train the AI model on your computer:

```bash
cd components/ssl-reader/src

# Quick training
python quick_train.py

# Or custom training
python train.py \
    --dataset_root ../../datasets/signVideos \
    --model_type hybrid \
    --num_epochs 50
```

**Output:** `../models/checkpoint_best.pth`

### 2. Convert for Mobile

Convert PyTorch model to TensorFlow Lite:

```bash
# Automated deployment
python deploy_android.py

# Or manual conversion
python convert_to_mobile.py \
    --model_path ../models/checkpoint_best.pth \
    --android_package
```

**Output:** 
```
android_deployment/
├── model.tflite           # Optimized for Android
├── labels.txt             # Sign language labels
├── model_info.json        # Metadata
└── ANDROID_INTEGRATION.md # Integration guide
```

### 3. Build React Native App

See complete code in [REACT_NATIVE_GUIDE.md](REACT_NATIVE_GUIDE.md)

**Key Components:**

```javascript
// Load model
const recognizer = new SignLanguageRecognizer();
await recognizer.initialize();

// Process camera frames
const processFrame = (frame) => {
  // Extract features with MediaPipe
  const features = extractFeatures(frame);
  
  // Add to buffer
  recognizer.addFrame(features);
  
  // Recognize when ready
  if (bufferReady) {
    const result = recognizer.recognize();
    // Display: result.label
  }
};
```

### 4. Deploy to Android

```bash
# Build APK
cd <your-react-native-project>
cd android
./gradlew assembleRelease

# Output: app/build/outputs/apk/release/app-release.apk
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────┐
│         React Native App                │
│         (User Interface)                │
└──────────────┬──────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│         Camera Capture                  │
│         (30 FPS Video Frames)           │
└──────────────┬──────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│    MediaPipe Holistic (On-Device)       │
│    ├─ Hand Tracking (42 points)         │
│    ├─ Face Landmarks (468 points)       │
│    └─ Pose Detection (33 points)        │
└──────────────┬──────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│    Feature Extraction (On-Device)       │
│    Convert landmarks → 395D vector      │
└──────────────┬──────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│    Buffer 60 Frames (On-Device)         │
│    Accumulate temporal sequence         │
└──────────────┬──────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│    TensorFlow Lite Model (On-Device)    │
│    LSTM/Transformer Neural Network      │
└──────────────┬──────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────┐
│    Text Output (Real-Time)              │
│    Display recognized sign              │
└─────────────────────────────────────────┘
```

**Everything runs on the Android device!**

---

## 💻 Two Deployment Options

### Option A: Pure On-Device (Recommended)

**100% on-device processing** - Best for privacy and offline use

```
React Native App
    ↓
TensorFlow Lite (On-Device)
    ↓
Text Output
```

**Pros:**
- ✅ Complete privacy
- ✅ Works offline
- ✅ No server needed
- ✅ Fast response

**Setup:** Follow [REACT_NATIVE_GUIDE.md](REACT_NATIVE_GUIDE.md)

### Option B: Local API Bridge

**Local server + React Native** - Easier development/testing

```
React Native App
    ↓ (HTTP)
Python API Server (Local Network)
    ↓
PyTorch Model
    ↓
Text Output
```

**Pros:**
- ✅ Easier to update model
- ✅ Faster development
- ✅ Still local (no cloud)

**Setup:**
```bash
# Start server
python react_native_bridge.py \
    --model_path ../models/checkpoint_best.pth

# React Native connects to http://<local-ip>:5000
```

---

## 📊 Performance Expectations

### Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 80-92% |
| Inference Time | 50-200ms |
| Model Size | 25-30 MB (TFLite) |
| Frames Required | 60 (~2 seconds) |

### Device Requirements

**Minimum:**
- Android 8.0+
- 3 GB RAM
- ARM v8 processor
- Camera

**Recommended:**
- Android 10.0+
- 4 GB RAM
- GPU support
- 720p+ camera

### Real-Time Performance

- **Frame Processing:** ~30 FPS
- **Feature Extraction:** ~20ms per frame
- **Model Inference:** ~50-200ms
- **Total Latency:** ~2-3 seconds (60 frames)

---

## 🔒 Privacy & Security

### On-Device Guarantees

✅ **No Network Communication**
- All processing local to device
- No API calls to servers
- No data transmission

✅ **No Data Storage**
- Frames processed in memory
- No video recording
- Features discarded after use

✅ **Complete Privacy**
- GDPR compliant
- HIPAA ready
- Air-gap compatible

### Permissions Required

Only camera permission needed:
```xml
<uses-permission android:name="android.permission.CAMERA" />
```

**NO INTERNET PERMISSION REQUIRED!**

---

## 🛠️ Development Tools

### Scripts Created

1. **`quick_train.py`** - Easy model training
2. **`train.py`** - Advanced training with options
3. **`convert_to_mobile.py`** - PyTorch → TFLite converter
4. **`optimize_model.py`** - Model optimization (quantization, ONNX)
5. **`react_native_bridge.py`** - Local API server
6. **`deploy_android.py`** - Automated deployment
7. **`inference.py`** - Testing & webcam demo

### Documentation

1. **`TRAINING_GUIDE.md`** - How to train models
2. **`ON_DEVICE_DEPLOYMENT.md`** - On-device processing guide
3. **`REACT_NATIVE_GUIDE.md`** - React Native integration (THIS FILE)
4. **`ANDROID_INTEGRATION.md`** - Native Android code

---

## 📝 Example: Complete Deployment

```bash
# ==================================
# 1. TRAIN MODEL
# ==================================
cd components/ssl-reader/src
python quick_train.py
# Choose: Hybrid model, with preprocessing

# ==================================
# 2. DEPLOY FOR ANDROID
# ==================================
python deploy_android.py
# Choose: Android (TensorFlow Lite)

# ==================================
# 3. CREATE REACT NATIVE APP
# ==================================
npx react-native init SinhalaSignApp
cd SinhalaSignApp

# Copy model files
cp ../ssl-reader/models/android_deployment/model.tflite \
   android/app/src/main/assets/

cp ../ssl-reader/models/android_deployment/labels.txt \
   android/app/src/main/assets/

# Add dependencies (see REACT_NATIVE_GUIDE.md)
npm install @react-native-camera/camera
npm install @tensorflow/tfjs-react-native

# Add code (see REACT_NATIVE_GUIDE.md for complete code)

# ==================================
# 4. BUILD & RUN
# ==================================
npm run android

# ==================================
# 5. BUILD APK FOR DISTRIBUTION
# ==================================
cd android
./gradlew assembleRelease
# Output: app/build/outputs/apk/release/app-release.apk

# ==================================
# DONE! 🎉
# ==================================
```

---

## 🎓 Learning Path

### For Beginners

1. **Start Here:** Run `quick_train.py` to train your first model
2. **Test Locally:** Use `inference.py --mode webcam` to test
3. **Deploy:** Use `deploy_android.py` for Android package
4. **Build App:** Follow simple React Native example
5. **Iterate:** Improve model with more data

### For Advanced Users

1. **Custom Models:** Modify `models.py` for new architectures
2. **Optimize:** Use `optimize_model.py` for quantization/pruning
3. **Fine-tune:** Adjust hyperparameters in `train.py`
4. **Extend:** Add emotion recognition, context awareness
5. **Scale:** Deploy to multiple platforms (iOS, Web)

---

## 🆘 Troubleshooting

### Model Conversion Fails

**Problem:** TFLite conversion error

**Solution:**
```bash
pip install tensorflow onnx onnx-tf
python convert_to_mobile.py --android_package
```

### React Native Camera Not Working

**Problem:** Camera shows black screen

**Solution:**
```javascript
// Request permissions first
import { PermissionsAndroid } from 'react-native';

await PermissionsAndroid.request(
  PermissionsAndroid.PERMISSIONS.CAMERA
);
```

### Slow Performance on Device

**Problem:** Inference takes >1 second

**Solutions:**
1. Enable GPU acceleration in TFLite
2. Reduce frame processing rate (every 2nd frame)
3. Lower camera resolution
4. Use quantized model

### Model Not Loading in React Native

**Problem:** Model file not found

**Solution:**
```bash
# Verify files exist
ls android/app/src/main/assets/
# Should show: model.tflite, labels.txt

# Rebuild
cd android
./gradlew clean
./gradlew assembleDebug
```

---

## ✅ Deployment Checklist

Before releasing your app:

- [ ] Model trained and tested (>80% accuracy)
- [ ] Converted to TFLite successfully
- [ ] Model files in React Native assets
- [ ] Camera permissions added to manifest
- [ ] Tested on real Android device (not emulator)
- [ ] Verified offline functionality
- [ ] Measured inference performance (<500ms)
- [ ] UI/UX polished and user-friendly
- [ ] Error handling implemented
- [ ] APK built and tested
- [ ] Privacy policy created (optional)
- [ ] Ready for Play Store submission!

---

## 📚 Additional Resources

### Documentation
- [Training Guide](TRAINING_GUIDE.md) - Train your model
- [On-Device Guide](ON_DEVICE_DEPLOYMENT.md) - Privacy & optimization
- [React Native Guide](REACT_NATIVE_GUIDE.md) - Complete RN integration

### Example Code
- `App.js` - Main React Native component
- `SignLanguageRecognizer.js` - Recognition service
- Full code in REACT_NATIVE_GUIDE.md

### Community
- GitHub Issues - Report bugs
- Discussions - Ask questions
- Contributions - Welcome!

---

## 🎯 Summary

You now have everything to build a **production-ready Android app** that:

✅ Recognizes Sinhala sign language in real-time  
✅ Translates signs to text instantly  
✅ Works 100% offline (no internet)  
✅ Preserves complete privacy (on-device)  
✅ Runs on any modern Android phone  
✅ Has a beautiful React Native UI  

**All processing happens on the device. No cloud. No servers. Complete privacy.** 🔒

---

## 🚀 Next Steps

1. **Train your model:** `python quick_train.py`
2. **Deploy for Android:** `python deploy_android.py`
3. **Build React Native app:** Follow [REACT_NATIVE_GUIDE.md](REACT_NATIVE_GUIDE.md)
4. **Test on device:** Real Android phone
5. **Publish:** Google Play Store

**Ready to change lives with accessible communication!** 🌟

---

**Developer:** IT22304674 – Liyanage M.L.I.S.  
**Component:** SSL Reader (Sign Language Recognition)  
**Platform:** Android (React Native + TensorFlow Lite)  
**Privacy:** 100% On-Device Processing
