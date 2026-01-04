# Sound Detection - Testing & Inference Guide

## Overview
This guide shows how to test your trained model and demonstrate that it's detecting sounds correctly.

## Files Created
- **[inference.py](src/inference.py)** - Core inference module with `SoundDetector` class
- **[demo_detection.py](src/demo_detection.py)** - Automated demo script testing all categories

## Quick Demo - Test All Categories

After training completes, run this to test your model:

```bash
cd components/sound-alert/src
.\..\..\..\..\venv\Scripts\python.exe demo_detection.py
```

This will automatically:
- Load your trained model
- Test on samples from each category (bus, car, motorcycle, train, truck horns)
- Show predictions with confidence scores
- Calculate accuracy per category
- Display overall performance

## Test Single Audio File

```bash
.\..\..\..\..\venv\Scripts\python.exe inference.py \
  --model_dir "../models/vehicle_horns_cnn" \
  --data_dir "../data/processed/vehicle_horns" \
  --audio_file "../../datasets/Vehicle Horns/car horns/car_horn_001.wav"
```

**Output:**
```
======================================================================
Analyzing: car_horn_001.wav
======================================================================

🎯 PREDICTION: car horns
📊 CONFIDENCE: 97.34%

📈 All Class Probabilities:
----------------------------------------------------------------------
  car horns            97.34% ████████████████████████████████████████████████
  truck horns           1.45% ▌
  bus horns             0.78% ▎
  motorcycle horns      0.32% 
  train horns           0.11% 
======================================================================
```

## Test Entire Category Folder

Test all car horn samples:

```bash
.\..\..\..\..\venv\Scripts\python.exe inference.py \
  --model_dir "../models/vehicle_horns_cnn" \
  --data_dir "../data/processed/vehicle_horns" \
  --test_folder "../../datasets/Vehicle Horns/car horns" \
  --expected_class "car horns" \
  --max_files 20
```

**Output:**
```
======================================================================
Testing on folder: car horns
Expected class: car horns
Number of files: 20
======================================================================

✓ [1/20] car_horn_001.wav    → car horns      (97.3%)
✓ [2/20] car_horn_002.wav    → car horns      (95.8%)
✓ [3/20] car_horn_003.wav    → car horns      (98.1%)
✗ [4/20] car_horn_004.wav    → truck horns    (62.4%)
✓ [5/20] car_horn_005.wav    → car horns      (96.7%)
...

======================================================================
RESULTS SUMMARY
======================================================================
Total files tested: 20
Correct predictions: 18/20
Accuracy: 90.00%
Average confidence: 89.45%
======================================================================
```

## Use in Python Code

```python
from components.sound_alert.src.inference import SoundDetector

# Initialize detector
detector = SoundDetector(
    model_path="components/sound-alert/models/vehicle_horns_cnn/best_model.keras",
    metadata_path="components/sound-alert/data/processed/vehicle_horns/metadata.json",
    label_mapping_path="components/sound-alert/data/processed/vehicle_horns/label_mapping.json"
)

# Test single file
predicted_class, confidence, all_probs = detector.predict("path/to/audio.wav")
print(f"Predicted: {predicted_class} ({confidence*100:.2f}%)")

# Show detailed results
detector.predict_with_details("path/to/audio.wav")

# Test multiple files
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
results = detector.predict_batch(audio_files)
for audio, (pred, conf, _) in zip(audio_files, results):
    print(f"{audio}: {pred} ({conf*100:.2f}%)")
```

## Verify Model Performance

### 1. Check Test Set Accuracy

After training completes, check the results:
```bash
# View training results
cat components/sound-alert/models/vehicle_horns_cnn/training_results.json
```

### 2. Run Demo on All Categories

```bash
cd components/sound-alert/src
.\..\..\..\..\venv\Scripts\python.exe demo_detection.py
```

### 3. Visual Inspection

Check the generated plots:
- `models/vehicle_horns_cnn/training_history.png` - Training curves
- `models/vehicle_horns_cnn/confusion_matrix.png` - Per-class performance

## Real-Time Audio Detection

For live microphone input (requires `pyaudio`):

```python
import pyaudio
import numpy as np
from components.sound_alert.src.inference import SoundDetector

# Initialize detector
detector = SoundDetector(...)

# Audio recording parameters
CHUNK = 1024
RATE = 22050
DURATION = 2.5

# Open microphone stream
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Listening...")

while True:
    # Record audio chunk
    frames = []
    for _ in range(0, int(RATE / CHUNK * DURATION)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.float32))
    
    audio = np.concatenate(frames)
    
    # Save temporarily and predict
    import soundfile as sf
    sf.write('temp.wav', audio, RATE)
    
    predicted_class, confidence, _ = detector.predict('temp.wav')
    
    if confidence > 0.8:  # High confidence threshold
        print(f"🔊 Detected: {predicted_class} ({confidence*100:.1f}%)")
```

## Understanding Results

### Confidence Score
- **>90%**: Very confident prediction
- **70-90%**: Good confidence
- **50-70%**: Uncertain, borderline
- **<50%**: Low confidence, may be wrong

### What to Look For

✅ **Good Model:**
- Test accuracy >85%
- High confidence scores (>80%)
- Clear separation in probability distribution
- Confusion matrix shows diagonal pattern

⚠️ **Needs Improvement:**
- Test accuracy <75%
- Low confidence scores (<60%)
- Many misclassifications between similar classes
- Scattered confusion matrix

## Troubleshooting

**"Model file not found"**
- Make sure training completed successfully
- Check path to `best_model.keras`

**"Metadata not found"**
- Ensure preprocessing was run first
- Check `data/processed/vehicle_horns/` directory

**Low accuracy on new samples**
- Model may be overfitting
- Try more diverse training data
- Check if audio quality matches training data

**Wrong predictions**
- Verify audio format matches training
- Check MFCC extraction parameters
- Ensure audio length is appropriate

## Next Steps

1. ✅ Train model
2. ✅ Test with inference scripts
3. **Deploy for real-time use**
4. **Integrate with mobile app**
5. **Create alert system**

## Command Reference

```bash
# Test single file
python inference.py --model_dir MODEL_DIR --data_dir DATA_DIR --audio_file AUDIO.wav

# Test folder
python inference.py --model_dir MODEL_DIR --data_dir DATA_DIR --test_folder FOLDER --expected_class "class name"

# Run full demo
python demo_detection.py
```
