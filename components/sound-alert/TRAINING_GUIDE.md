# Sound Detection - Model Training Guide

## Overview
This guide covers training a CNN model for sound classification using preprocessed MFCC features.

## Model Architecture

**Convolutional Neural Network (CNN)**
- 3 Convolutional blocks (32, 64, 128 filters)
- Batch Normalization after each Conv2D
- MaxPooling2D for spatial reduction
- Dropout for regularization (0.25-0.5)
- 2 Dense layers (256, 128 units)
- Softmax output for multi-class classification

**Input Shape:** `(13, 40, 1)` - MFCC coefficients × time frames × channels  
**Output:** Probability distribution over sound categories

## Training Features

✓ **Early Stopping** - Stops training when validation loss stops improving (patience=15)  
✓ **Model Checkpoint** - Saves best model based on validation accuracy  
✓ **Learning Rate Reduction** - Reduces LR when loss plateaus  
✓ **TensorBoard Logging** - Visual training monitoring  
✓ **Batch Normalization** - Faster convergence  
✓ **Dropout** - Prevents overfitting

## Installation

Ensure TensorFlow is installed:
```bash
pip install tensorflow>=2.15.0
```

Or install all requirements:
```bash
cd components/sound-alert
pip install -r requirements.txt
```

## Usage

### Basic Training

Train on preprocessed Vehicle Horns data:

```bash
cd components/sound-alert/src

.\..\..\..\..\venv\Scripts\python.exe train_model.py \
  --data_dir "../data/processed/vehicle_horns" \
  --model_dir "../models/vehicle_horns_cnn"
```

### Custom Parameters

```bash
.\..\..\..\..\venv\Scripts\python.exe train_model.py \
  --data_dir "../data/processed/vehicle_horns" \
  --model_dir "../models/vehicle_horns_cnn" \
  --epochs 150 \
  --batch_size 64 \
  --learning_rate 0.0005 \
  --validation_split 0.2
```

### Python API

```python
from train_model import train_sound_classifier

train_sound_classifier(
    data_dir="components/sound-alert/data/processed/vehicle_horns",
    model_dir="components/sound-alert/models/vehicle_horns_cnn",
    epochs=100,
    batch_size=32,
    validation_split=0.15,
    learning_rate=0.001
)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_dir` | Required | Directory with preprocessed .npy files |
| `model_dir` | Required | Directory to save trained model |
| `epochs` | 100 | Maximum training epochs |
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 0.001 | Initial learning rate |
| `validation_split` | 0.15 | % of training data for validation |

## Expected Output

```
====================================================================
LOADING PREPROCESSED DATA
====================================================================
Data directory: components/sound-alert/data/processed/vehicle_horns

Data loaded successfully!
X_train shape: (656, 520)
X_test shape: (164, 520)
Number of classes: 5
Classes: ['bus horns', 'car horns', 'motorcycle horns', 'train horns', 'truck horns']

Reshaping data for CNN...
Reshaped X_train: (656, 13, 40, 1)

====================================================================
BUILDING MODEL
====================================================================
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 13, 40, 32)        320       
 batch_normalization         (None, 13, 40, 32)        128       
 max_pooling2d              (None, 6, 20, 32)         0         
 dropout                    (None, 6, 20, 32)         0         
 ...
=================================================================
Total params: 1,234,567

====================================================================
TRAINING CNN MODEL
====================================================================
Training samples: 557
Validation samples: 99
Batch size: 32
Epochs: 100

Epoch 1/100
18/18 [==============================] - 2s 50ms/step
Epoch 15/100
18/18 [==============================] - 1s 45ms/step - val_accuracy: 0.9293

====================================================================
EVALUATING MODEL
====================================================================
Test Loss: 0.2156
Test Accuracy: 94.51%

====================================================================
CLASSIFICATION REPORT
====================================================================
                    precision    recall  f1-score   support

      bus horns       0.9500    0.9500    0.9500        30
      car horns       0.9750    0.9750    0.9750        40
motorcycle horns      0.9444    0.9444    0.9444        36
    train horns       0.9167    0.9167    0.9167        24
    truck horns       0.9286    0.9286    0.9286        34

       accuracy                           0.9451       164
      macro avg       0.9429    0.9429    0.9429       164
   weighted avg       0.9451    0.9451    0.9451       164

====================================================================
TRAINING PIPELINE COMPLETE!
====================================================================
✓ Model saved: models/vehicle_horns_cnn/best_model.keras
✓ Results saved: models/vehicle_horns_cnn/training_results.json
✓ Plots saved: models/vehicle_horns_cnn

Final Test Accuracy: 94.51%
✓ Target accuracy (85%) achieved!
====================================================================
```

## Output Files

After training:

```
models/vehicle_horns_cnn/
├── best_model.keras          # Best model (highest val accuracy)
├── final_model.keras          # Final model after all epochs
├── training_results.json      # Metrics summary
├── training_history.png       # Loss & accuracy curves
├── confusion_matrix.png       # Confusion matrix heatmap
└── logs/                      # TensorBoard logs
    └── 20260102-143025/
```

## Loading Trained Model

```python
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model('models/vehicle_horns_cnn/best_model.keras')

# Prepare input (MFCC features: 13x40)
X_input = np.random.randn(1, 13, 40, 1)  # Shape: (1, 13, 40, 1)

# Predict
predictions = model.predict(X_input)
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class]

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence*100:.2f}%")
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir models/vehicle_horns_cnn/logs
```

Open browser to `http://localhost:6006`

### Real-time Progress

Watch the terminal for:
- Loss decreasing
- Accuracy increasing
- Validation metrics improving

## Tips for Better Accuracy

### If accuracy < 85%:

1. **Increase epochs**: `--epochs 150`
2. **Reduce learning rate**: `--learning_rate 0.0005`
3. **Larger batch size**: `--batch_size 64`
4. **More data**: Add more audio samples
5. **Data augmentation**: Add noise, time-shift, pitch-shift

### If overfitting (train acc >> val acc):

1. **Increase dropout**: Modify dropout rates in code
2. **More regularization**: Add L2 regularization
3. **Early stopping**: Already enabled (patience=15)
4. **Reduce model complexity**: Fewer Conv layers

### If underfitting (both low):

1. **Larger model**: More Conv filters
2. **Longer training**: More epochs
3. **Better features**: Try different MFCC parameters
4. **Check data quality**: Verify preprocessing

## Next Steps

1. ✅ **Preprocessing** - Extract MFCC features
2. ✅ **Training** - Train CNN model
3. **Deployment** - Create inference pipeline
4. **Real-time Detection** - Audio stream processing
5. **Mobile Deployment** - TFLite conversion

## Example: Train Both Datasets

```bash
# Train Vehicle Horns
python train_model.py \
  --data_dir "../data/processed/vehicle_horns" \
  --model_dir "../models/vehicle_horns_cnn"

# Train Sirens
python train_model.py \
  --data_dir "../data/processed/sirens" \
  --model_dir "../models/sirens_cnn"
```

## Troubleshooting

**"No module named tensorflow"**
```bash
pip install tensorflow
```

**Out of memory error**
- Reduce `batch_size` to 16 or 8
- Use smaller model architecture

**Training too slow**
- Use GPU if available
- Increase `batch_size` to 64
- Reduce validation frequency

**Low accuracy**
- Check data quality and labels
- Ensure preprocessing was correct
- Try different hyperparameters
- Collect more training data
