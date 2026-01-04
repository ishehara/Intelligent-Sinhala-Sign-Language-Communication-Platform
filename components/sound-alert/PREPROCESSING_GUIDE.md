# Sound Detection Component - Quick Start Guide

## Overview
This component processes audio files for sound detection in the Sinhala Sign Language learning app. It extracts MFCC features from audio files and prepares them for machine learning.

## Features Extracted
- **MFCC (Mel-Frequency Cepstral Coefficients)**: 13 coefficients
- **Frames**: 40 time frames per audio sample
- **Output**: Flattened feature vector (13 × 40 = 520 features per sample)

## Installation

```bash
cd components/sound-alert
pip install -r requirements.txt
```

## Dataset Structure

Your audio files should be organized in folders by category:

```
datasets/
├── Vehicle Horns/
│   ├── bus horns/
│   │   ├── audio1.wav
│   │   ├── audio2.wav
│   │   └── ...
│   ├── car horns/
│   ├── motorcycle horns/
│   ├── train horns/
│   └── truck horns/
└── sirens/
    ├── ambulance/
    ├── firetruck/
    ├── police/
    └── traffic/
```

## Usage

### Method 1: Process All Datasets (Batch)

Process both Vehicle Horns and Sirens datasets:

```bash
cd components/sound-alert/src
python run_preprocessing.py
```

This will create processed data in:
- `components/sound-alert/data/processed/vehicle_horns/`
- `components/sound-alert/data/processed/sirens/`

### Method 2: Process Single Dataset

Process a specific dataset:

```bash
python preprocessing.py \
  --data_dir "../../datasets/Vehicle Horns" \
  --output_dir "../data/processed/vehicle_horns" \
  --n_mfcc 13 \
  --n_frames 40 \
  --test_size 0.2
```

### Method 3: Use as Python Module

```python
from components.sound_alert.src.preprocessing import preprocess_audio_dataset

# Process your custom dataset
preprocess_audio_dataset(
    data_dir="path/to/your/audio/folders",
    output_dir="path/to/save/processed/data",
    n_mfcc=13,
    n_frames=40,
    test_size=0.2
)
```

## Output Files

After preprocessing, you'll get:

```
data/processed/vehicle_horns/
├── X_train.npy          # Training features (80%)
├── X_test.npy           # Testing features (20%)
├── y_train.npy          # Training labels
├── y_test.npy           # Testing labels
├── label_mapping.json   # Category name to label mapping
└── metadata.json        # Processing parameters
```

## Loading Processed Data

```python
import numpy as np
import json

# Load data
X_train = np.load('data/processed/vehicle_horns/X_train.npy')
X_test = np.load('data/processed/vehicle_horns/X_test.npy')
y_train = np.load('data/processed/vehicle_horns/y_train.npy')
y_test = np.load('data/processed/vehicle_horns/y_test.npy')

# Load label mapping
with open('data/processed/vehicle_horns/label_mapping.json', 'r') as f:
    label_info = json.load(f)
    
print(f"Categories: {label_info['encoder']}")
print(f"Training samples: {X_train.shape}")
print(f"Feature dimension: {X_train.shape[1]}")
```

## Supported Audio Formats

- `.wav` (recommended)
- `.mp3`
- `.flac`

## Parameters

- **n_mfcc**: Number of MFCC coefficients (default: 13)
- **n_frames**: Number of time frames (default: 40)
- **sample_rate**: Audio sample rate in Hz (default: 22050)
- **duration**: Audio duration to load in seconds (default: 2.5)
- **test_size**: Proportion of data for testing (default: 0.2 = 20%)

## Next Steps

After preprocessing, you can:
1. Train a machine learning model (SVM, Random Forest, Neural Network)
2. Build a real-time sound detection system
3. Deploy the model for inference

## Troubleshooting

### "No audio files found"
- Check your folder structure
- Ensure audio files have supported extensions (.wav, .mp3, .flac)
- Verify the `data_dir` path is correct

### "Error loading audio file"
- Install ffmpeg: `conda install -c conda-forge ffmpeg`
- Or use system package manager: `apt-get install ffmpeg`

### Memory issues
- Reduce `n_frames` or `duration` 
- Process datasets one at a time
- Use smaller batch sizes

## Example Output

```
====================================================================
Audio Dataset Preprocessing Pipeline
====================================================================
Input directory: datasets/Vehicle Horns
Output directory: components/sound-alert/data/processed/vehicle_horns
MFCC coefficients: 13
Number of frames: 40
Test size: 20%
====================================================================

[1/4] Loading audio files...
Found 5 categories in datasets/Vehicle Horns
Processing 'bus horns': 150 files
Processing 'car horns': 200 files
Processing 'motorcycle horns': 180 files
Processing 'train horns': 120 files
Processing 'truck horns': 170 files

Dataset loaded successfully!
Total samples: 820
Feature shape: (820, 520)
Categories: ['bus horns', 'car horns', 'motorcycle horns', 'train horns', 'truck horns']

[2/4] Splitting dataset...
Training samples: 656 (80%)
Testing samples: 164 (20%)

[3/4] Saving processed data...
Data saved successfully to components/sound-alert/data/processed/vehicle_horns/
Files created:
  - X_train.npy: (656, 520)
  - X_test.npy: (164, 520)
  - y_train.npy: (656,)
  - y_test.npy: (164,)
  - label_mapping.json
  - metadata.json

[4/4] Complete!
====================================================================
Preprocessing completed successfully!
====================================================================
```
