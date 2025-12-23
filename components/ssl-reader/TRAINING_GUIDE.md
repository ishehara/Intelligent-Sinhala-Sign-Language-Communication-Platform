# Sinhala Sign Language Recognition - Training Guide

## 🎯 Overview

This component trains a deep learning model to recognize Sinhala sign language from video data using multimodal features (hands, face, and body pose).

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **CUDA** (optional, for GPU acceleration)
3. **Video dataset** in the correct structure

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd components/ssl-reader
pip install -r requirements.txt
```

### 2. Verify Dataset Structure

Your dataset should be organized like this:
```
datasets/signVideos/
├── Adjectives/
│   ├── Good/
│   │   ├── Good_001.mp4
│   │   ├── Good_002.mp4
│   │   └── ...
│   └── Bad/
├── Greetings/
│   ├── Hello/
│   └── Thank you/
└── ...
```

### 3. Train the Model (Easy Way)

```bash
cd src
python quick_train.py
```

This interactive script will:
- Let you choose a model type (LSTM/Transformer/Hybrid)
- Optionally preprocess all videos
- Train the model with optimal settings
- Save the best model automatically

### 4. Train the Model (Advanced)

For more control, use the training script directly:

```bash
cd src
python train.py \
    --dataset_root ../../datasets/signVideos \
    --model_type lstm \
    --batch_size 16 \
    --num_epochs 50 \
    --learning_rate 0.001 \
    --cache_dir ../data/processed \
    --preprocess
```

#### Training Arguments

**Data Arguments:**
- `--dataset_root`: Path to video dataset (default: `../../datasets/signVideos`)
- `--cache_dir`: Directory to cache preprocessed features (default: `../data/processed`)
- `--preprocess`: Preprocess all videos before training (recommended for faster training)

**Model Arguments:**
- `--model_type`: Model architecture - `lstm`, `transformer`, or `hybrid` (default: `lstm`)
- `--hidden_dim`: Hidden layer dimension (default: 256)
- `--num_layers`: Number of layers (default: 2)
- `--dropout`: Dropout rate for regularization (default: 0.3)

**Training Arguments:**
- `--batch_size`: Batch size (default: 16)
- `--num_epochs`: Maximum training epochs (default: 50)
- `--learning_rate`: Learning rate (default: 0.001)
- `--patience`: Early stopping patience (default: 10)
- `--max_frames`: Maximum frames per video (default: 60)

**Other Arguments:**
- `--device`: Device to use - `cuda` or `cpu` (auto-detected)
- `--num_workers`: Data loading workers (default: 4)
- `--save_dir`: Model save directory (default: `../models`)
- `--log_dir`: TensorBoard logs directory (default: `../logs`)

## 📊 Model Architectures

### 1. LSTM Model (Recommended for Quick Testing)
- Fast training and inference
- Good baseline performance
- Lower memory requirements
- **Best for**: Quick experiments, limited hardware

### 2. Transformer Model
- Better accuracy on complex signs
- Captures long-range dependencies
- Higher computational cost
- **Best for**: Maximum accuracy, sufficient GPU

### 3. Hybrid Model (Recommended for Best Results)
- Combines LSTM and Transformer
- Best of both worlds
- Highest accuracy
- **Best for**: Production deployment

## 🎓 Training Process

1. **Data Loading**: Videos are loaded and split into train/val/test sets (70/15/15)
2. **Feature Extraction**: MediaPipe extracts hand, face, and pose landmarks
3. **Caching** (optional): Preprocessed features are saved to disk for faster loading
4. **Training**: Model learns to classify signs using temporal sequences
5. **Validation**: Performance monitored on validation set
6. **Early Stopping**: Training stops if no improvement for N epochs
7. **Checkpointing**: Best model saved automatically

## 📈 Monitoring Training

### TensorBoard

View training progress in real-time:

```bash
tensorboard --logdir ../logs
```

Then open http://localhost:6006 in your browser.

You can monitor:
- Training/validation loss
- Training/validation accuracy
- F1 score
- Learning rate

### Training Logs

Check console output for:
- Epoch progress
- Loss and accuracy metrics
- Learning rate updates
- Early stopping notifications

## 💾 Outputs

After training, you'll find:

```
components/ssl-reader/
├── models/
│   ├── checkpoint_best.pth      # Best model (highest validation accuracy)
│   └── checkpoint_latest.pth    # Latest model
├── logs/
│   └── run_YYYYMMDD_HHMMSS/     # TensorBoard logs
└── data/
    └── processed/                # Cached features (if --preprocess used)
```

## 🧪 Testing the Model

### Single Video

```bash
cd src
python inference.py \
    --model_path ../models/checkpoint_best.pth \
    --mode video \
    --video_path ../../datasets/signVideos/Greetings/Hello/Hello_001.mp4
```

### Webcam (Real-time)

```bash
python inference.py \
    --model_path ../models/checkpoint_best.pth \
    --mode webcam
```

Controls:
- **SPACE**: Start/stop recording
- **R**: Reset buffer
- **Q**: Quit

### Batch Processing

```bash
python inference.py \
    --model_path ../models/checkpoint_best.pth \
    --mode batch \
    --video_dir ../../datasets/signVideos/Greetings/Hello \
    --output_file results.json
```

## 🔧 Troubleshooting

### Out of Memory Error

Reduce batch size or max frames:
```bash
python train.py --batch_size 8 --max_frames 40
```

### Slow Training

1. Enable preprocessing to cache features:
```bash
python train.py --preprocess
```

2. Reduce number of workers if disk I/O is slow:
```bash
python train.py --num_workers 2
```

3. Use GPU if available (automatic)

### Low Accuracy

1. Try a more powerful model:
```bash
python train.py --model_type hybrid
```

2. Increase model capacity:
```bash
python train.py --hidden_dim 512 --num_layers 4
```

3. Train longer:
```bash
python train.py --num_epochs 100 --patience 20
```

4. Reduce dropout if overfitting:
```bash
python train.py --dropout 0.2
```

### MediaPipe Import Error

```bash
pip install --upgrade mediapipe opencv-python
```

## 📊 Expected Performance

On a well-prepared Sinhala sign language dataset:

- **LSTM**: 70-85% accuracy
- **Transformer**: 75-90% accuracy  
- **Hybrid**: 80-92% accuracy

Performance depends on:
- Dataset quality and size
- Video quality and consistency
- Number of classes
- Training duration

## 🎯 Best Practices

1. **Preprocess first**: Use `--preprocess` flag to cache features
2. **Monitor training**: Use TensorBoard to watch for overfitting
3. **Use early stopping**: Prevents wasting time on converged models
4. **Save checkpoints**: Best model is automatically saved
5. **Test thoroughly**: Use validation set to tune hyperparameters, test set only for final evaluation

## 📝 Example Training Session

```bash
# Navigate to source directory
cd components/ssl-reader/src

# Run quick training
python quick_train.py

# Or manual training with specific settings
python train.py \
    --dataset_root ../../datasets/signVideos \
    --model_type hybrid \
    --batch_size 16 \
    --num_epochs 50 \
    --preprocess \
    --cache_dir ../data/processed

# Monitor with TensorBoard
tensorboard --logdir ../logs

# Test the trained model
python inference.py \
    --model_path ../models/checkpoint_best.pth \
    --mode webcam
```

## 🆘 Need Help?

Check the main component README for more details on the overall architecture and features.
