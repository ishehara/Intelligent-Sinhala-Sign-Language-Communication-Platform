"""
CNN Model Training for Sound Detection
Trains a convolutional neural network on MFCC features for audio classification.
"""

import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from datetime import datetime

# Configure GPU
try:
    # Enable GPU memory growth to avoid OOM errors
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"\n{'='*60}")
        print(f"GPU DETECTED: {len(gpus)} GPU(s) available")
        print(f"GPU Name: {gpus[0].name}")
        print(f"Training will use GPU acceleration")
        print(f"{'='*60}\n")
    else:
        print("\nNo GPU detected. Training will use CPU.\n")
except Exception as e:
    print(f"GPU configuration error: {e}")
    print("Continuing with default settings...\n")


class SoundClassifierCNN:
    """
    CNN-based sound classifier for audio detection.
    
    Args:
        n_mfcc: Number of MFCC coefficients (default: 13)
        n_frames: Number of time frames (default: 40)
        n_classes: Number of sound categories
    """
    
    def __init__(self, n_mfcc=13, n_frames=40, n_classes=5):
        self.n_mfcc = n_mfcc
        self.n_frames = n_frames
        self.n_classes = n_classes
        self.model = None
        self.history = None
        
    def build_model(self):
        """
        Build CNN architecture for sound classification.
        
        Architecture:
        - Conv2D layers for feature extraction
        - MaxPooling for dimensionality reduction
        - Dropout for regularization
        - Dense layers for classification
        """
        model = models.Sequential([
            # Input layer - reshape for CNN
            layers.Input(shape=(self.n_mfcc, self.n_frames, 1)),
            
            # First Convolutional Block
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.3),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Output Layer
            layers.Dense(self.n_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer and loss function.
        
        Args:
            learning_rate: Learning rate for Adam optimizer
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model compiled successfully!")
        print(f"Optimizer: Adam (lr={learning_rate})")
        print("Loss: Sparse Categorical Crossentropy")
        print("Metrics: Accuracy\n")
    
    def get_callbacks(self, model_dir, patience=15):
        """
        Create training callbacks.
        
        Args:
            model_dir: Directory to save model checkpoints
            patience: Patience for early stopping
            
        Returns:
            List of callbacks
        """
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        callback_list = [
            # Early stopping to prevent overfitting
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Save best model
            callbacks.ModelCheckpoint(
                filepath=str(model_path / 'best_model.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            callbacks.TensorBoard(
                log_dir=str(model_path / 'logs' / datetime.now().strftime("%Y%m%d-%H%M%S")),
                histogram_freq=1
            )
        ]
        
        return callback_list
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, model_dir='models'):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            model_dir: Directory to save models
            
        Returns:
            Training history
        """
        print("="*60)
        print("TRAINING CNN MODEL")
        print("="*60)
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
        print("="*60)
        
        # Get callbacks
        callback_list = self.get_callbacks(model_dir)
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60)
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get loss
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy*100:.2f}%")
        print("="*60)
        
        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        return results
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history (loss and accuracy).
        
        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            print("No training history available!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        ax1.plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        ax2.plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def save_model(self, save_path):
        """
        Save the trained model.
        
        Args:
            save_path: Path to save the model
        """
        self.model.save(save_path)
        print(f"Model saved to {save_path}")
    
    def print_summary(self):
        """Print model architecture summary."""
        self.model.summary()


def load_preprocessed_data(data_dir):
    """
    Load preprocessed data from numpy files.
    
    Args:
        data_dir: Directory containing .npy files
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, metadata)
    """
    data_path = Path(data_dir)
    
    print("="*60)
    print("LOADING PREPROCESSED DATA")
    print("="*60)
    print(f"Data directory: {data_path}")
    
    # Load arrays
    X_train = np.load(data_path / 'X_train.npy')
    X_test = np.load(data_path / 'X_test.npy')
    y_train = np.load(data_path / 'y_train.npy')
    y_test = np.load(data_path / 'y_test.npy')
    
    # Load metadata
    with open(data_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    with open(data_path / 'label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
    
    metadata['label_mapping'] = label_mapping
    
    print(f"\nData loaded successfully!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Number of classes: {metadata['n_classes']}")
    print(f"Classes: {list(label_mapping['encoder'].keys())}")
    print("="*60)
    
    return X_train, X_test, y_train, y_test, metadata


def reshape_for_cnn(X, n_mfcc, n_frames):
    """
    Reshape flattened MFCC features for CNN input.
    
    Args:
        X: Flattened features (samples, n_mfcc * n_frames)
        n_mfcc: Number of MFCC coefficients
        n_frames: Number of frames
        
    Returns:
        Reshaped features (samples, n_mfcc, n_frames, 1)
    """
    n_samples = X.shape[0]
    X_reshaped = X.reshape(n_samples, n_mfcc, n_frames, 1)
    return X_reshaped


def train_sound_classifier(data_dir, model_dir, epochs=100, batch_size=32, 
                          validation_split=0.15, learning_rate=0.001):
    """
    Complete training pipeline for sound classification.
    
    Args:
        data_dir: Directory with preprocessed data
        model_dir: Directory to save models and results
        epochs: Number of training epochs
        batch_size: Batch size
        validation_split: Proportion of training data for validation
        learning_rate: Learning rate
    """
    # Create model directory
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test, metadata = load_preprocessed_data(data_dir)
    
    # Reshape for CNN
    n_mfcc = metadata['n_mfcc']
    n_frames = metadata['n_frames']
    n_classes = metadata['n_classes']
    
    print("\nReshaping data for CNN...")
    X_train_cnn = reshape_for_cnn(X_train, n_mfcc, n_frames)
    X_test_cnn = reshape_for_cnn(X_test, n_mfcc, n_frames)
    print(f"Reshaped X_train: {X_train_cnn.shape}")
    print(f"Reshaped X_test: {X_test_cnn.shape}")
    
    # Split training data for validation
    n_val = int(len(X_train_cnn) * validation_split)
    X_val = X_train_cnn[-n_val:]
    y_val = y_train[-n_val:]
    X_train_final = X_train_cnn[:-n_val]
    y_train_final = y_train[:-n_val]
    
    print(f"\nFinal data split:")
    print(f"Training: {len(X_train_final)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Testing: {len(X_test_cnn)} samples")
    
    # Build model
    print("\n" + "="*60)
    print("BUILDING MODEL")
    print("="*60)
    classifier = SoundClassifierCNN(n_mfcc=n_mfcc, n_frames=n_frames, n_classes=n_classes)
    classifier.build_model()
    classifier.compile_model(learning_rate=learning_rate)
    classifier.print_summary()
    
    # Train model
    history = classifier.train(
        X_train_final, y_train_final,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        model_dir=str(model_path)
    )
    
    # Evaluate model
    results = classifier.evaluate(X_test_cnn, y_test)
    
    # Print classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    class_names = list(metadata['label_mapping']['encoder'].keys())
    # Get unique labels present in test set to avoid mismatch errors
    unique_labels = sorted(np.unique(np.concatenate([y_test, results['predictions']])))
    labels_in_test = [class_names[i] for i in unique_labels if i < len(class_names)]
    print(classification_report(y_test, results['predictions'], 
                               labels=unique_labels,
                               target_names=labels_in_test, digits=4))
    
    # Plot training history
    print("\nGenerating training history plots...")
    classifier.plot_training_history(save_path=str(model_path / 'training_history.png'))
    
    # Plot confusion matrix
    print("Generating confusion matrix...")
    classifier.plot_confusion_matrix(
        y_test, results['predictions'], class_names,
        save_path=str(model_path / 'confusion_matrix.png')
    )
    
    # Save final model
    print("\nSaving final model...")
    classifier.save_model(str(model_path / 'final_model.keras'))
    
    # Save results
    results_summary = {
        'test_accuracy': results['test_accuracy'],
        'test_loss': results['test_loss'],
        'n_classes': n_classes,
        'class_names': class_names,
        'training_samples': len(X_train_final),
        'validation_samples': len(X_val),
        'test_samples': len(X_test_cnn),
        'epochs_trained': len(history.history['loss']),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'final_train_accuracy': float(history.history['accuracy'][-1])
    }
    
    with open(model_path / 'training_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*60)
    print(f"✓ Model saved: {model_path / 'best_model.keras'}")
    print(f"✓ Results saved: {model_path / 'training_results.json'}")
    print(f"✓ Plots saved: {model_path}")
    print(f"\nFinal Test Accuracy: {results['test_accuracy']*100:.2f}%")
    if results['test_accuracy'] >= 0.85:
        print("✓ Target accuracy (85%) achieved!")
    else:
        print(f"⚠ Target accuracy not reached. Consider:")
        print("  - Increasing epochs")
        print("  - Adjusting learning rate")
        print("  - Adding data augmentation")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CNN for sound classification')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing preprocessed data (.npy files)')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory to save trained model')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--validation_split', type=float, default=0.15,
                       help='Validation split (default: 0.15)')
    
    args = parser.parse_args()
    
    train_sound_classifier(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        learning_rate=args.learning_rate
    )
