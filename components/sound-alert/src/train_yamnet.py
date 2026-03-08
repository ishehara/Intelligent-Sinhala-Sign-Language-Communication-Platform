"""
YAMNet Transfer Learning — Classifier Head Training

Architecture:
  Frozen YAMNet (run at embedding extraction time)
       ↓
  1024-dim mean-pooled embedding  (loaded from .npy files)
       ↓
  Dense(256, relu) + BatchNorm + Dropout(0.4)
       ↓
  Dense(128, relu) + BatchNorm + Dropout(0.3)
       ↓
  Dense(n_classes, softmax)

Because YAMNet already encodes rich audio semantics, this tiny head
converges quickly and generalises well even from 5-15 real examples per class.
"""

import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg')   # headless — safe on machines without a display
import matplotlib.pyplot as plt

# ── GPU setup ─────────────────────────────────────────────────────────────────
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"\nGPU detected: {gpus[0].name}")
    else:
        print("\nNo GPU — training on CPU (fast because head is tiny)")
except Exception as e:
    print(f"GPU setup: {e}")

CLASS_NAMES = ['bus horns', 'car horns', 'train horns', 'truck horns',
               'ambulance', 'firetruck', 'police', 'traffic']


# ── Model ─────────────────────────────────────────────────────────────────────
def build_head(n_classes: int, embedding_dim: int = 1024) -> keras.Model:
    """
    Lightweight classifier head on top of frozen YAMNet embeddings.
    Input shape: (embedding_dim,)  e.g. (1024,)
    """
    model = keras.Sequential([
        layers.Input(shape=(embedding_dim,)),

        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(n_classes, activation='softmax'),
    ], name='yamnet_classifier_head')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ── Training ──────────────────────────────────────────────────────────────────
def train(data_dir: str, model_dir: str, epochs: int = 80, batch_size: int = 32):
    data_path  = Path(data_dir)
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    # Load pre-extracted embeddings
    print("\n" + "=" * 60)
    print("LOADING YAMNET EMBEDDINGS")
    print("=" * 60)

    X_train = np.load(data_path / 'X_train.npy')
    X_test  = np.load(data_path / 'X_test.npy')
    y_train = np.load(data_path / 'y_train.npy')
    y_test  = np.load(data_path / 'y_test.npy')

    with open(data_path / 'metadata.json') as f:
        meta = json.load(f)

    embedding_dim = meta.get('embedding_dim', 1024)
    n_classes     = meta.get('n_classes', len(CLASS_NAMES))

    print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"X_test : {X_test.shape}   y_test : {y_test.shape}")
    print(f"Classes: {n_classes}  |  Embedding dim: {embedding_dim}")

    # Validation split from training data
    val_size = 0.15
    val_n = max(1, int(len(X_train) * val_size))
    X_val, y_val = X_train[-val_n:], y_train[-val_n:]
    X_tr,  y_tr  = X_train[:-val_n],  y_train[:-val_n]

    print(f"\nTrain: {len(X_tr)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # Class weights
    unique = np.unique(y_tr)
    weights = compute_class_weight('balanced', classes=unique, y=y_tr)
    class_weight_dict = dict(zip(unique.tolist(), weights.tolist()))
    print("\nClass weights:")
    for cls, w in sorted(class_weight_dict.items()):
        print(f"  {CLASS_NAMES[cls]:20s}: {w:.4f}")

    # Build model
    print("\n" + "=" * 60)
    print("BUILDING CLASSIFIER HEAD")
    print("=" * 60)
    model = build_head(n_classes, embedding_dim)
    model.summary()

    # Callbacks
    cb_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=12,
                                restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(
            filepath=str(model_path / 'best_model.keras'),
            monitor='val_accuracy', save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=5, min_lr=1e-7, verbose=1),
    ]

    # Train
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        class_weight=class_weight_dict,
        verbose=1,
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss    : {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    # Only report classes present in test set
    present = sorted(np.unique(np.concatenate([y_test, y_pred])).tolist())
    target_names = [CLASS_NAMES[i] for i in present]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=present,
                                 target_names=target_names, digits=4))

    # Save model + metadata
    model.save(str(model_path / 'final_model.keras'))

    results = {
        'test_loss':     float(test_loss),
        'test_accuracy': float(test_acc),
        'n_classes':     n_classes,
        'class_names':   CLASS_NAMES,
        'model_type':    'yamnet_transfer',
        'trained_at':    datetime.now().isoformat(),
    }
    with open(model_path / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Plots
    _plot(history, model_path)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"✓ Best model : {model_path / 'best_model.keras'}")
    print(f"✓ Final model: {model_path / 'final_model.keras'}")
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
    print(f"\nNext step — real-time detection:")
    print(f"  python components\\sound-alert\\src\\realtime_yamnet.py \\")
    print(f"    --model_dir \"{model_dir}\" \\")
    print(f"    --data_dir  \"{data_dir}\"")

    return model, history


def _plot(history, model_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history['loss'],     label='Train', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val',   linewidth=2)
    ax1.set_title('Loss');  ax1.set_xlabel('Epoch'); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(history.history['accuracy'],     label='Train', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Val',   linewidth=2)
    ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = str(model_path / 'training_history.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Training plot: {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Train YAMNet classifier head')
    parser.add_argument('--data_dir',   default='components/sound-alert/data/processed/yamnet')
    parser.add_argument('--model_dir',  default='components/sound-alert/models/yamnet_cnn')
    parser.add_argument('--epochs',     type=int,   default=80)
    parser.add_argument('--batch_size', type=int,   default=32)
    args = parser.parse_args()

    train(args.data_dir, args.model_dir, args.epochs, args.batch_size)


if __name__ == '__main__':
    main()
