"""
YAMNet Embedding Extraction for Transfer Learning
Loads audio files, passes them through the frozen YAMNet model,
and saves the resulting 1024-dim embeddings as numpy arrays for training.

YAMNet was trained on Google AudioSet (~2M clips, 521 classes) which includes
vehicle horns and sirens — making it an ideal feature extractor for this task.

Requirements:
    pip install tensorflow-hub
"""

import os
import sys
import numpy as np
import json
import argparse
from pathlib import Path

import tensorflow as tf
import tensorflow_hub as hub

try:
    import librosa
except ImportError:
    print("ERROR: librosa not found. Run: pip install librosa")
    sys.exit(1)

from sklearn.model_selection import train_test_split

# ── Class definitions (same as the MFCC pipeline) ────────────────────────────
CATEGORY_MAP = {
    'bus horns':       'bus horns',
    'car horns':       'car horns',
    'train horns':     'train horns',
    'truck horns':     'truck horns',
    'ambulance':       'ambulance',
    'ambulance siren': 'ambulance',
    'firetruck':       'firetruck',
    'firetruck siren': 'firetruck',
    'police':          'police',
    'police siren':    'police',
    'traffic':         'traffic',
    'traffic siren':   'traffic',
}

CLASS_NAMES = ['bus horns', 'car horns', 'train horns', 'truck horns',
               'ambulance', 'firetruck', 'police', 'traffic']
CLASS_TO_LABEL = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# YAMNet requires 16 kHz mono audio
YAMNET_SR = 16000

# ── Augmentation ──────────────────────────────────────────────────────────────
AUG_TARGET = 400           # target samples per class
AUG_MAX_MULTIPLIER = 3     # never create more than 3x real data


def augment_waveforms(waveforms: list, target_count: int) -> list:
    """Return a new list of waveforms augmented to target_count using librosa."""
    augmented = list(waveforms)
    rng = np.random.default_rng(42)
    idx = 0
    while len(augmented) < target_count:
        audio = waveforms[idx % len(waveforms)].copy()
        choice = idx % 4
        try:
            if choice == 0:
                audio = audio + rng.normal(0, 0.005, len(audio)).astype(np.float32)
            elif choice == 1:
                rate = float(rng.uniform(0.85, 1.15))
                audio = librosa.effects.time_stretch(audio, rate=rate)
            elif choice == 2:
                steps = float(rng.uniform(-2, 2))
                audio = librosa.effects.pitch_shift(audio, sr=YAMNET_SR, n_steps=steps)
            else:
                audio = audio + rng.normal(0, 0.003, len(audio)).astype(np.float32)
                steps = float(rng.uniform(-1, 1))
                audio = librosa.effects.pitch_shift(audio, sr=YAMNET_SR, n_steps=steps)
            augmented.append(audio.astype(np.float32))
        except Exception:
            augmented.append(audio.astype(np.float32))
        idx += 1
    return augmented[:target_count]


def load_and_resample(file_path: str, duration: float = 3.0) -> np.ndarray:
    """Load an audio file, resample to 16 kHz, return float32 mono waveform."""
    try:
        audio, _ = librosa.load(file_path, sr=YAMNET_SR, duration=duration, mono=True)
        return audio.astype(np.float32)
    except Exception as e:
        print(f"    ⚠️  Could not load {Path(file_path).name}: {e}")
        return None


def extract_embedding(yamnet_model, waveform: np.ndarray) -> np.ndarray:
    """
    Run waveform through YAMNet and return a single 1024-dim embedding
    by mean-pooling across all 0.96 s windows.
    """
    # YAMNet expects a 1-D float32 tensor
    waveform_tensor = tf.constant(waveform, dtype=tf.float32)
    scores, embeddings, _ = yamnet_model(waveform_tensor)
    # embeddings shape: (n_windows, 1024) — average over time
    return embeddings.numpy().mean(axis=0)          # → (1024,)


def process_datasets(dataset_dirs, dataset_names, output_dir, test_size=0.2):
    print("=" * 70)
    print("YAMNET TRANSFER LEARNING — EMBEDDING EXTRACTION")
    print("=" * 70)

    # Load YAMNet from TF-Hub (downloads ~27 MB once, cached locally)
    print("\nLoading YAMNet from TF-Hub (first run downloads ~27 MB)...")
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    print("✓ YAMNet loaded\n")

    all_embeddings = []
    all_labels = []

    for dataset_dir, dataset_name in zip(dataset_dirs, dataset_names):
        if not os.path.exists(dataset_dir):
            print(f"⚠️  Dataset not found: {dataset_dir}")
            continue

        print(f"\n{'=' * 70}")
        print(f"Processing: {dataset_name}  ({dataset_dir})")
        print(f"{'=' * 70}")

        for folder in sorted(Path(dataset_dir).iterdir()):
            if not folder.is_dir():
                continue
            folder_key = folder.name.lower()
            if folder_key not in CATEGORY_MAP:
                print(f"  ⚠️  Unknown folder '{folder.name}', skipping")
                continue

            class_name = CATEGORY_MAP[folder_key]
            label = CLASS_TO_LABEL[class_name]

            audio_files = [f for f in folder.iterdir()
                           if f.suffix.lower() in ('.wav', '.mp3', '.flac')]
            if not audio_files:
                print(f"  ⚠️  {folder.name}: 0 audio files, skipping")
                continue

            # Load waveforms
            waveforms = []
            for af in audio_files:
                w = load_and_resample(str(af))
                if w is not None and len(w) > YAMNET_SR * 0.3:
                    waveforms.append(w)

            if not waveforms:
                print(f"  ⚠️  {folder.name}: all files failed to load, skipping")
                continue

            original_count = len(waveforms)

            # Augment small classes (capped at AUG_MAX_MULTIPLIER × real data)
            if original_count < AUG_TARGET:
                capped = min(AUG_TARGET, original_count * AUG_MAX_MULTIPLIER)
                if capped > original_count:
                    waveforms = augment_waveforms(waveforms, capped)
                    aug_note = f"augmented → {len(waveforms)}"
                else:
                    aug_note = f"too few to augment safely ({original_count})"
            else:
                aug_note = f"{original_count} samples"

            # Extract YAMNet embedding for each waveform
            embeddings = []
            for i, w in enumerate(waveforms):
                emb = extract_embedding(yamnet_model, w)
                embeddings.append(emb)

            all_embeddings.extend(embeddings)
            all_labels.extend([label] * len(embeddings))
            print(f"  ✓ {folder.name:25s} → {class_name:20s}  ({aug_note})")

    if not all_embeddings:
        print("\n❌ No embeddings extracted!")
        return

    X = np.array(all_embeddings, dtype=np.float32)   # (N, 1024)
    y = np.array(all_labels,    dtype=np.int32)

    print(f"\n{'=' * 70}")
    print("DATASET SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total samples  : {len(X)}")
    for idx, name in enumerate(CLASS_NAMES):
        count = int(np.sum(y == idx))
        print(f"  {name:20s} (label {idx}): {count}")
    print(f"  Embedding shape : {X.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
    print(f"\n  Train: {len(X_train)}  |  Test: {len(X_test)}")

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / 'X_train.npy', X_train)
    np.save(out / 'X_test.npy',  X_test)
    np.save(out / 'y_train.npy', y_train)
    np.save(out / 'y_test.npy',  y_test)

    label_mapping = {
        'label_to_class': {str(i): n for i, n in enumerate(CLASS_NAMES)},
        'class_to_label': CLASS_TO_LABEL,
    }
    metadata = {
        'embedding_dim': 1024,
        'n_classes': len(CLASS_NAMES),
        'sample_rate': YAMNET_SR,
        'train_samples': len(X_train),
        'test_samples':  len(X_test),
        'model_type': 'yamnet_transfer',
        'yamnet_url': 'https://tfhub.dev/google/yamnet/1',
    }

    with open(out / 'metadata.json',      'w') as f: json.dump(metadata,      f, indent=2)
    with open(out / 'label_mapping.json', 'w') as f: json.dump(label_mapping, f, indent=2)

    print(f"\n✅ Saved to: {out}")
    print(f"   X_train: {X_train.shape}  |  X_test: {X_test.shape}")
    print(f"\nNext step:")
    print(f"  python components\\sound-alert\\src\\train_yamnet.py \\")
    print(f"    --data_dir \"{output_dir}\" \\")
    print(f"    --model_dir \"components/sound-alert/models/yamnet_cnn\"")


def main():
    parser = argparse.ArgumentParser(description='Extract YAMNet embeddings from audio datasets')
    parser.add_argument('--horns_dir',  default='datasets/Vehicle Horns')
    parser.add_argument('--sirens_dir', default='datasets/Siren')
    parser.add_argument('--output_dir', default='components/sound-alert/data/processed/yamnet')
    parser.add_argument('--test_size',  type=float, default=0.2)
    args = parser.parse_args()

    process_datasets(
        dataset_dirs=[args.horns_dir, args.sirens_dir],
        dataset_names=['Vehicle Horns', 'Sirens'],
        output_dir=args.output_dir,
        test_size=args.test_size,
    )


if __name__ == '__main__':
    main()
