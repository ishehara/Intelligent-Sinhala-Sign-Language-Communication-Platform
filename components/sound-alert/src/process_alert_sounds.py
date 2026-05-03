"""
Process Multiple Sound Datasets for Combined Training
Processes horns and sirens datasets into 9 individual classes:
  Horns (5):  bus horns, car horns, motorcycle horns, train horns, truck horns
  Sirens (4): ambulance, firetruck, police, traffic
 
Sometimes we don't have enough audio files for the computer to learn properly. 
This script takes the existing sounds and creates new, fake variations of them 
(by adding background noise, making them faster, or changing the pitch). 
This gives the computer a much larger dataset to practice with.
"""

import os
import sys
import numpy as np
import json
import argparse
from pathlib import Path

# Add the current directory to path
sys.path.append(str(Path(__file__).parent))

from preprocessing import AudioPreprocessor

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️  librosa not available — augmentation disabled")

AUG_TARGET = 400  # minimum samples per class after augmentation
AUG_MAX_MULTIPLIER = 3  # never augment a class by more than 3x its real sample count

"""Mapping of dataset folder names to canonical class names."""
# Maps folder name (lowercase) → canonical class name
CATEGORY_MAP = {
    # Horns
    'bus horns':        'bus horns',
    'car horns':        'car horns',
    'train horns':      'train horns',
    'truck horns':      'truck horns',
    # Sirens (with or without " siren" suffix)
    'ambulance':        'ambulance',
    'ambulance siren':  'ambulance',
    'firetruck':        'firetruck',
    'firetruck siren':  'firetruck',
    'police':           'police',
    'police siren':     'police',
    'traffic':          'traffic',
    'traffic siren':    'traffic',
}

# Fixed ordered class list (label index = position in list)
# motorcycle horns excluded — no audio files available
CLASS_NAMES = ['bus horns', 'car horns', 'train horns', 'truck horns',
               'ambulance', 'firetruck', 'police', 'traffic']
CLASS_TO_LABEL = {name: idx for idx, name in enumerate(CLASS_NAMES)}

"""Augment features by applying random transformations to the raw audio."""

def augment_features(features, n_mfcc, n_frames, target_count, preprocessor, raw_audios, sr):
    """
    Generate augmented MFCC features from raw audio until target_count is reached.
    Augmentations: noise, time-stretch, pitch-shift.
    Falls back to simple feature jitter if librosa unavailable.
    """
    augmented = list(features)
    needed = target_count - len(features)
    if needed <= 0:
        return features

    if LIBROSA_AVAILABLE and len(raw_audios) > 0:
        rng = np.random.default_rng(42)
        idx = 0
        while len(augmented) < target_count:
            audio = raw_audios[idx % len(raw_audios)].copy()
            choice = idx % 4
            try:
                if choice == 0:  # add noise
                    noise = rng.normal(0, 0.005, len(audio))
                    audio = audio + noise
                elif choice == 1:  # time stretch
                    rate = rng.uniform(0.85, 1.15)
                    audio = librosa.effects.time_stretch(audio, rate=rate)
                elif choice == 2:  # pitch shift
                    steps = rng.uniform(-2, 2)
                    audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)
                else:  # combined noise + pitch
                    noise = rng.normal(0, 0.003, len(audio))
                    audio = audio + noise
                    steps = rng.uniform(-1, 1)
                    audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)
                mfcc = preprocessor.extract_mfcc(audio)
                if mfcc is not None:
                    augmented.append(mfcc.flatten())
            except Exception:
                pass
            idx += 1
    else:
        # Fallback: add small random noise to existing features
        rng = np.random.default_rng(42)
        while len(augmented) < target_count:
            base = features[len(augmented) % len(features)]
            noise = rng.normal(0, 0.01 * np.std(base), base.shape)
            augmented.append(base + noise)

    return np.array(augmented[:target_count])


def process_combined_datasets(dataset_dirs, dataset_names, output_dir,
                              n_mfcc=13, n_frames=40, test_size=0.2):
    """
    Process multiple datasets into 9 individual classes.
    """
    print("="*70)
    print("PROCESSING COMBINED DATASETS: HORNS + SIRENS  →  9 CLASSES")
    print("="*70)

    preprocessor = AudioPreprocessor(n_mfcc=n_mfcc, n_frames=n_frames)

    all_features = []
    all_labels = []

    for dataset_dir, dataset_name in zip(dataset_dirs, dataset_names):
        if not os.path.exists(dataset_dir):
            print(f"\n⚠️  Dataset not found: {dataset_dir}")
            continue

        print(f"\n{'='*70}")
        print(f"Processing: {dataset_name}")
        print(f"Directory: {dataset_dir}")
        print(f"{'='*70}")

        # Load each category folder individually so we can remap to horn/siren
        dataset_path = Path(dataset_dir)
        for category_folder in sorted(dataset_path.iterdir()):
            if not category_folder.is_dir():
                continue

            folder_name = category_folder.name.lower()

            if folder_name not in CATEGORY_MAP:
                print(f"  ⚠️  Unknown category '{folder_name}', skipping")
                continue

            group_name = CATEGORY_MAP[folder_name]
            group_label = CLASS_TO_LABEL[group_name]

            # Load audio files in this folder
            audio_files = [f for f in category_folder.iterdir()
                           if f.suffix.lower() in ['.wav', '.mp3', '.flac']]

            if len(audio_files) == 0:
                print(f"  ⚠️  {folder_name}: 0 files, skipping")
                continue

            folder_features = []
            folder_raw_audios = []
            for audio_file in audio_files:
                audio = preprocessor.load_audio_file(str(audio_file))
                if audio is None:
                    continue
                mfcc = preprocessor.extract_mfcc(audio)
                if mfcc is None:
                    continue
                folder_features.append(mfcc.flatten())
                folder_raw_audios.append(audio)

            if len(folder_features) == 0:
                continue

            original_count = len(folder_features)
            folder_features = np.array(folder_features)

            # Augment only up to AUG_MAX_MULTIPLIER × real samples, capped at AUG_TARGET
            # This prevents the model from memorising synthetic noise from tiny datasets
            if original_count < AUG_TARGET:
                capped_target = min(AUG_TARGET, original_count * AUG_MAX_MULTIPLIER)
                if capped_target > original_count:
                    folder_features = augment_features(
                        folder_features, n_mfcc, n_frames, capped_target,
                        preprocessor, folder_raw_audios, preprocessor.sample_rate
                    )
                    print(f"  ✓ {folder_name:25s} → {group_name:20s}  ({original_count} → {len(folder_features)} samples, augmented ≤{AUG_MAX_MULTIPLIER}x)")
                else:
                    print(f"  ⚠ {folder_name:25s} → {group_name:20s}  ({original_count} samples — too few to augment safely, included as-is)")
            else:
                print(f"  ✓ {folder_name:25s} → {group_name:20s}  ({original_count} samples)")

            folder_labels = np.full(len(folder_features), group_label, dtype=np.int32)

            all_features.append(folder_features)
            all_labels.append(folder_labels)
    
    # Combine all data
    if len(all_features) == 0:
        print("\n❌ No data was loaded from any dataset!")
        return

    print(f"\n{'='*70}")
    print("COMBINING DATASETS")
    print(f"{'='*70}")

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)

    print(f"\nCombined Dataset:")
    print(f"  Total samples : {len(X)}")
    for idx, name in enumerate(CLASS_NAMES):
        count = int(np.sum(y == idx))
        print(f"  {name:20s} (label {idx}): {count}")
    print(f"  Feature shape  : {X.shape}")

    # Split data
    print(f"\n{'='*70}")
    print(f"SPLITTING DATA (test_size={test_size})")
    print(f"{'='*70}")

    X_train, X_test, y_train, y_test = preprocessor.split_dataset(X, y, test_size=test_size)

    print(f"\nTraining samples: {len(X_train)} ({(1-test_size)*100:.0f}%)")
    print(f"Testing samples : {len(X_test)} ({test_size*100:.0f}%)")

    # ── Per-coefficient MFCC z-score normalization ──────────────────────────
    # Compute mean and std from X_train only, then apply to both splits.
    # This standardizes each of the n_mfcc coefficients independently so the
    # CNN sees zero-mean, unit-variance features regardless of recording condition.
    # The saved stats are loaded at inference time for consistent transformation.
    X_train_rs = X_train.reshape(len(X_train), n_mfcc, n_frames)
    mfcc_mean = X_train_rs.mean(axis=(0, 2))          # shape: (n_mfcc,)
    mfcc_std  = X_train_rs.std(axis=(0, 2)) + 1e-8   # shape: (n_mfcc,)

    def _normalize(arr):
        Xr = arr.reshape(len(arr), n_mfcc, n_frames)
        Xr = (Xr - mfcc_mean[None, :, None]) / mfcc_std[None, :, None]
        return Xr.reshape(len(arr), -1).astype(np.float32)

    X_train = _normalize(X_train)
    X_test  = _normalize(X_test)
    print(f"  ✓ MFCC features normalized (per-coefficient z-score, computed from training set)")
    # ────────────────────────────────────────────────────────────────────────

    # Create label mapping (9 classes)
    label_mapping = {
        'label_to_class': {str(idx): name for idx, name in enumerate(CLASS_NAMES)},
        'class_to_label': CLASS_TO_LABEL
    }

    # Save processed data
    print(f"\n{'='*70}")
    print("SAVING PROCESSED DATA")
    print(f"{'='*70}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / 'X_train.npy', X_train)
    np.save(output_path / 'X_test.npy', X_test)
    np.save(output_path / 'y_train.npy', y_train)
    np.save(output_path / 'y_test.npy', y_test)

    print(f"✓ X_train.npy: {X_train.shape}")
    print(f"✓ X_test.npy : {X_test.shape}")
    print(f"✓ y_train.npy: {y_train.shape}")
    print(f"✓ y_test.npy : {y_test.shape}")

    # Save normalization stats alongside data files so inference can apply
    # the identical transformation without re-loading training data.
    np.save(output_path / 'mfcc_mean.npy', mfcc_mean)
    np.save(output_path / 'mfcc_std.npy', mfcc_std)
    print(f"✓ mfcc_mean.npy / mfcc_std.npy  (normalization stats)")

    # Save metadata
    metadata = {
        'n_mfcc': n_mfcc,
        'n_frames': n_frames,
        'sample_rate': preprocessor.sample_rate,
        'duration': preprocessor.duration,
        'n_classes': len(CLASS_NAMES),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'feature_dim': X_train.shape[1],
        'datasets': dataset_names,
        'mfcc_mean': mfcc_mean.tolist(),
        'mfcc_std': mfcc_std.tolist()
    }

    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ metadata.json")

    with open(output_path / 'label_mapping.json', 'w') as f:
        json.dump(label_mapping, f, indent=2)
    print(f"✓ label_mapping.json")

    print(f"\n{'='*70}")
    print("✅ PROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nOutput directory: {output_path}")
    print(f"\nNext step: Train the model")
    print(f"  python components\\sound-alert\\src\\train_model.py \\")
    print(f"    --data_dir \"{output_dir}\" \\")
    print(f"    --model_dir \"components/sound-alert/models/alert_sounds_cnn\" \\")
    print(f"    --epochs 100")


def main():
    parser = argparse.ArgumentParser(
        description='Process multiple sound datasets for combined training'
    )
    
    parser.add_argument('--horns_dir', type=str,
                       default='datasets/Vehicle Horns',
                       help='Path to Vehicle Horns dataset')
    parser.add_argument('--sirens_dir', type=str,
                       default='datasets/sirens',
                       help='Path to sirens dataset')
    parser.add_argument('--output_dir', type=str,
                       default='components/sound-alert/data/processed/alert_sounds',
                       help='Output directory for processed data')
    parser.add_argument('--n_mfcc', type=int, default=13,
                       help='Number of MFCC coefficients')
    parser.add_argument('--n_frames', type=int, default=40,
                       help='Number of time frames')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test split ratio (0.0 to 1.0)')
    
    args = parser.parse_args()
    
    # Define datasets to process
    dataset_dirs = [args.horns_dir, args.sirens_dir]
    dataset_names = ['Vehicle Horns', 'Sirens']
    
    # Process combined datasets
    process_combined_datasets(
        dataset_dirs=dataset_dirs,
        dataset_names=dataset_names,
        output_dir=args.output_dir,
        n_mfcc=args.n_mfcc,
        n_frames=args.n_frames,
        test_size=args.test_size
    )


if __name__ == '__main__':
    main()
