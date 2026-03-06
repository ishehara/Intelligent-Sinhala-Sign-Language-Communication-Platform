"""
Process Multiple Sound Datasets for Combined Training
Processes horns and sirens datasets, groups them into 2 classes: horn / siren
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

# All horn folder names → label 0 "horn"
HORN_CATEGORIES = {'bus horns', 'car horns', 'motorcycle horns', 'train horns', 'truck horns'}
# All siren folder names → label 1 "siren"
SIREN_CATEGORIES = {'ambulance', 'firetruck', 'police', 'traffic'}

CLASS_NAMES = ['horn', 'siren']


def process_combined_datasets(dataset_dirs, dataset_names, output_dir,
                              n_mfcc=13, n_frames=40, test_size=0.2):
    """
    Process multiple datasets and group into 2 classes: horn / siren
    """
    print("="*70)
    print("PROCESSING COMBINED DATASETS: HORNS + SIRENS  →  2 CLASSES")
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

            if folder_name in HORN_CATEGORIES:
                group_label = 0  # horn
                group_name = 'horn'
            elif folder_name in SIREN_CATEGORIES:
                group_label = 1  # siren
                group_name = 'siren'
            else:
                print(f"  ⚠️  Unknown category '{folder_name}', skipping")
                continue

            # Load audio files in this folder
            audio_files = [f for f in category_folder.iterdir()
                           if f.suffix.lower() in ['.wav', '.mp3', '.flac']]

            if len(audio_files) == 0:
                print(f"  ⚠️  {folder_name}: 0 files, skipping")
                continue

            folder_features = []
            for audio_file in audio_files:
                audio = preprocessor.load_audio_file(str(audio_file))
                if audio is None:
                    continue
                mfcc = preprocessor.extract_mfcc(audio)
                if mfcc is None:
                    continue
                folder_features.append(mfcc.flatten())

            if len(folder_features) == 0:
                continue

            folder_features = np.array(folder_features)
            folder_labels = np.full(len(folder_features), group_label, dtype=np.int32)

            all_features.append(folder_features)
            all_labels.append(folder_labels)

            print(f"  ✓ {folder_name:25s} → {group_name:5s}  ({len(folder_features)} samples)")
    
    # Combine all data
    if len(all_features) == 0:
        print("\n❌ No data was loaded from any dataset!")
        return

    print(f"\n{'='*70}")
    print("COMBINING DATASETS")
    print(f"{'='*70}")

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)

    horn_count = int(np.sum(y == 0))
    siren_count = int(np.sum(y == 1))

    print(f"\nCombined Dataset:")
    print(f"  Total samples : {len(X)}")
    print(f"  horn  (label 0): {horn_count}")
    print(f"  siren (label 1): {siren_count}")
    print(f"  Feature shape  : {X.shape}")

    # Split data
    print(f"\n{'='*70}")
    print(f"SPLITTING DATA (test_size={test_size})")
    print(f"{'='*70}")

    X_train, X_test, y_train, y_test = preprocessor.split_dataset(X, y, test_size=test_size)

    print(f"\nTraining samples: {len(X_train)} ({(1-test_size)*100:.0f}%)")
    print(f"Testing samples : {len(X_test)} ({test_size*100:.0f}%)")

    # Create label mapping  (2 classes: horn / siren)
    label_mapping = {
        'label_to_class': {'0': 'horn', '1': 'siren'},
        'class_to_label': {'horn': 0, 'siren': 1}
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

    # Save metadata
    metadata = {
        'n_mfcc': n_mfcc,
        'n_frames': n_frames,
        'sample_rate': preprocessor.sample_rate,
        'duration': preprocessor.duration,
        'n_classes': 2,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'feature_dim': X.shape[1],
        'datasets': dataset_names
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
