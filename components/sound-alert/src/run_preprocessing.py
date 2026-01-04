"""
Run audio preprocessing for all sound datasets.
This script processes Vehicle Horns and Sirens datasets.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from components.sound_alert.src.preprocessing import preprocess_audio_dataset


def main():
    """Process all sound datasets."""
    
    # Define base paths
    base_dir = Path(__file__).parent.parent.parent.parent
    datasets_dir = base_dir / "datasets"
    output_base = base_dir / "components" / "sound-alert" / "data" / "processed"
    
    # Dataset configurations
    datasets = [
        {
            'name': 'Vehicle Horns',
            'path': datasets_dir / "Vehicle Horns",
            'output': output_base / "vehicle_horns"
        },
        {
            'name': 'Sirens',
            'path': datasets_dir / "sirens",
            'output': output_base / "sirens"
        }
    ]
    
    print("="*70)
    print("SOUND DETECTION - BATCH PREPROCESSING")
    print("="*70)
    
    for dataset_config in datasets:
        dataset_name = dataset_config['name']
        dataset_path = dataset_config['path']
        output_path = dataset_config['output']
        
        if not dataset_path.exists():
            print(f"\n⚠️  Skipping '{dataset_name}': Directory not found at {dataset_path}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*70}")
        
        try:
            preprocess_audio_dataset(
                data_dir=str(dataset_path),
                output_dir=str(output_path),
                n_mfcc=13,
                n_frames=40,
                test_size=0.2
            )
        except Exception as e:
            print(f"\n❌ Error processing '{dataset_name}': {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*70)
    print("BATCH PREPROCESSING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
