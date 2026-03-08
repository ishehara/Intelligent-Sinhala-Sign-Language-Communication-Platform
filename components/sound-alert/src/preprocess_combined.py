"""
Combined Audio Preprocessing for Horns and Sirens
Processes both vehicle horns and siren sounds for unified detection model
"""

import os
import shutil
import argparse
from pathlib import Path


def create_combined_dataset(horns_dir, sirens_dir, output_dir):
    """
    Create a combined dataset structure with both horns and sirens
    
    Args:
        horns_dir: Path to Vehicle Horns directory
        sirens_dir: Path to sirens directory
        output_dir: Path to output combined dataset
    """
    # Create output directory
    combined_dir = Path(output_dir)
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("CREATING COMBINED DATASET: HORNS + SIRENS")
    print("="*70)
    
    # Copy vehicle horns
    horns_path = Path(horns_dir)
    if horns_path.exists():
        print(f"\n📁 Processing Vehicle Horns from: {horns_dir}")
        for category_folder in horns_path.iterdir():
            if category_folder.is_dir():
                dest_folder = combined_dir / category_folder.name
                if dest_folder.exists():
                    print(f"  ⚠️  {category_folder.name} already exists, skipping...")
                else:
                    shutil.copytree(category_folder, dest_folder)
                    file_count = len(list(dest_folder.glob('*.*')))
                    print(f"  ✓ {category_folder.name}: {file_count} files")
    else:
        print(f"⚠️  Vehicle Horns directory not found: {horns_dir}")
    
    # Copy sirens
    sirens_path = Path(sirens_dir)
    if sirens_path.exists():
        print(f"\n📁 Processing Sirens from: {sirens_dir}")
        for category_folder in sirens_path.iterdir():
            if category_folder.is_dir():
                dest_folder = combined_dir / category_folder.name
                if dest_folder.exists():
                    print(f"  ⚠️  {category_folder.name} already exists, skipping...")
                else:
                    shutil.copytree(category_folder, dest_folder)
                    file_count = len(list(dest_folder.glob('*.*')))
                    print(f"  ✓ {category_folder.name}: {file_count} files")
    else:
        print(f"⚠️  Sirens directory not found: {sirens_dir}")
    
    # Summary
    print("\n" + "="*70)
    print("COMBINED DATASET STRUCTURE")
    print("="*70)
    
    total_files = 0
    categories = []
    for category_folder in combined_dir.iterdir():
        if category_folder.is_dir():
            file_count = len(list(category_folder.glob('*.*')))
            total_files += file_count
            categories.append(category_folder.name)
            print(f"  {category_folder.name:25s}: {file_count:5d} files")
    
    print("="*70)
    print(f"Total Categories: {len(categories)}")
    print(f"Total Files: {total_files}")
    print(f"Output Directory: {combined_dir}")
    print("="*70)
    
    return str(combined_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Combine Vehicle Horns and Sirens datasets'
    )
    
    parser.add_argument('--horns_dir', type=str,
                       default='datasets/Vehicle Horns',
                       help='Path to Vehicle Horns dataset')
    parser.add_argument('--sirens_dir', type=str,
                       default='datasets/sirens',
                       help='Path to sirens dataset')
    parser.add_argument('--output_dir', type=str,
                       default='datasets/combined_alert_sounds',
                       help='Output directory for combined dataset')
    
    args = parser.parse_args()
    
    # Create combined dataset
    combined_path = create_combined_dataset(
        args.horns_dir,
        args.sirens_dir,
        args.output_dir
    )
    
    print(f"\n✅ Combined dataset created successfully!")
    print(f"\nNext steps:")
    print(f"1. Run preprocessing:")
    print(f"   python components\\sound-alert\\src\\preprocessing.py \\")
    print(f"     --data_dir \"{combined_path}\" \\")
    print(f"     --output_dir \"components/sound-alert/data/processed/alert_sounds\"")
    print(f"\n2. Train the model:")
    print(f"   python components\\sound-alert\\src\\train_model.py \\")
    print(f"     --data_dir \"components/sound-alert/data/processed/alert_sounds\" \\")
    print(f"     --model_dir \"components/sound-alert/models/alert_sounds_cnn\"")


if __name__ == '__main__':
    main()
