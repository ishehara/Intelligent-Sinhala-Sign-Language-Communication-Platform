"""
Quick demo script to test the trained sound detection model.
Tests the model on samples from each category in the dataset.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from components.sound_alert.src.inference import demonstrate_detection, test_on_folder


def main():
    """Run comprehensive testing on all sound categories."""
    
    # Define paths
    base_dir = Path(__file__).parent.parent.parent.parent
    model_dir = base_dir / "components" / "sound-alert" / "models" / "vehicle_horns_cnn"
    data_dir = base_dir / "components" / "sound-alert" / "data" / "processed" / "vehicle_horns"
    dataset_dir = base_dir / "datasets" / "Vehicle Horns"
    
    print("="*70)
    print("SOUND DETECTION MODEL - DEMONSTRATION")
    print("="*70)
    
    # Initialize detector
    detector = demonstrate_detection(str(model_dir), str(data_dir))
    
    if detector is None:
        print("\n❌ Failed to initialize detector. Make sure model is trained!")
        return
    
    # Test on each category
    categories = [
        'bus horns',
        'car horns',
        'motorcycle horns',
        'train horns',
        'truck horns'
    ]
    
    print("\n" + "="*70)
    print("TESTING MODEL ON ALL CATEGORIES")
    print("="*70)
    
    overall_correct = 0
    overall_total = 0
    
    for category in categories:
        category_path = dataset_dir / category
        
        if category_path.exists():
            print(f"\n\n{'#'*70}")
            print(f"# TESTING: {category.upper()}")
            print(f"{'#'*70}")
            
            results = test_on_folder(
                detector, 
                str(category_path), 
                expected_class=category,
                max_files=20
            )
            
            if results:
                correct = sum(1 for r in results if r['correct'])
                overall_correct += correct
                overall_total += len(results)
        else:
            print(f"\n⚠️  Skipping {category}: Directory not found")
    
    # Overall summary
    if overall_total > 0:
        print("\n" + "="*70)
        print("OVERALL PERFORMANCE")
        print("="*70)
        print(f"Total samples tested: {overall_total}")
        print(f"Correct predictions: {overall_correct}")
        print(f"Overall accuracy: {(overall_correct/overall_total)*100:.2f}%")
        print("="*70)
        
        if (overall_correct/overall_total) >= 0.85:
            print("\n✅ EXCELLENT! Model achieves >85% accuracy!")
        elif (overall_correct/overall_total) >= 0.75:
            print("\n✓ GOOD! Model shows strong performance.")
        else:
            print("\n⚠️  Model needs improvement. Consider more training.")


if __name__ == "__main__":
    main()
