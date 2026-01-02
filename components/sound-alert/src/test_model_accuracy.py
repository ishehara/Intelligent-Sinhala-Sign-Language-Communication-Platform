"""
Test sound detection by playing audio files.
Analyzes pre-recorded audio files instead of microphone input.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from inference import SoundDetector


def main():
    """Test the model on audio files."""
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent.parent
    model_dir = base_dir / "components" / "sound-alert" / "models" / "vehicle_horns_cnn"
    data_dir = base_dir / "components" / "sound-alert" / "data" / "processed" / "vehicle_horns"
    dataset_dir = base_dir / "datasets" / "Vehicle Horns"
    
    print("="*70)
    print("SOUND DETECTION - FILE TESTING")
    print("="*70)
    
    # Initialize detector
    detector = SoundDetector(
        model_path=str(model_dir / "best_model.keras"),
        metadata_path=str(data_dir / "metadata.json"),
        label_mapping_path=str(data_dir / "label_mapping.json")
    )
    
    # Test each category with a few samples
    categories = {
        'bus horns': 3,
        'car horns': 3,
        'motorcycle horns': 3,
        'train horns': 3,
        'truck horns': 3
    }
    
    total_correct = 0
    total_tested = 0
    
    for category, num_samples in categories.items():
        category_path = dataset_dir / category
        
        if not category_path.exists():
            print(f"\n⚠️  Skipping {category}: not found")
            continue
        
        # Get audio files
        audio_files = list(category_path.glob("*.wav"))[:num_samples]
        
        if not audio_files:
            audio_files = list(category_path.glob("*.mp3"))[:num_samples]
        
        print(f"\n{'='*70}")
        print(f"Testing: {category.upper()}")
        print(f"{'='*70}")
        
        correct = 0
        
        for audio_file in audio_files:
            predicted, confidence, probs = detector.predict(str(audio_file))
            
            is_correct = predicted == category
            status = "✓" if is_correct else "✗"
            
            print(f"{status} {audio_file.name:30s} → {predicted:20s} ({confidence*100:.1f}%)")
            
            if is_correct:
                correct += 1
            total_tested += 1
        
        total_correct += correct
        accuracy = (correct / len(audio_files)) * 100 if audio_files else 0
        print(f"\nCategory accuracy: {correct}/{len(audio_files)} ({accuracy:.1f}%)")
    
    # Overall summary
    print("\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)
    overall_accuracy = (total_correct / total_tested) * 100 if total_tested > 0 else 0
    print(f"Total tested: {total_tested}")
    print(f"Correct: {total_correct}")
    print(f"Accuracy: {overall_accuracy:.2f}%")
    print("="*70)
    
    if overall_accuracy >= 90:
        print("\n✅ EXCELLENT! Model is working correctly!")
        print("\n💡 TIP: For microphone detection:")
        print("  1. Increase microphone volume in system settings")
        print("  2. Place microphone very close to speaker")
        print("  3. Play sounds at higher volume")
        print("  4. Reduce background noise")
    else:
        print("\n⚠️  Model may need retraining")


if __name__ == "__main__":
    main()
