"""
Sound Detection Inference Script
Demonstrates how to use the trained model to detect sounds in audio files.
"""

import numpy as np
import json
import tensorflow as tf
import librosa
from pathlib import Path
import argparse
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class SoundDetector:
    """
    Sound detector using trained CNN model.
    
    Args:
        model_path: Path to trained model (.keras file)
        metadata_path: Path to metadata.json from preprocessing
        label_mapping_path: Path to label_mapping.json
    """
    
    def __init__(self, model_path: str, metadata_path: str, label_mapping_path: str):
        # Load model
        print("Loading trained model...")
        self.model = tf.keras.models.load_model(model_path)
        print(f"✓ Model loaded from {model_path}")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load label mapping
        with open(label_mapping_path, 'r') as f:
            label_info = json.load(f)
            self.label_decoder = {int(k): v for k, v in label_info['decoder'].items()}
            self.class_names = list(label_info['encoder'].keys())
        
        self.n_mfcc = self.metadata['n_mfcc']
        self.n_frames = self.metadata['n_frames']
        self.sample_rate = self.metadata['sample_rate']
        self.duration = self.metadata['duration']
        
        print(f"✓ Model configured for {len(self.class_names)} classes")
        print(f"  Classes: {', '.join(self.class_names)}")
        print(f"  MFCC coefficients: {self.n_mfcc}")
        print(f"  Time frames: {self.n_frames}\n")
    
    def extract_features(self, audio_path: str) -> np.ndarray:
        """
        Extract MFCC features from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            MFCC features with shape (1, n_mfcc, n_frames, 1)
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, 
                                duration=self.duration, mono=True)
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
        
        # Resize to fixed number of frames
        if mfcc.shape[1] < self.n_frames:
            pad_width = self.n_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :self.n_frames]
        
        # Reshape for CNN: (1, n_mfcc, n_frames, 1)
        mfcc = mfcc.reshape(1, self.n_mfcc, self.n_frames, 1)
        
        return mfcc
    
    def predict(self, audio_path: str) -> Tuple[str, float, np.ndarray]:
        """
        Predict sound class for an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (predicted_class, confidence, all_probabilities)
        """
        # Extract features
        features = self.extract_features(audio_path)
        
        # Predict
        probabilities = self.model.predict(features, verbose=0)[0]
        
        # Get prediction
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.label_decoder[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        return predicted_class, confidence, probabilities
    
    def predict_batch(self, audio_paths: List[str]) -> List[Tuple[str, float, np.ndarray]]:
        """
        Predict sound classes for multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            List of (predicted_class, confidence, all_probabilities) tuples
        """
        results = []
        for audio_path in audio_paths:
            result = self.predict(audio_path)
            results.append(result)
        return results
    
    def predict_with_details(self, audio_path: str, show_all_probs: bool = True):
        """
        Predict and display detailed results.
        
        Args:
            audio_path: Path to audio file
            show_all_probs: Show probabilities for all classes
        """
        print(f"\n{'='*70}")
        print(f"Analyzing: {Path(audio_path).name}")
        print(f"{'='*70}")
        
        # Predict
        predicted_class, confidence, probabilities = self.predict(audio_path)
        
        # Display results
        print(f"\n🎯 PREDICTION: {predicted_class}")
        print(f"📊 CONFIDENCE: {confidence*100:.2f}%")
        
        if show_all_probs:
            print(f"\n📈 All Class Probabilities:")
            print(f"{'-'*70}")
            # Sort by probability
            sorted_indices = np.argsort(probabilities)[::-1]
            for idx in sorted_indices:
                class_name = self.label_decoder[idx]
                prob = probabilities[idx]
                bar = '█' * int(prob * 50)
                print(f"  {class_name:20s} {prob*100:6.2f}% {bar}")
        
        print(f"{'='*70}\n")
        
        return predicted_class, confidence


def test_on_folder(detector: SoundDetector, folder_path: str, 
                  expected_class: str = None, max_files: int = 10):
    """
    Test the detector on all audio files in a folder.
    
    Args:
        detector: SoundDetector instance
        folder_path: Path to folder containing audio files
        expected_class: Expected class name (for accuracy calculation)
        max_files: Maximum number of files to test
    """
    folder = Path(folder_path)
    audio_extensions = ['.wav', '.mp3', '.flac']
    
    # Get audio files
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(list(folder.glob(f'*{ext}')))
        audio_files.extend(list(folder.glob(f'*{ext.upper()}')))
    
    audio_files = audio_files[:max_files]
    
    if not audio_files:
        print(f"⚠️  No audio files found in {folder_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"Testing on folder: {folder.name}")
    if expected_class:
        print(f"Expected class: {expected_class}")
    print(f"Number of files: {len(audio_files)}")
    print(f"{'='*70}\n")
    
    correct = 0
    results = []
    
    for i, audio_file in enumerate(audio_files, 1):
        predicted_class, confidence, _ = detector.predict(str(audio_file))
        
        is_correct = predicted_class == expected_class if expected_class else None
        status = "✓" if is_correct else "✗" if is_correct is False else "?"
        
        print(f"{status} [{i}/{len(audio_files)}] {audio_file.name:40s} → "
              f"{predicted_class:20s} ({confidence*100:.1f}%)")
        
        if is_correct:
            correct += 1
        
        results.append({
            'file': audio_file.name,
            'predicted': predicted_class,
            'confidence': confidence,
            'correct': is_correct
        })
    
    # Summary
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Total files tested: {len(audio_files)}")
    if expected_class:
        accuracy = (correct / len(audio_files)) * 100
        print(f"Correct predictions: {correct}/{len(audio_files)}")
        print(f"Accuracy: {accuracy:.2f}%")
    
    avg_confidence = np.mean([r['confidence'] for r in results])
    print(f"Average confidence: {avg_confidence*100:.2f}%")
    print(f"{'='*70}\n")
    
    return results


def demonstrate_detection(model_dir: str, data_dir: str):
    """
    Demonstrate sound detection on test samples.
    
    Args:
        model_dir: Directory containing trained model
        data_dir: Directory containing preprocessed data (for metadata)
    """
    model_path = Path(model_dir) / 'best_model.keras'
    metadata_path = Path(data_dir) / 'metadata.json'
    label_mapping_path = Path(data_dir) / 'label_mapping.json'
    
    # Check if files exist
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    if not metadata_path.exists():
        print(f"❌ Metadata not found: {metadata_path}")
        return
    
    # Initialize detector
    print(f"\n{'='*70}")
    print("SOUND DETECTION DEMONSTRATION")
    print(f"{'='*70}\n")
    
    detector = SoundDetector(
        model_path=str(model_path),
        metadata_path=str(metadata_path),
        label_mapping_path=str(label_mapping_path)
    )
    
    return detector


def main():
    parser = argparse.ArgumentParser(description='Sound Detection Inference')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing preprocessing metadata')
    parser.add_argument('--audio_file', type=str,
                       help='Single audio file to test')
    parser.add_argument('--test_folder', type=str,
                       help='Folder containing audio files to test')
    parser.add_argument('--expected_class', type=str,
                       help='Expected class name (for folder testing)')
    parser.add_argument('--max_files', type=int, default=10,
                       help='Maximum files to test from folder')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = demonstrate_detection(args.model_dir, args.data_dir)
    
    if args.audio_file:
        # Test single file
        detector.predict_with_details(args.audio_file)
    
    elif args.test_folder:
        # Test folder
        test_on_folder(detector, args.test_folder, 
                      args.expected_class, args.max_files)
    
    else:
        print("\n⚠️  Please specify --audio_file or --test_folder")
        print("\nExample usage:")
        print("  Single file:")
        print("    python inference.py --model_dir ../models/vehicle_horns_cnn \\")
        print("                        --data_dir ../data/processed/vehicle_horns \\")
        print("                        --audio_file path/to/audio.wav")
        print("\n  Test folder:")
        print("    python inference.py --model_dir ../models/vehicle_horns_cnn \\")
        print("                        --data_dir ../data/processed/vehicle_horns \\")
        print("                        --test_folder ../../datasets/Vehicle\\ Horns/car\\ horns \\")
        print("                        --expected_class \"car horns\"")


if __name__ == "__main__":
    main()
