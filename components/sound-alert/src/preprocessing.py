"""
Audio Preprocessing Module for Sound Detection Component
Loads audio files, extracts MFCC features, and prepares data for training.
"""

import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class AudioPreprocessor:
    """
    Preprocessor for audio data that extracts MFCC features.
    
    Args:
        n_mfcc: Number of MFCC coefficients (default: 13)
        n_frames: Number of frames to extract (default: 40)
        sample_rate: Target sample rate for audio (default: 22050 Hz)
        duration: Duration to load from each audio file in seconds (default: 2.5)
    """
    
    def __init__(self, n_mfcc: int = 13, n_frames: int = 40, 
                 sample_rate: int = 22050, duration: float = 2.5):
        self.n_mfcc = n_mfcc
        self.n_frames = n_frames
        self.sample_rate = sample_rate
        self.duration = duration
        self.label_encoder = {}
        self.label_decoder = {}
        
    def load_audio_file(self, file_path: str) -> np.ndarray:
        """
        Load an audio file and return the waveform.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Audio waveform as numpy array
        """
        try:
            # Load audio file with specified sample rate and duration
            audio, sr = librosa.load(file_path, sr=self.sample_rate, 
                                    duration=self.duration, mono=True)
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio waveform.
        
        Args:
            audio: Audio waveform
            
        Returns:
            MFCC features with shape (n_mfcc, n_frames)
        """
        try:
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, 
                                       n_mfcc=self.n_mfcc)
            
            # Resize to fixed number of frames
            if mfcc.shape[1] < self.n_frames:
                # Pad if too short
                pad_width = self.n_frames - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
            else:
                # Truncate if too long
                mfcc = mfcc[:, :self.n_frames]
            
            return mfcc
        except Exception as e:
            print(f"Error extracting MFCC: {e}")
            return None
    
    def load_dataset_from_folders(self, root_dir: str, 
                                  audio_extensions: List[str] = ['.wav', '.mp3', '.flac']) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all audio files from categorized folders and extract features.
        
        Args:
            root_dir: Root directory containing category folders
            audio_extensions: List of valid audio file extensions
            
        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        features = []
        labels = []
        label_names = []
        
        root_path = Path(root_dir)
        
        if not root_path.exists():
            raise ValueError(f"Directory not found: {root_dir}")
        
        # Get all category folders
        category_folders = [f for f in root_path.iterdir() if f.is_dir()]
        
        if not category_folders:
            raise ValueError(f"No category folders found in {root_dir}")
        
        print(f"Found {len(category_folders)} categories in {root_dir}")
        
        # Process each category
        for category_idx, category_folder in enumerate(sorted(category_folders)):
            category_name = category_folder.name
            label_names.append(category_name)
            
            # Get all audio files in this category
            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(list(category_folder.glob(f"*{ext}")))
                audio_files.extend(list(category_folder.glob(f"*{ext.upper()}")))
            
            print(f"Processing '{category_name}': {len(audio_files)} files")
            
            # Process each audio file
            for audio_file in audio_files:
                # Load audio
                audio = self.load_audio_file(str(audio_file))
                if audio is None:
                    continue
                
                # Extract MFCC features
                mfcc = self.extract_mfcc(audio)
                if mfcc is None:
                    continue
                
                # Flatten MFCC to 1D vector
                mfcc_flat = mfcc.flatten()
                
                features.append(mfcc_flat)
                labels.append(category_idx)
        
        # Create label mappings
        self.label_encoder = {name: idx for idx, name in enumerate(label_names)}
        self.label_decoder = {idx: name for idx, name in enumerate(label_names)}
        
        # Convert to numpy arrays
        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        
        print(f"\nDataset loaded successfully!")
        print(f"Total samples: {len(features)}")
        print(f"Feature shape: {features.shape}")
        print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        print(f"Categories: {label_names}")
        
        return features, labels
    
    def split_dataset(self, features: np.ndarray, labels: np.ndarray, 
                     test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split dataset into training and testing sets.
        
        Args:
            features: Feature array
            labels: Label array
            test_size: Proportion of dataset for testing (default: 0.2)
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, 
            random_state=random_state, stratify=labels
        )
        
        print(f"\nDataset split complete!")
        print(f"Training samples: {len(X_train)} ({(1-test_size)*100:.0f}%)")
        print(f"Testing samples: {len(X_test)} ({test_size*100:.0f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, X_train: np.ndarray, X_test: np.ndarray,
                           y_train: np.ndarray, y_test: np.ndarray,
                           output_dir: str):
        """
        Save processed data as numpy arrays.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training labels
            y_test: Testing labels
            output_dir: Directory to save the files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save numpy arrays
        np.save(output_path / 'X_train.npy', X_train)
        np.save(output_path / 'X_test.npy', X_test)
        np.save(output_path / 'y_train.npy', y_train)
        np.save(output_path / 'y_test.npy', y_test)
        
        # Save label mappings
        label_mapping = {
            'encoder': self.label_encoder,
            'decoder': {str(k): v for k, v in self.label_decoder.items()},
            'n_classes': len(self.label_encoder)
        }
        
        with open(output_path / 'label_mapping.json', 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        # Save metadata
        metadata = {
            'n_mfcc': self.n_mfcc,
            'n_frames': self.n_frames,
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'feature_shape': X_train.shape[1],
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'n_classes': len(self.label_encoder)
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nData saved successfully to {output_path}/")
        print(f"Files created:")
        print(f"  - X_train.npy: {X_train.shape}")
        print(f"  - X_test.npy: {X_test.shape}")
        print(f"  - y_train.npy: {y_train.shape}")
        print(f"  - y_test.npy: {y_test.shape}")
        print(f"  - label_mapping.json")
        print(f"  - metadata.json")


def preprocess_audio_dataset(data_dir: str, output_dir: str, 
                            n_mfcc: int = 13, n_frames: int = 40,
                            test_size: float = 0.2):
    """
    Complete pipeline to preprocess audio dataset.
    
    Args:
        data_dir: Directory containing categorized audio files
        output_dir: Directory to save processed data
        n_mfcc: Number of MFCC coefficients
        n_frames: Number of frames
        test_size: Proportion for test set
    """
    print("="*60)
    print("Audio Dataset Preprocessing Pipeline")
    print("="*60)
    print(f"Input directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"MFCC coefficients: {n_mfcc}")
    print(f"Number of frames: {n_frames}")
    print(f"Test size: {test_size*100:.0f}%")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(n_mfcc=n_mfcc, n_frames=n_frames)
    
    # Load dataset
    print("\n[1/4] Loading audio files...")
    features, labels = preprocessor.load_dataset_from_folders(data_dir)
    
    # Split dataset
    print("\n[2/4] Splitting dataset...")
    X_train, X_test, y_train, y_test = preprocessor.split_dataset(
        features, labels, test_size=test_size
    )
    
    # Save processed data
    print("\n[3/4] Saving processed data...")
    preprocessor.save_processed_data(X_train, X_test, y_train, y_test, output_dir)
    
    print("\n[4/4] Complete!")
    print("="*60)
    print("Preprocessing completed successfully!")
    print("="*60)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess audio dataset for sound detection')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing categorized audio files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save processed data')
    parser.add_argument('--n_mfcc', type=int, default=13,
                       help='Number of MFCC coefficients (default: 13)')
    parser.add_argument('--n_frames', type=int, default=40,
                       help='Number of frames (default: 40)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set proportion (default: 0.2)')
    
    args = parser.parse_args()
    
    preprocess_audio_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_mfcc=args.n_mfcc,
        n_frames=args.n_frames,
        test_size=args.test_size
    )
