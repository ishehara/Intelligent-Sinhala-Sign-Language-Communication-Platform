"""
Real-Time Sound Detection using Microphone
Captures audio from microphone and detects sound category in real-time.
"""

import numpy as np
import json
import tensorflow as tf
import librosa
import sounddevice as sd
from pathlib import Path
import argparse
import time
import sys
import threading
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class RealtimeSoundDetector:
    """
    Real-time sound detector using microphone input.
    """
    
    def __init__(self, model_path: str, metadata_path: str, label_mapping_path: str):
        # Load model
        print("Loading trained model...")
        self.model = tf.keras.models.load_model(model_path)
        print(f"✓ Model loaded successfully\n")
        
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
        
        print("="*70)
        print("REAL-TIME SOUND DETECTOR")
        print("="*70)
        print(f"Detecting: {', '.join(self.class_names)}")
        print(f"Sample Rate: {self.sample_rate} Hz")
        print(f"Duration: {self.duration} seconds")
        print("="*70)
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio to match training data quality.
        Applies normalization, noise reduction, and signal enhancement.
        
        Args:
            audio: Raw audio waveform
            
        Returns:
            Preprocessed audio waveform
        """
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # Normalize amplitude to [-1, 1] range
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        
        # Apply pre-emphasis filter to amplify high frequencies
        # This helps with MFCC feature extraction
        pre_emphasis = 0.97
        audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        # Simple noise gate - remove very low amplitude signals (likely noise)
        noise_threshold = 0.01  # Adjust this value if needed
        audio = np.where(np.abs(audio) < noise_threshold, 0, audio)
        
        # Normalize again after noise gate
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        
        return audio
    
    def extract_features_from_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio array.
        
        Args:
            audio: Audio waveform
            
        Returns:
            MFCC features with shape (1, n_mfcc, n_frames, 1)
        """
        # Preprocess audio to improve quality
        audio = self.preprocess_audio(audio)
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        
        # Resize to fixed number of frames
        if mfcc.shape[1] < self.n_frames:
            pad_width = self.n_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :self.n_frames]
        
        # Reshape for CNN
        mfcc = mfcc.reshape(1, self.n_mfcc, self.n_frames, 1)
        
        return mfcc
    
    def predict_from_audio(self, audio: np.ndarray):
        """
        Predict sound class from audio array.
        
        Args:
            audio: Audio waveform
            
        Returns:
            Tuple of (predicted_class, confidence, all_probabilities)
        """
        # Extract features
        features = self.extract_features_from_audio(audio)
        
        # Predict
        probabilities = self.model.predict(features, verbose=0)[0]
        
        # Get prediction
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.label_decoder[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        return predicted_class, confidence, probabilities
    
    def record_audio(self):
        """
        Record audio from microphone.
        
        Returns:
            Audio waveform as numpy array
        """
        print("\n🎤 Recording audio...", end='', flush=True)
        
        # Record audio
        audio = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait until recording is finished
        
        # Convert to mono if needed
        audio = audio.flatten()
        
        print(" Done!")
        
        return audio
    
    def display_prediction(self, predicted_class: str, confidence: float, 
                          probabilities: np.ndarray, show_all: bool = True):
        """
        Display prediction results in a nice format.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        print("\n" + "="*70)
        print(f"⏰ Time: {timestamp}")
        print("="*70)
        print(f"\n🔊 DETECTED SOUND: {predicted_class.upper()}")
        print(f"📊 CONFIDENCE: {confidence*100:.2f}%")
        
        # Confidence bar
        bar_length = int(confidence * 50)
        confidence_bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"\n{confidence_bar} {confidence*100:.1f}%")
        
        if show_all:
            print(f"\n📈 All Probabilities:")
            print("-" * 70)
            sorted_indices = np.argsort(probabilities)[::-1]
            for idx in sorted_indices:
                class_name = self.label_decoder[idx]
                prob = probabilities[idx]
                bar = '█' * int(prob * 40)
                marker = "👉" if idx == sorted_indices[0] else "  "
                print(f"{marker} {class_name:20s} {prob*100:6.2f}% {bar}")
        
        print("="*70)
    
    def display_detection_summary(self, detections_history: list, threshold: float):
        """
        Display summary of all detections.
        
        Args:
            detections_history: List of all detections
            threshold: Confidence threshold used
        """
        if not detections_history:
            print("\n📊 DETECTION SUMMARY")
            print("="*70)
            print("No sounds detected above threshold.")
            print("="*70)
            return
        
        print("\n📊 DETECTION SUMMARY")
        print("="*70)
        print(f"Total detections: {len(detections_history)}")
        print(f"Confidence threshold: {threshold*100:.0f}%")
        print("="*70)
        
        # Count detections by class
        class_counts = {}
        class_confidences = {}
        
        for detection in detections_history:
            cls = detection['class']
            conf = detection['confidence']
            
            if cls not in class_counts:
                class_counts[cls] = 0
                class_confidences[cls] = []
            
            class_counts[cls] += 1
            class_confidences[cls].append(conf)
        
        # Display detected sounds
        print("\n🔊 DETECTED SOUNDS:")
        print("-" * 70)
        
        # Sort by count (most frequent first)
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        
        for cls, count in sorted_classes:
            avg_confidence = np.mean(class_confidences[cls]) * 100
            max_confidence = np.max(class_confidences[cls]) * 100
            percentage = (count / len(detections_history)) * 100
            
            bar = '█' * int(percentage / 2)
            
            print(f"\n  {cls.upper()}")
            print(f"    Count: {count} ({percentage:.1f}%)")
            print(f"    Avg Confidence: {avg_confidence:.2f}%")
            print(f"    Max Confidence: {max_confidence:.2f}%")
            print(f"    {bar}")
        
        # Display timeline
        print("\n📅 DETECTION TIMELINE:")
        print("-" * 70)
        
        for i, detection in enumerate(detections_history[-10:], 1):  # Show last 10
            timestamp = detection['timestamp']
            cls = detection['class']
            conf = detection['confidence'] * 100
            print(f"  [{timestamp}] {cls:20s} ({conf:.1f}%)")
        
        if len(detections_history) > 10:
            print(f"  ... and {len(detections_history) - 10} more earlier detections")
        
        # Overall statistics
        print("\n📈 STATISTICS:")
        print("-" * 70)
        
        all_confidences = [d['confidence'] for d in detections_history]
        avg_overall = np.mean(all_confidences) * 100
        max_overall = np.max(all_confidences) * 100
        min_overall = np.min(all_confidences) * 100
        
        print(f"  Average Confidence: {avg_overall:.2f}%")
        print(f"  Highest Confidence: {max_overall:.2f}%")
        print(f"  Lowest Confidence: {min_overall:.2f}%")
        
        # Most common detection
        most_common = sorted_classes[0]
        print(f"\n  Most Detected: {most_common[0].upper()} ({most_common[1]} times)")
        
        print("="*70)
    
    def run_continuous(self, confidence_threshold: float = 0.5, 
                      show_all_probs: bool = True, delay: float = 1.0):
        """
        Run continuous sound detection.
        
        Args:
            confidence_threshold: Minimum confidence to display prediction
            show_all_probs: Show all class probabilities
            delay: Delay between recordings in seconds
        """
        print("\n" + "="*70)
        print("🎙️  STARTING CONTINUOUS SOUND DETECTION")
        print("="*70)
        print(f"Confidence threshold: {confidence_threshold*100:.0f}%")
        print(f"Recording interval: {self.duration}s")
        print("\nPress Ctrl+C to stop")
        print("="*70)
        
        detection_count = 0
        detections_history = []  # Store all detections
        
        try:
            while True:
                # Record audio
                audio = self.record_audio()
                
                # Predict
                predicted_class, confidence, probabilities = self.predict_from_audio(audio)
                
                # Display if confidence is high enough
                if confidence >= confidence_threshold:
                    detection_count += 1
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    # Store detection
                    detections_history.append({
                        'timestamp': timestamp,
                        'class': predicted_class,
                        'confidence': confidence,
                        'probabilities': probabilities
                    })
                    
                    self.display_prediction(predicted_class, confidence, 
                                          probabilities, show_all_probs)
                else:
                    print(f"⚪ Low confidence: {predicted_class} ({confidence*100:.1f}%) - Ignored")
                
                # Wait before next recording
                if delay > 0:
                    time.sleep(delay)
        
        except KeyboardInterrupt:
            print("\n\n" + "="*70)
            print("🛑 DETECTION STOPPED")
            print("="*70)
            
            # Display summary
            self.display_detection_summary(detections_history, confidence_threshold)
    
    def run_single(self, show_all_probs: bool = True):
        """
        Run single sound detection.
        
        Args:
            show_all_probs: Show all class probabilities
        """
        print("\n" + "="*70)
        print("🎙️  SINGLE SOUND DETECTION")
        print("="*70)
        print(f"Recording duration: {self.duration} seconds")
        print("Get ready to make a sound!")
        print("="*70)
        
        # Countdown
        for i in range(3, 0, -1):
            print(f"\nStarting in {i}...", flush=True)
            time.sleep(1)
        
        # Record and predict
        audio = self.record_audio()
        predicted_class, confidence, probabilities = self.predict_from_audio(audio)
        
        # Display results
        self.display_prediction(predicted_class, confidence, probabilities, show_all_probs)
    
    def run_interactive(self, show_all_probs: bool = True, max_duration: float = 30.0):
        """
        Run interactive sound detection.
        User controls when to start and stop listening.
        
        Args:
            show_all_probs: Show all class probabilities
            max_duration: Maximum recording duration in seconds (default: 30s)
        """
        print("\n" + "="*70)
        print("🎙️  INTERACTIVE SOUND DETECTION")
        print("="*70)
        print(f"Max recording duration: {max_duration} seconds")
        print("\nControls:")
        print("  Press ENTER to START listening")
        print("  Press ENTER again to STOP recording")
        print("  Type 'quit' or 'exit' to end the session")
        print("="*70)
        
        detection_count = 0
        detections_history = []
        
        try:
            while True:
                # Wait for user to press Enter
                user_input = input("\n🎤 Press ENTER to start listening (or 'quit' to exit): ").strip().lower()
                
                if user_input in ['quit', 'exit', 'q']:
                    break
                
                # Record audio with manual stop capability
                print("\n🔴 RECORDING... Press ENTER to stop")
                print("=" * 70)
                
                # Use a flag to track if user stopped recording
                stop_flag = [False]
                audio_chunks = []
                start_time = time.time()
                
                def record_audio():
                    """Record audio in background thread."""
                    chunk_size = int(self.sample_rate * 0.1)  # 100ms chunks
                    max_chunks = int(max_duration / 0.1)
                    
                    for i in range(max_chunks):
                        if stop_flag[0]:
                            break
                        chunk = sd.rec(chunk_size, samplerate=self.sample_rate, 
                                     channels=1, dtype='float32')
                        sd.wait()
                        audio_chunks.append(chunk.flatten())
                        
                        # Show elapsed time every second
                        if (i + 1) % 10 == 0:
                            elapsed = time.time() - start_time
                            print(f"\r⏱️  Recording: {elapsed:.1f}s (Press ENTER to stop)", end='', flush=True)
                
                def wait_for_stop():
                    """Wait for user to press Enter to stop."""
                    input()
                    stop_flag[0] = True
                
                # Start recording in background
                record_thread = threading.Thread(target=record_audio)
                stop_thread = threading.Thread(target=wait_for_stop)
                
                record_thread.start()
                stop_thread.start()
                
                # Wait for recording to finish or be stopped
                record_thread.join()
                
                if not stop_flag[0]:
                    # Recording completed full duration
                    stop_thread.join(timeout=0.1)
                
                # Combine audio chunks
                if audio_chunks:
                    audio = np.concatenate(audio_chunks)
                    duration_recorded = len(audio) / self.sample_rate
                    print(f"\n✓ Recording stopped! ({duration_recorded:.1f} seconds captured)")
                else:
                    print("\n⚠️  No audio captured!")
                    continue
                
                # Ensure minimum duration for MFCC extraction
                min_length = int(self.sample_rate * 1.0)  # At least 1 second
                if len(audio) < min_length:
                    print(f"⚠️  Recording too short ({duration_recorded:.1f}s). Need at least 1 second.")
                    continue
                
                # Use the actual recorded audio length for MFCC extraction
                # Update duration temporarily for this prediction
                original_duration = self.duration
                self.duration = duration_recorded
                
                # Analyze the sound
                print("🔍 Analyzing sound...")
                predicted_class, confidence, probabilities = self.predict_from_audio(audio)
                
                # Restore original duration
                self.duration = original_duration
                
                # Store detection
                detection_count += 1
                timestamp = datetime.now().strftime("%H:%M:%S")
                detections_history.append({
                    'timestamp': timestamp,
                    'class': predicted_class,
                    'confidence': confidence,
                    'probabilities': probabilities,
                    'duration': duration_recorded
                })
                
                # Display results
                self.display_prediction(predicted_class, confidence, probabilities, show_all_probs)
                print(f"\n📏 Recording length: {duration_recorded:.1f} seconds")
        
        except KeyboardInterrupt:
            print("\n")
        
        # Display summary
        print("\n" + "="*70)
        print("🛑 SESSION ENDED")
        print("="*70)
        self.display_detection_summary(detections_history, 0.0)


def main():
    parser = argparse.ArgumentParser(description='Real-time Sound Detection using Microphone')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing preprocessing metadata')
    parser.add_argument('--mode', type=str, choices=['single', 'continuous', 'interactive'], default='interactive',
                       help='Detection mode: single, continuous, or interactive')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold for continuous mode (0.0-1.0)')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between recordings in continuous mode (seconds)')
    parser.add_argument('--max_duration', type=float, default=30.0,
                       help='Maximum recording duration in interactive mode (seconds)')
    parser.add_argument('--show_all', action='store_true',
                       help='Show probabilities for all classes')
    
    args = parser.parse_args()
    
    # Construct paths
    model_path = Path(args.model_dir) / 'best_model.keras'
    metadata_path = Path(args.data_dir) / 'metadata.json'
    label_mapping_path = Path(args.data_dir) / 'label_mapping.json'
    
    # Check if files exist
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        print("Make sure training is complete!")
        sys.exit(1)
    
    if not metadata_path.exists():
        print(f"❌ Error: Metadata not found at {metadata_path}")
        print("Make sure preprocessing was done!")
        sys.exit(1)
    
    # Initialize detector
    detector = RealtimeSoundDetector(
        model_path=str(model_path),
        metadata_path=str(metadata_path),
        label_mapping_path=str(label_mapping_path)
    )
    
    # Run detection
    if args.mode == 'single':
        detector.run_single(show_all_probs=args.show_all)
    elif args.mode == 'interactive':
        detector.run_interactive(show_all_probs=args.show_all, max_duration=args.max_duration)
    else:
        detector.run_continuous(
            confidence_threshold=args.threshold,
            show_all_probs=args.show_all,
            delay=args.delay
        )


if __name__ == "__main__":
    main()
