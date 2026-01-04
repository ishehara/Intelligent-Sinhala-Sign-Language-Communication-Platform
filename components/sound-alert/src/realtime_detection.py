"""
Real-time Sound Detection using Microphone
Loads trained model and detects sounds in real-time
"""

import numpy as np
import sounddevice as sd
import librosa
import json
import argparse
import time
import os
from tensorflow import keras


class RealtimeSoundDetector:
    def __init__(self, model_path, metadata_path, label_mapping_path, 
                 sample_rate=22050, duration=2.5, n_mfcc=13, n_frames=40):
        """
        Initialize the real-time sound detector
        
        Args:
            model_path: Path to trained Keras model
            metadata_path: Path to metadata.json
            label_mapping_path: Path to label_mapping.json
            sample_rate: Audio sample rate
            duration: Recording duration in seconds
            n_mfcc: Number of MFCC coefficients
            n_frames: Number of time frames for MFCC
        """
        self.model = keras.models.load_model(model_path)
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.n_frames = n_frames
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load label mapping
        with open(label_mapping_path, 'r') as f:
            label_data = json.load(f)
            self.label_to_class = label_data['label_to_class']
            self.class_to_label = label_data['class_to_label']
        
        print(f"✓ Model loaded: {model_path}")
        print(f"✓ Classes: {list(self.class_to_label.keys())}")
        print(f"✓ Sample rate: {self.sample_rate} Hz")
        print(f"✓ Recording duration: {self.duration}s")
    
    def preprocess_audio(self, audio):
        """
        Preprocess audio to match training data characteristics
        
        Args:
            audio: Raw audio array
            
        Returns:
            Preprocessed audio array
        """
        # Convert to float32
        audio = audio.astype(np.float32)
        
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # Normalize amplitude
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        # Apply pre-emphasis filter to amplify high frequencies
        pre_emphasis = 0.97
        audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        # Simple noise gate - remove very quiet parts
        threshold = 0.01
        audio = np.where(np.abs(audio) < threshold, 0, audio)
        
        # Final normalization
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        return audio
    
    def extract_mfcc(self, audio):
        """
        Extract MFCC features from audio
        
        Args:
            audio: Audio array (numpy array)
            
        Returns:
            MFCC features as 1D array
        """
        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc
        )
        
        # Resize to fixed number of frames
        if mfcc.shape[1] < self.n_frames:
            # Pad with zeros if too short
            pad_width = self.n_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            # Truncate if too long
            mfcc = mfcc[:, :self.n_frames]
        
        # Flatten to 1D array
        mfcc_features = mfcc.flatten()
        
        return mfcc_features
    
    def record_audio(self, duration=None):
        """
        Record audio from microphone
        
        Args:
            duration: Recording duration (uses self.duration if None)
            
        Returns:
            Recorded audio as numpy array
        """
        if duration is None:
            duration = self.duration
            
        print(f"\n🎤 Recording for {duration} seconds...")
        
        # Record audio
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait until recording is finished
        
        # Convert to 1D array
        audio = audio.flatten()
        
        print("✓ Recording complete!")
        
        return audio
    
    def predict(self, audio):
        """
        Predict sound class from audio
        
        Args:
            audio: Audio array
            
        Returns:
            Tuple of (predicted_class, confidence, all_probabilities)
        """
        # Preprocess audio
        audio = self.preprocess_audio(audio)
        
        # Extract MFCC features
        mfcc_features = self.extract_mfcc(audio)
        
        # Reshape for model input
        mfcc_features = mfcc_features.reshape(1, -1)
        
        # Predict
        predictions = self.model.predict(mfcc_features, verbose=0)
        
        # Get predicted class
        predicted_label = np.argmax(predictions[0])
        predicted_class = self.label_to_class[str(predicted_label)]
        confidence = predictions[0][predicted_label]
        
        return predicted_class, confidence, predictions[0]
    
    def run_interactive(self, max_duration=60, show_all=False):
        """
        Run interactive mode - user controls when to start/stop recording
        
        Args:
            max_duration: Maximum recording duration in seconds
            show_all: Show all class probabilities
        """
        print("\n" + "="*70)
        print("🎙️  INTERACTIVE SOUND DETECTION MODE")
        print("="*70)
        print(f"\nDetectable sounds: {', '.join(self.class_to_label.keys())}")
        print(f"\nInstructions:")
        print("  1. Press ENTER to START recording")
        print("  2. Make a sound or play audio")
        print("  3. Press ENTER to STOP recording")
        print("  4. Type 'quit' to exit")
        print(f"\nMax recording duration: {max_duration} seconds")
        print("="*70 + "\n")
        
        detection_count = 0
        
        try:
            while True:
                # Wait for user to start
                user_input = input("Press ENTER to start recording (or 'quit' to exit): ")
                
                if user_input.lower() == 'quit':
                    print("\n👋 Exiting...")
                    break
                
                # Start recording
                start_time = time.time()
                print(f"\n🔴 RECORDING... Press ENTER to stop (max {max_duration}s)")
                
                # Record in a separate thread so we can listen for input
                recording = []
                chunk_duration = 0.1  # Record in 100ms chunks
                
                def callback(indata, frames, time_info, status):
                    recording.append(indata.copy())
                
                # Start streaming
                with sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype='float32',
                    callback=callback
                ):
                    # Wait for user to press ENTER or timeout
                    import threading
                    import sys
                    
                    stop_event = threading.Event()
                    
                    def wait_for_input():
                        input()
                        stop_event.set()
                    
                    input_thread = threading.Thread(target=wait_for_input)
                    input_thread.daemon = True
                    input_thread.start()
                    
                    # Wait for stop event or timeout
                    stop_event.wait(timeout=max_duration)
                    
                    elapsed_time = time.time() - start_time
                
                # Concatenate all recorded chunks
                if len(recording) > 0:
                    audio = np.concatenate(recording, axis=0).flatten()
                    
                    print(f"⏹️  Recording stopped ({elapsed_time:.1f}s)")
                    
                    # Make prediction
                    print("\n🔍 Analyzing sound...")
                    predicted_class, confidence, all_probs = self.predict(audio)
                    
                    detection_count += 1
                    
                    # Display results
                    print("\n" + "─"*70)
                    print(f"📊 DETECTION #{detection_count}")
                    print("─"*70)
                    print(f"🎯 Detected Sound: {predicted_class.upper()}")
                    print(f"📈 Confidence: {confidence*100:.2f}%")
                    
                    if show_all:
                        print(f"\n📋 All Probabilities:")
                        for class_name in sorted(self.class_to_label.keys()):
                            label = int(self.class_to_label[class_name])
                            prob = all_probs[label]
                            bar_length = int(prob * 40)
                            bar = "█" * bar_length + "░" * (40 - bar_length)
                            print(f"  {class_name:20s} {bar} {prob*100:5.2f}%")
                    
                    print("─"*70 + "\n")
                else:
                    print("⚠️  No audio recorded\n")
                
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
    
    def run_continuous(self, interval=3.0, show_all=False):
        """
        Run continuous detection mode - automatically detect every few seconds
        
        Args:
            interval: Time between detections in seconds
            show_all: Show all class probabilities
        """
        print("\n" + "="*70)
        print("🎙️  CONTINUOUS SOUND DETECTION MODE")
        print("="*70)
        print(f"\nDetectable sounds: {', '.join(self.class_to_label.keys())}")
        print(f"Detection interval: {interval} seconds")
        print("Press Ctrl+C to stop")
        print("="*70 + "\n")
        
        detection_count = 0
        
        try:
            while True:
                detection_count += 1
                
                # Record audio
                audio = self.record_audio(interval)
                
                # Make prediction
                print("🔍 Analyzing sound...")
                predicted_class, confidence, all_probs = self.predict(audio)
                
                # Display results
                print("\n" + "─"*70)
                print(f"📊 DETECTION #{detection_count}")
                print("─"*70)
                print(f"🎯 Detected Sound: {predicted_class.upper()}")
                print(f"📈 Confidence: {confidence*100:.2f}%")
                
                if show_all:
                    print(f"\n📋 All Probabilities:")
                    for class_name in sorted(self.class_to_label.keys()):
                        label = int(self.class_to_label[class_name])
                        prob = all_probs[label]
                        bar_length = int(prob * 40)
                        bar = "█" * bar_length + "░" * (40 - bar_length)
                        print(f"  {class_name:20s} {bar} {prob*100:5.2f}%")
                
                print("─"*70 + "\n")
                
                # Small pause before next detection
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n\n👋 Stopping detection...")


def main():
    parser = argparse.ArgumentParser(description='Real-time Sound Detection')
    
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing the trained model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing processed data (for metadata)')
    parser.add_argument('--mode', type=str, default='interactive',
                        choices=['interactive', 'continuous'],
                        help='Detection mode: interactive (user-controlled) or continuous (automatic)')
    parser.add_argument('--interval', type=float, default=3.0,
                        help='Detection interval for continuous mode (seconds)')
    parser.add_argument('--max_duration', type=int, default=60,
                        help='Maximum recording duration for interactive mode (seconds)')
    parser.add_argument('--show_all', action='store_true',
                        help='Show all class probabilities')
    
    args = parser.parse_args()
    
    # Build paths
    model_path = os.path.join(args.model_dir, 'best_model.keras')
    if not os.path.exists(model_path):
        model_path = os.path.join(args.model_dir, 'final_model.keras')
    
    metadata_path = os.path.join(args.data_dir, 'metadata.json')
    label_mapping_path = os.path.join(args.data_dir, 'label_mapping.json')
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print("Please train the model first using train_model.py")
        return
    
    if not os.path.exists(metadata_path):
        print(f"❌ Error: Metadata not found at {metadata_path}")
        return
    
    if not os.path.exists(label_mapping_path):
        print(f"❌ Error: Label mapping not found at {label_mapping_path}")
        return
    
    # Create detector
    detector = RealtimeSoundDetector(
        model_path=model_path,
        metadata_path=metadata_path,
        label_mapping_path=label_mapping_path
    )
    
    # Run detection
    if args.mode == 'interactive':
        detector.run_interactive(
            max_duration=args.max_duration,
            show_all=args.show_all
        )
    else:
        detector.run_continuous(
            interval=args.interval,
            show_all=args.show_all
        )


if __name__ == '__main__':
    main()
