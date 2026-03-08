"""
Real-Time Environmental Sound Detection using YAMNet Transfer Learning

At inference the pipeline is:
  mic waveform (16 kHz)
       ↓
  frozen YAMNet  →  1024-dim mean-pooled embedding
       ↓
  trained classifier head  →  8-class softmax

No MFCC, no normalization stats, no preprocessing mismatch issues.
"""

import numpy as np
import json
import threading
import time
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import tensorflow as tf
import tensorflow_hub as hub
import sounddevice as sd

warnings.filterwarnings('ignore')

# YAMNet requires 16 kHz audio
YAMNET_SR = 16000

CLASS_NAMES = ['bus horns', 'car horns', 'train horns', 'truck horns',
               'ambulance', 'firetruck', 'police', 'traffic']


class YAMNetDetector:

    def __init__(self, model_dir: str, data_dir: str):
        print("\n" + "=" * 70)
        print("LOADING YAMNET ENVIRONMENTAL SOUND DETECTOR")
        print("=" * 70)

        # Load YAMNet feature extractor
        print("Loading YAMNet from TF-Hub (cached after first download)...")
        self.yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
        print("✓ YAMNet loaded")

        # Load trained classifier head
        model_path = Path(model_dir)
        best = model_path / 'best_model.keras'
        final = model_path / 'final_model.keras'
        chosen = best if best.exists() else final
        if not chosen.exists():
            raise FileNotFoundError(
                f"No trained model found in {model_dir}. Run train_yamnet.py first.")
        self.head = tf.keras.models.load_model(str(chosen))
        print(f"✓ Classifier head loaded: {chosen.name}")

        # Load label mapping
        label_file = Path(data_dir) / 'label_mapping.json'
        with open(label_file) as f:
            lm = json.load(f)
        self.label_decoder = {int(k): v for k, v in lm['label_to_class'].items()}
        self.class_names = list(lm['class_to_label'].keys())
        self.n_classes = len(self.class_names)

        print(f"✓ {self.n_classes} classes: {', '.join(self.class_names)}")
        print("=" * 70)

    # ── Feature extraction ────────────────────────────────────────────────────
    def _extract_embedding(self, waveform: np.ndarray) -> np.ndarray:
        """Pass waveform through frozen YAMNet, return mean-pooled 1024-dim embedding."""
        wt = tf.constant(waveform.astype(np.float32))
        _, embeddings, _ = self.yamnet(wt)
        return embeddings.numpy().mean(axis=0)   # (1024,)

    def _preprocess_and_embed(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalise raw mic audio and extract YAMNet embedding.
        No MFCC, no z-score stats needed — YAMNet handles feature extraction.
        """
        # DC offset removal + amplitude normalisation
        audio = audio - np.mean(audio)
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val

        # Trim leading/trailing silence
        try:
            import librosa
            trimmed, _ = librosa.effects.trim(audio, top_db=40)
            if len(trimmed) >= int(YAMNET_SR * 0.3):
                audio = trimmed
        except Exception:
            pass

        emb = self._extract_embedding(audio)
        return emb.reshape(1, -1)    # (1, 1024) for model.predict

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict(self, audio: np.ndarray):
        features = self._preprocess_and_embed(audio)
        probs = self.head.predict(features, verbose=0)[0]
        idx = int(np.argmax(probs))
        return self.label_decoder[idx], float(probs[idx]), probs

    @staticmethod
    def rms_energy(audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio ** 2)))

    # ── Display ───────────────────────────────────────────────────────────────
    def display(self, predicted_class: str, confidence: float,
                probs: np.ndarray, show_all: bool = True):
        ts = datetime.now().strftime("%H:%M:%S")
        print("\n" + "=" * 70)
        print(f"⏰ {ts}")
        print("=" * 70)
        print(f"\n🔊 DETECTED: {predicted_class.upper()}")
        bar = "█" * int(confidence * 50) + "░" * (50 - int(confidence * 50))
        print(f"📊 CONFIDENCE: {confidence * 100:.2f}%")
        print(f"\n{bar} {confidence * 100:.1f}%")

        if show_all:
            print(f"\n📈 All Probabilities:")
            print("-" * 70)
            for i in np.argsort(probs)[::-1]:
                name = self.label_decoder[i]
                p = probs[i]
                marker = "👉" if i == int(np.argmax(probs)) else "  "
                b = '█' * int(p * 40)
                print(f"{marker} {name:20s} {p * 100:6.2f}%  {b}")
        print("=" * 70)

    # ── Interactive mode ──────────────────────────────────────────────────────
    def run_interactive(self, show_all: bool = True, max_duration: float = 30.0):
        print("\n" + "=" * 70)
        print("🎙️  ENVIRONMENTAL SOUND DETECTION  (YAMNet)")
        print("=" * 70)
        print(f"Max recording: {max_duration}s")
        print("  ENTER → start  |  ENTER → stop  |  'quit' → exit")
        print("=" * 70)

        detections = []

        try:
            while True:
                user_input = input("\n🎤 Press ENTER to start (or 'quit'): ").strip().lower()
                if user_input in ('quit', 'exit', 'q'):
                    break

                print("\n🔴 RECORDING... Press ENTER to stop")
                print("=" * 70)

                audio_chunks = []
                start_time = time.time()
                stop_event = threading.Event()

                def audio_callback(indata, frames, time_info, status):
                    audio_chunks.append(indata[:, 0].copy())
                    elapsed = time.time() - start_time
                    if int(elapsed * 10) % 10 == 0:
                        print(f"\r⏱️  Recording: {elapsed:.1f}s", end='', flush=True)
                    if elapsed >= max_duration:
                        stop_event.set()

                stream = sd.InputStream(
                    samplerate=YAMNET_SR,
                    channels=1,
                    dtype='float32',
                    blocksize=int(YAMNET_SR * 0.1),
                    callback=audio_callback,
                )

                stop_thread = threading.Thread(
                    target=lambda: (input(), stop_event.set()), daemon=True)
                stop_thread.start()

                with stream:
                    stop_event.wait()

                if not audio_chunks:
                    print("\n⚠️  No audio captured!")
                    continue

                audio = np.concatenate(audio_chunks)
                audio_chunks.clear()
                dur = len(audio) / YAMNET_SR
                print(f"\n✓ Stopped ({dur:.1f}s captured)")

                if dur < 0.5:
                    print("⚠️  Too short (need ≥ 0.5s)")
                    continue

                rms = self.rms_energy(audio)
                RMS_THRESHOLD = 0.0001
                print(f"🔊 Audio RMS = {rms:.6f}  (threshold = {RMS_THRESHOLD})")
                if rms < RMS_THRESHOLD:
                    print("🔇 Audio too quiet — skipping")
                    continue

                print("🔍 Analyzing...")
                pred_class, confidence, probs = self.predict(audio)

                detections.append({
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'class':      pred_class,
                    'confidence': confidence,
                    'duration':   dur,
                })

                self.display(pred_class, confidence, probs, show_all)

        except KeyboardInterrupt:
            print("\n")

        # Session summary
        print("\n" + "=" * 70)
        print("SESSION SUMMARY")
        print("=" * 70)
        if detections:
            print(f"Total detections: {len(detections)}")
            from collections import Counter
            counts = Counter(d['class'] for d in detections)
            for cls, cnt in counts.most_common():
                avg_conf = np.mean([d['confidence'] for d in detections if d['class'] == cls])
                print(f"  {cls:20s}: {cnt}x  (avg {avg_conf * 100:.1f}%)")
        else:
            print("No detections.")
        print("=" * 70)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='YAMNet real-time sound detector')
    parser.add_argument('--model_dir', required=True,
                        help='Directory containing best_model.keras or final_model.keras')
    parser.add_argument('--data_dir',  required=True,
                        help='Directory containing label_mapping.json')
    parser.add_argument('--show_all',  action='store_true', default=True,
                        help='Show all class probabilities')
    parser.add_argument('--max_duration', type=float, default=30.0)
    args = parser.parse_args()

    detector = YAMNetDetector(args.model_dir, args.data_dir)
    detector.run_interactive(show_all=args.show_all, max_duration=args.max_duration)


if __name__ == '__main__':
    main()
