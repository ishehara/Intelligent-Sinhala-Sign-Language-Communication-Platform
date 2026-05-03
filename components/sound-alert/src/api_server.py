"""
Flask API Server — Sound Alert Component
=========================================
Accepts audio recorded by the phone microphone (via Expo Go/expo-av),
runs the trained CNN model, and returns the predicted sound class.

Flow
----
  Phone mic → expo-av records 2.5 s WAV/M4A
            → POST /predict (multipart form, field name: 'audio')
            → Flask decodes audio, runs CNN model
            → JSON { id, type, title, icon, severity, confidence, timestamp, detected }
            → SoundMonitoringScreen shows alert banner when detected == true

Endpoints
---------
  GET  /health   → liveness check (open in phone browser to verify connectivity)
  POST /predict  → accepts multipart audio file, returns prediction JSON

Run
---
    cd components/sound-alert
    python src/api_server.py

The server prints the Network URL on startup — copy it into the
SoundMonitoringScreen.tsx in your frontend repo (SOUND_ALERT_URL constant).

Notes
-----
* WAV (iOS) is decoded by soundfile — no extra tools needed.
* M4A/AAC (Android) needs ffmpeg in PATH for librosa/audioread to decode it.
  Install ffmpeg: https://ffmpeg.org/download.html  then add ffmpeg/bin to PATH.
"""

import os
import sys
import uuid
import tempfile
import traceback
import subprocess
from pathlib import Path
from datetime import datetime

# Ensure ffmpeg is in PATH for M4A/AAC decoding (audioread needs it)
# Search known locations in priority order
_FFMPEG_CANDIDATES = [
    r"C:\KMPlayer\ffmpeg.exe",
    r"C:\Program Files\GNU Octave\Octave-10.3.0\mingw64\bin\ffmpeg.exe",
    os.path.expandvars(
        r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"
        r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
        r"\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"
    ),
]
FFMPEG_EXE = None
for _candidate in _FFMPEG_CANDIDATES:
    if os.path.isfile(_candidate):
        FFMPEG_EXE = _candidate
        _ffmpeg_bin_dir = str(Path(_candidate).parent)
        if _ffmpeg_bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = os.environ["PATH"] + os.pathsep + _ffmpeg_bin_dir
        print(f"[INFO] ffmpeg found: {FFMPEG_EXE}")
        break
if FFMPEG_EXE is None:
    print("[WARN] ffmpeg not found — M4A files from Android may fail to decode")

import numpy as np
import librosa
import soundfile as sf
from flask import Flask, jsonify, request
from flask_cors import CORS

# ── Make realtime_detection importable ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from realtime_detection import RealtimeSoundDetector  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR  = Path(__file__).parent.parent
MODEL_PATH = _BASE_DIR / "models" / "alert_sounds_cnn" / "best_model.keras"
META_PATH  = _BASE_DIR / "data"   / "processed" / "alert_sounds" / "metadata.json"
LABEL_PATH = _BASE_DIR / "data"   / "processed" / "alert_sounds" / "label_mapping.json"

# ── Class map: backend label → (frontend AlertType, icon, severity) ──────────
CLASS_MAP: dict = {
    "bus horns":   ("bus-horn",        "🚌", "medium"),
    "car horns":   ("car-horn",        "🚗", "low"),
    "train horns": ("train-horn",      "🚂", "high"),
    "truck horns": ("truck-horn",      "🚛", "medium"),
    "ambulance":   ("ambulance-siren", "🚑", "high"),
    "firetruck":   ("fire-alarm",      "🚒", "high"),
    "police":      ("ambulance-siren", "🚔", "high"),
    "traffic":     ("loudspeaker",     "🚦", "low"),
}

# ── Lazy model loader ─────────────────────────────────────────────────────────
_detector: RealtimeSoundDetector | None = None


def get_detector() -> RealtimeSoundDetector:
    global _detector
    if _detector is None:
        print("[INFO] Loading Keras model — this takes a few seconds on first run…")
        _detector = RealtimeSoundDetector(
            model_path=str(MODEL_PATH),
            metadata_path=str(META_PATH),
            label_mapping_path=str(LABEL_PATH),
        )
        print("[INFO] Model ready.")
    return _detector


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # required so Expo Go (different origin) can reach the server


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Open http://<your-ip>:5003/health on the phone to confirm connectivity."""
    return jsonify({
        "status": "ok",
        "model_loaded": True
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a multipart/form-data request with an 'audio' file.
    Returns JSON:
      { id, type, title, icon, severity, confidence (0-100), timestamp, detected (bool) }
    'detected' is True when confidence >= 60 % — the frontend should only show
    an alert banner when detected == true.
    """
    if "audio" not in request.files:
        return jsonify({"error": "Missing 'audio' field in multipart form data"}), 400

    audio_file = request.files["audio"]
    filename   = audio_file.filename or "recording"
    suffix     = os.path.splitext(filename)[1] or ".m4a"

    tmp_path = None
    try:
        # Save to a temp file so librosa can open it by path
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            audio_file.save(tmp_path)

        det = get_detector()

        # ── Decode audio ──────────────────────────────────────────────────────
        # soundfile handles WAV/FLAC/OGG natively (no ffmpeg required).
        # M4A/AAC (Android) is converted to WAV via ffmpeg subprocess first.
        decode_path = tmp_path
        wav_tmp_path = None

        # If the file is M4A/AAC/MP4 and ffmpeg is available, convert it first
        if suffix.lower() in (".m4a", ".aac", ".mp4") and FFMPEG_EXE:
            try:
                wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                wav_tmp_path = wav_tmp.name
                wav_tmp.close()
                result = subprocess.run(
                    [FFMPEG_EXE, "-y", "-i", tmp_path, "-ar", str(det.sample_rate),
                     "-ac", "1", "-f", "wav", wav_tmp_path],
                    capture_output=True, timeout=30
                )
                if result.returncode == 0:
                    decode_path = wav_tmp_path
                    print(f"[INFO] M4A converted to WAV via ffmpeg")
                else:
                    print(f"[WARN] ffmpeg conversion failed: {result.stderr.decode()[:200]}")
            except Exception as e:
                print(f"[WARN] ffmpeg conversion error: {e}")

        try:
            audio, orig_sr = sf.read(decode_path, always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)          # stereo → mono
            audio = audio.astype("float32")
            if orig_sr != det.sample_rate:
                audio = librosa.resample(
                    audio, orig_sr=orig_sr, target_sr=det.sample_rate
                )
        except Exception:
            # Final fallback: librosa with audioread
            audio, _ = librosa.load(
                decode_path, sr=det.sample_rate, mono=True, duration=det.duration
            )

        # Pad/trim to exactly the duration the model was trained on
        target_len = int(det.duration * det.sample_rate)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]

        # ── Silence gate: skip very quiet recordings ──────────────────────────
        # Compute RMS energy to filter out silent/ambient background noise
        rms = float(np.sqrt(np.mean(audio.astype('float64')**2)))
        if rms < 0.008:
            # Too quiet — return an ambient detection (low confidence)
            print(
                f"[PREDICT] RMS energy too low ({rms:.6f} < 0.008) — skipping prediction"
            )
            return jsonify({
                "id":         str(uuid.uuid4()),
                "type":       "ambient",
                "title":      "Ambient Background Noise",
                "icon":       "🔇",
                "severity":   "low",
                "confidence": 0,
                "timestamp":  datetime.now().isoformat(),
                "detected":   False,
            })

        predicted_class, confidence, _ = det.predict_from_audio(audio)
        alert_type, icon, severity = CLASS_MAP.get(
            predicted_class,
            (predicted_class.replace(" ", "-"), "🔊", "medium"),
        )

        detected = float(confidence) >= 0.60
        print(
            f"[PREDICT] {predicted_class:<20s} {confidence*100:5.1f}%"
            f"  {'✓ ALERT' if detected else '○ skip'}"
        )

        return jsonify({
            "id":         str(uuid.uuid4()),
            "type":       alert_type,
            "title":      f"{predicted_class.title()} Detected",
            "icon":       icon,
            "severity":   severity,
            "confidence": round(float(confidence) * 100),
            "timestamp":  datetime.now().isoformat(),
            "detected":   detected,
        })

    except Exception as exc:
        traceback.print_exc()
        print(f"[ERROR] {exc!r}")
        return jsonify({"error": repr(exc)}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if wav_tmp_path and os.path.exists(wav_tmp_path):
            os.unlink(wav_tmp_path)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import socket

    try:
        lan_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        lan_ip = "127.0.0.1"

    print("=" * 60)
    print("  Sound Alert API Server")
    print("=" * 60)
    print(f"  Local:    http://127.0.0.1:5003")
    print(f"  Network:  http://{lan_ip}:5003")
    print()
    print("  ► Copy the Network URL into SoundMonitoringScreen.tsx in the frontend repo")
    print(f"  ► Open http://{lan_ip}:5003/health on your phone to confirm connectivity")
    print("  ► POST /predict  — phone uploads audio, server returns prediction")
    print("=" * 60)

    get_detector()   # pre-load model so first request is fast
    app.run(host="0.0.0.0", port=5003, debug=False)
