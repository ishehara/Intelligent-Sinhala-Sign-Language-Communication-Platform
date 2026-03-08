import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import base64
from PIL import Image, ImageOps
import io
import math
import mediapipe as mp
import time
import unicodedata
from collections import Counter, deque
from rl_feedback_agent import RLFeedbackAgent
from rl_enhanced_agent import AdaptiveLearningAgent
from rl_thompson_agent import ThompsonFeedbackAgent
from preprocessing_enhanced import EnhancedPreprocessor
from video_sign_predictor import VideoSignPredictor, VIDEO_SIGN_LABELS
from performance_history import (
    PerformanceHistory, suggest_next_in_level,
    get_level_progress, get_unlocked_levels, get_practice_suggestions,
)
try:
    from cvzone.HandTrackingModule import HandDetector
    USE_CVZONE = True
except ImportError:
    USE_CVZONE = False
    print("⚠️  cvzone not installed, falling back to raw MediaPipe")

app = Flask(__name__)
CORS(app)

# ── RL Feedback Agent (original) ──────────────────────────────
rl_agent = RLFeedbackAgent(
    save_path=os.path.join(os.path.dirname(__file__), 'rl_q_table.json'),
)

# ── Enhanced RL Agent (adaptive curriculum + spaced repetition) ─
enhanced_agent = AdaptiveLearningAgent(
    save_path=os.path.join(os.path.dirname(__file__), 'rl_enhanced_q_table.json'),
)

# ── Thompson Sampling Agent (contextual bandit) ───────────────
thompson_agent = ThompsonFeedbackAgent(
    save_path=os.path.join(os.path.dirname(__file__), 'rl_thompson_params.json'),
)

# ── Video Sign Predictor (dynamic/video signs) ───────────────
video_predictor = VideoSignPredictor()

# ── Performance History (attempt-level tracking) ─────────────
perf_history = PerformanceHistory(
    save_path=os.path.join(os.path.dirname(__file__), 'performance_history.json'),
)

# ── Enhanced Preprocessor ─────────────────────────────────────
enhanced_preprocessor = EnhancedPreprocessor(
    enable_clahe=True,
    enable_white_balance=True,
    enable_background_reduction=False,  # enable for cluttered backgrounds
    enable_landmarks=True,
)

# ── Debug folder for saving images ────────────────────────────
DEBUG_DIR = os.path.join(os.path.dirname(__file__), '..', 'debug_images')
os.makedirs(DEBUG_DIR, exist_ok=True)

# ── Load Models ───────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'sinhala_sign_language_classifier.keras')
model = load_model(MODEL_PATH)

# ── EfficientNet Combined Model (letter + correctness) ────────
EFFNET_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'Combine_model', 'sinhala_sign_efficientnet.h5')
EFFNET_METADATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'Combine_model', 'model_metadata_efficientnet.json')
try:
    effnet_model = load_model(EFFNET_MODEL_PATH)
    with open(EFFNET_METADATA_PATH, 'r', encoding='utf-8') as f:
        effnet_metadata = json.load(f)
    effnet_letter_labels = [effnet_metadata['class_labels']['letter'][str(i)] for i in range(25)]
    EFFNET_IMG_SIZE = tuple(effnet_metadata['img_size'])
    EFFNET_CORRECT_THRESHOLD = 0.5
    print(f"✅ EfficientNet model loaded: {len(effnet_letter_labels)} letters + correctness head")
except Exception as e:
    effnet_model = None
    effnet_letter_labels = []
    EFFNET_IMG_SIZE = (240, 240)
    EFFNET_CORRECT_THRESHOLD = 0.5
    print(f"⚠️  EfficientNet model not loaded: {e}")

# ── Class Labels ──────────────────────────────────────────────
# Order follows alphabetical sort of folder names (1,10,11,...,2,20,...,9)
# Verified against class_indices.json
class_labels = [
    unicodedata.normalize('NFC', c) for c in [
        'අ', 'ද', 'ත', 'ඩ', 'න', 'ප', 'බ', 'ම', 'ය', 'ර', 'ල',
        'ආ', 'ව', 'ස', 'හ', 'ං', 'ච', 'ෆ', 'ඇ', 'ඉ', 'උ', 'එ',
        'ක', 'ග', 'ට'
    ]
]

# ── Parameters (identical to prediction.py / dataset.py) ─────
TARGET_SIZE = (224, 224)
IMG_SIZE = 300        # white-canvas size
OFFSET = 20           # crop padding around hand

# ── Hand detector (match prediction.py exactly) ──────────────
if USE_CVZONE:
    # Same detector as prediction.py  ─ detectionCon=0.8
    cvzone_detector = HandDetector(maxHands=1, detectionCon=0.8)
    print("✅ Using cvzone HandDetector (same as prediction.py)")
else:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands_detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.8,
    )
    print("⚠️  cvzone not found — using MediaPipe Hands with landmark drawing")
    print("   For best results, install cvzone:  pip install cvzone==1.6.1")

# ── Prediction smoothing buffer ───────────────────────────
PREDICTION_BUFFER_SIZE = 5
prediction_buffer = []  # stores last N (label, confidence) tuples
_last_expected_label = None  # reset buffer when user switches sign

# ── Sign-specific correction tips (static signs) ─────────────
SIGN_CORRECTION_FEEDBACK = {
    'අ': 'The palm should be with fingers together, not separated.',
    'ආ': 'The thumb and other four fingers should be slightly closer.',
    'ඇ': 'The index finger should be slightly bent.',
    'ඉ': 'The palm should bend slightly like a cap shape.',
    'උ': 'The fingers and hand should form a 90-degree position.',
    'එ': 'Slightly rotate the wrist.',
    'ක': 'The thumb should be positioned between the index and middle finger.',
    'ග': 'The index finger should bend to approximately 90 degrees.',
    'ට': 'The index finger should bend more.',
    'ද': 'The hand should rotate to 90 degrees.',
    'ත': 'The hand should rotate to the opposite side.',
    'ඩ': 'The palm should rotate to the opposite side.',
    'න': 'The palm should rotate slightly to the other side.',
    'ප': 'The index finger should point downward, not upward.',
    'බ': 'The palm should rotate to the other side.',
    'ම': 'The palm should rotate to the other side.',
    'ය': 'The middle three fingers should bend more.',
    'ර': 'The index and middle fingers should stay together, not separated.',
    'ල': 'The thumb and middle finger should not bend.',
    'ව': 'The three fingers are positioned incorrectly.',
    'ස': 'The palm should rotate to the other side.',
    'හ': 'The hand position should rotate upside down.',
    'ං': 'All five fingers should bend together, not just two.',
    'ච': 'This is not a thumbs-up position.',
    'ෆ': 'Three fingers should bend slightly.',
}


# ── Helper: detect hand bbox via MediaPipe/cvzone ────────────
def detect_hand(img_bgr):
    """Return (bbox, drawn_img) or (None, img_bgr).
    prediction.py uses findHands(img) with draw=True (default) and then crops
    from the DRAWN image.  We replicate that exactly here."""
    if USE_CVZONE:
        # Match prediction.py exactly: hands, img = detector.findHands(img)
        # draw=True is the default — landmarks are drawn ON img_bgr
        hands, img_drawn = cvzone_detector.findHands(img_bgr)
        if hands:
            return tuple(hands[0]['bbox']), img_drawn
        return None, img_bgr
    else:
        h_img, w_img = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(img_rgb)

        if not results.multi_hand_landmarks:
            return None, img_bgr

        lm = results.multi_hand_landmarks[0]
        xs = [p.x for p in lm.landmark]
        ys = [p.y for p in lm.landmark]

        x_min = max(0, int(min(xs) * w_img))
        y_min = max(0, int(min(ys) * h_img))
        x_max = min(w_img, int(max(xs) * w_img))
        y_max = min(h_img, int(max(ys) * h_img))

        # ═══ CRITICAL: Draw landmarks on the image ═══
        # The model was trained on images WITH hand landmarks drawn.
        # Without landmarks, the model cannot predict correctly.
        mp_drawing.draw_landmarks(
            img_bgr,
            lm,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

        return (x_min, y_min, x_max - x_min, y_max - y_min), img_bgr


# ── Helper: crop hand & place on white canvas ────────────────
def crop_and_canvas(img_bgr, x, y, w, h):
    """
    Crop hand region and place on 300x300 white canvas.
    Exactly matches prediction.py logic.
    """
    y1 = max(0, y - OFFSET)
    y2 = min(img_bgr.shape[0], y + h + OFFSET)
    x1 = max(0, x - OFFSET)
    x2 = min(img_bgr.shape[1], x + w + OFFSET)
    crop = img_bgr[y1:y2, x1:x2]

    h_c, w_c = crop.shape[:2]
    if h_c == 0 or w_c == 0:
        return None

    # White canvas + aspect-ratio resize (same as prediction.py)------
    imgWhite = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
    aspectRatio = h_c / w_c

    if aspectRatio > 1:
        k = IMG_SIZE / h_c
        wCal = math.ceil(k * w_c)
        if wCal > 0:
            imageResize = cv2.resize(crop, (wCal, IMG_SIZE))
            wGap = math.ceil((IMG_SIZE - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imageResize
    else:
        k = IMG_SIZE / w_c
        hCal = math.ceil(k * h_c)
        if hCal > 0:
            imageResize = cv2.resize(crop, (IMG_SIZE, hCal))
            hGap = math.ceil((IMG_SIZE - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imageResize

    return imgWhite


# ── Helper: normalise canvas for EfficientNet correctness head ──
def normalize_canvas_for_correctness(canvas_bgr):
    """
    Apply CLAHE + gamma correction + unsharp mask so the EfficientNet
    correctness head sees a consistent image regardless of mobile
    lighting/exposure differences.
    """
    lab = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Gamma correction (brighten slightly)
    gamma = 1.5
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                     for i in range(256)], dtype=np.uint8)
    img = cv2.LUT(img, lut)

    # Unsharp mask
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return img


# ── Helper: preprocess exactly like prediction.py ─────────────
def preprocess_for_model(pil_image, is_front_camera=False, save_debug=True):
    """
    Reproduce prediction.py EXACTLY:
      1. Letterbox-scale to 640×480 (preserve aspect ratio, pad with black)
      2. If front camera → flip horizontally (undo mirror)
      3. cvzone findHands with draw=True (same as prediction.py)
      4. Crop from DRAWN image + offset → 300×300 white canvas
      5. Resize to 224×224, normalise to [0,1]
    Returns (preprocessed_batch, hand_detected_bool, debug_info, hand_canvas_or_None)
    """
    img = np.array(pil_image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    debug_info = {
        'original_shape': img_bgr.shape,
        'is_front_camera': is_front_camera,
    }

    # ═══ FIX: Letterbox to 640×480 (prediction.py webcam resolution) ═══
    h_orig, w_orig = img_bgr.shape[:2]
    target_w, target_h = 640, 480
    scale = min(target_w / w_orig, target_h / h_orig)
    new_w = int(w_orig * scale)
    new_h = int(h_orig * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h))
    # Centre on a black canvas
    letterboxed = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    letterboxed[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    img_bgr = letterboxed
    debug_info['scaled_to'] = f'{new_w}x{new_h} -> 640x480 letterbox'

    # If front camera, flip once to undo mirror
    if is_front_camera:
        img_bgr = cv2.flip(img_bgr, 1)
        debug_info['flipped_front_camera'] = True

    # Save received image for debugging
    if save_debug:
        ts = int(time.time() * 1000)
        cv2.imwrite(os.path.join(DEBUG_DIR, f'{ts}_1_received.jpg'), img_bgr)

    # ═══ Detect hand — match prediction.py: draw=True, crop from drawn img ═══
    bbox, img_drawn = detect_hand(img_bgr)
    
    if bbox is None:
        print("⚠️  No hand detected")
        if save_debug:
            cv2.imwrite(os.path.join(DEBUG_DIR, f'{ts}_2_NO_HAND.jpg'), img_bgr)
        # Fallback: send full image (low quality)
        img_resized = cv2.resize(img_bgr, TARGET_SIZE)
        return np.expand_dims(img_resized / 255.0, axis=0), False, debug_info, None

    x, y, w, h = bbox
    print(f"✅ Hand bbox: x={x} y={y} w={w} h={h}")
    debug_info['bbox'] = bbox
    
    # Crop from the DRAWN image (prediction.py does: imageCrop = img[y1:y2,x1:x2]
    # where img already has landmarks drawn by findHands)
    canvas = crop_and_canvas(img_drawn, x, y, w, h)
    
    if canvas is None:
        print("⚠️  Canvas creation failed")
        img_resized = cv2.resize(img_bgr, TARGET_SIZE)
        return np.expand_dims(img_resized / 255.0, axis=0), False, debug_info, None

    if save_debug:
        cv2.imwrite(os.path.join(DEBUG_DIR, f'{ts}_2_canvas.jpg'), canvas)

    # Normalised canvas for EfficientNet correctness head
    hand_canvas = normalize_canvas_for_correctness(canvas.copy())

    final = cv2.resize(canvas, TARGET_SIZE) / 255.0
    
    if save_debug:
        cv2.imwrite(os.path.join(DEBUG_DIR, f'{ts}_3_model_input.jpg'), 
                     (final * 255).astype(np.uint8))
    
    return np.expand_dims(final, axis=0), True, debug_info, hand_canvas


# ── Helper: feedback (now powered by RL agent) ───────────────
# The old hardcoded generate_feedback is replaced by rl_agent.get_feedback().
# The RL agent learns which feedback messages help users improve the most.


# ══════════════════════════════════════════════════════════════
# ── ROUTES ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "service": "Sign Language Learning Backend",
        "status": "running",
        "endpoints": {
            "GET /health": "Health check",
            "GET /labels": "Get all sign labels",
            "GET /all-labels": "Get static + dynamic labels",
            "POST /predict-sign": "Predict sign from image",
            "POST /predict-video-sign": "Predict dynamic sign from video frame",
            "POST /performance-history": "Get attempt history",
            "POST /level-summary": "Get level summary stats",
            "POST /suggest-next-letter": "Get next letter suggestion",
            "POST /all-level-summaries": "Get all level summaries",
            "POST /level-progress": "Get detailed level progress",
            "POST /unlocked-levels": "Get unlocked levels",
            "POST /practice-suggestions": "Get signs needing re-practice",
            "GET /thompson-stats": "Thompson sampling stats",
        },
    })


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "effnet_model_loaded": effnet_model is not None,
        "num_classes": len(class_labels),
        "num_video_classes": len(VIDEO_SIGN_LABELS),
        "detector": "cvzone" if USE_CVZONE else "mediapipe_fallback",
        "landmarks_drawn": True,
        "video_model_loaded": video_predictor.model is not None,
    })


@app.route('/predict-sign', methods=['POST'])
def predict_sign():
    """
    Receive base64 image -> detect hand -> preprocess -> predict -> JSON.
    Request:  { "image": "<base64>", "expected_label": "අ" (optional) }
    """
    global _last_expected_label
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode base64
        raw = data['image']
        if ',' in raw:
            raw = raw.split(',')[1]

        image = Image.open(io.BytesIO(base64.b64decode(raw)))
        image = ImageOps.exif_transpose(image)  # Fix rotation from mobile camera
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        print(f"📷 Received image: {image.size} mode={image.mode}")

        # Check if front camera was used (for mirror flip)
        is_front_camera = data.get('is_front_camera', False)

        # Preprocess (same pipeline as prediction.py)
        batch, hand_detected, debug_info, hand_canvas = preprocess_for_model(
            image, is_front_camera=is_front_camera
        )
        
        print(f"📊 Debug: original={debug_info['original_shape']} "
              f"resized=640x480 front_cam={is_front_camera}")

        # Predict
        preds = model.predict(batch, verbose=0)
        raw_idx = int(np.argmax(preds))
        raw_conf = float(np.max(preds))
        raw_label = class_labels[raw_idx]
        
        # Get top 3 predictions for debugging
        top3_indices = np.argsort(preds[0])[-3:][::-1]
        top3 = [(class_labels[i], float(preds[0][i])) for i in top3_indices]
        print(f"🎯 Raw: {raw_label} ({raw_conf*100:.1f}%)  Top3: {[(l, f'{c*100:.1f}%') for l, c in top3]}")

        # ── Normalise expected_label (Unicode NFC) ──
        expected = data.get('expected_label', None)
        if expected:
            expected = unicodedata.normalize('NFC', expected)

        # ── Reset prediction buffer when the user switches to a new sign ──
        if expected != _last_expected_label:
            prediction_buffer.clear()
            _last_expected_label = expected

        # ── Prediction smoothing (majority vote) ──
        prediction_buffer.append((raw_label, raw_conf))
        if len(prediction_buffer) > PREDICTION_BUFFER_SIZE:
            prediction_buffer.pop(0)

        if hand_detected and len(prediction_buffer) >= 2:
            labels_in_buffer = [lb for lb, _ in prediction_buffer]
            label_counts = Counter(labels_in_buffer)
            majority_label, majority_count = label_counts.most_common(1)[0]

            if majority_count >= max(2, len(prediction_buffer) * 0.4):
                label = majority_label
                majority_confs = [c for lb, c in prediction_buffer if lb == majority_label]
                conf = float(np.mean(majority_confs))
            else:
                label = raw_label
                conf = raw_conf
        else:
            label = raw_label
            conf = raw_conf
        
        print(f"📊 Smoothed: {label} ({conf*100:.1f}%) buffer={[lb for lb,_ in prediction_buffer]}")

        # ── EfficientNet dual-head (letter + correctness) ──
        effnet_correctness = None
        effnet_label = None
        if effnet_model is not None and hand_canvas is not None:
            try:
                eff_input = cv2.resize(hand_canvas, TARGET_SIZE) / 255.0
                eff_batch = np.expand_dims(eff_input, axis=0)
                eff_preds = effnet_model.predict(eff_batch, verbose=0)
                if isinstance(eff_preds, list) and len(eff_preds) == 2:
                    letter_preds, corr_preds = eff_preds
                    effnet_idx = int(np.argmax(letter_preds))
                    effnet_label = class_labels[effnet_idx] if effnet_idx < len(class_labels) else None
                    effnet_correctness = float(corr_preds[0][0])
                    print(f"🔬 EfficientNet: letter={effnet_label} correctness={effnet_correctness:.2f}")
                else:
                    letter_preds = eff_preds
                    effnet_idx = int(np.argmax(letter_preds))
                    effnet_label = class_labels[effnet_idx] if effnet_idx < len(class_labels) else None
            except Exception as eff_err:
                print(f"⚠️ EfficientNet error: {eff_err}")

        attempt_count = data.get('attempt_count', 1)

        # ── Confidence-based status (restored thresholds) ──
        correction_tip = ""
        if expected:
            if label == expected and conf >= 0.55:
                status = "correct"
                is_correct = True
            elif conf < 0.45:
                status = "incorrect"
                is_correct = False
                correction_tip = SIGN_CORRECTION_FEEDBACK.get(expected, "")
            else:
                # 45% – 55% confidence zone
                status = "try_again"
                is_correct = False
        else:
            status = "predicted"
            is_correct = True

        # ── RL Agent selects feedback ──
        rl_result = rl_agent.get_feedback(
            confidence=conf,
            is_correct=is_correct,
            hand_detected=hand_detected,
            attempt_count=attempt_count,
            predicted_label=label,
            expected_label=expected,
        )
        feedback = rl_result["feedback"]
        feedback_level = rl_result["feedback_level"]

        # ── Override feedback based on confidence thresholds ──
        if expected:
            if status == "correct":
                feedback = "Correct Sign"
                feedback_level = "excellent"
            elif status == "incorrect":
                feedback = "Incorrect Sign"
                feedback_level = "poor"
            elif status == "try_again":
                feedback = "Try Again \u2013 Adjust Your Hand Position"
                feedback_level = "fair"

        # ── Thompson sampling feedback (learn which message helps most) ──
        thompson_fb = None
        if thompson_agent and expected:
            try:
                thompson_fb = thompson_agent.get_feedback(
                    confidence=conf,
                    is_correct=is_correct,
                    hand_detected=hand_detected,
                    attempt_count=attempt_count,
                    predicted_label=label,
                    expected_label=expected,
                )
                if thompson_fb:
                    feedback = thompson_fb.get("feedback", feedback)
            except Exception:
                pass

        # ── Record attempt in performance history ──
        attempt_level = ""
        attempt_recorded = False
        if expected and perf_history:
            try:
                user_id = data.get('user_id', 'default_user')
                state = enhanced_agent.get_user_sign_state(user_id, expected)
                rec = perf_history.add_attempt(
                    user_id=user_id,
                    sign=expected,
                    predicted_label=label,
                    expected_label=expected,
                    is_correct=is_correct,
                    confidence=conf,
                    response_time_ms=data.get('response_time_ms', 0),
                    feedback=feedback,
                    mastery_level=state.mastery_level if state else 0,
                    sign_type="static",
                )
                attempt_level = rec.get("level", "")
                attempt_recorded = True
            except Exception as hist_err:
                print(f"⚠️ History recording error: {hist_err}")

        resp = {
            "predicted_label": label,
            "confidence": round(conf, 4),
            "status": status,
            "feedback": feedback,
            "feedback_level": feedback_level,
            "tip": rl_result.get("tip", ""),
            "correction_tip": correction_tip,
            "session_id": rl_result.get("session_id", ""),
            "hand_detected": hand_detected,
            "used_flip": debug_info.get('used_flip', False),
            "attempt_recorded": attempt_recorded,
            "attempt_level": attempt_level,
            "all_predictions": {
                class_labels[i]: round(float(preds[0][i]), 4)
                for i in range(len(class_labels))
            },
        }
        if effnet_correctness is not None:
            resp["effnet_correctness"] = round(effnet_correctness, 4)
        if effnet_label is not None:
            resp["effnet_label"] = effnet_label
        if not hand_detected:
            resp["warning"] = (
                "No hand detected. Make sure your hand is clearly visible "
                "and well-lit inside the frame."
            )

        return jsonify(resp)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "message": "Failed to process image"}), 500


@app.route('/debug-images', methods=['GET'])
def list_debug_images():
    """List recent debug images saved by the server."""
    files = sorted(os.listdir(DEBUG_DIR), reverse=True)[:30]
    return jsonify({"debug_dir": DEBUG_DIR, "files": files})


@app.route('/debug-images/<filename>', methods=['GET'])
def get_debug_image(filename):
    """Serve a debug image."""
    return send_from_directory(DEBUG_DIR, filename)


@app.route('/labels', methods=['GET'])
def get_labels():
    return jsonify({"labels": class_labels, "total": len(class_labels)})


# ── RL Reward Endpoint ────────────────────────────────────────
@app.route('/rl-reward', methods=['POST'])
def rl_reward():
    """
    Receive reward signal from the frontend.
    Request: {
        "session_id": "s_123...",
        "reward_type": "improved" | "correct" | "retry" | "give_up" | "no_change",
        "new_confidence": 0.85,      (optional)
        "new_is_correct": true        (optional)
    }
    """
    try:
        data = request.get_json()
        if not data or 'session_id' not in data:
            return jsonify({"error": "session_id required"}), 400

        result = rl_agent.receive_reward(
            session_id=data['session_id'],
            reward_type=data.get('reward_type', 'no_change'),
            new_confidence=data.get('new_confidence'),
            new_is_correct=data.get('new_is_correct'),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── RL Stats Endpoint ─────────────────────────────────────────
@app.route('/rl-stats', methods=['GET'])
def rl_stats():
    """Return RL agent learning statistics."""
    return jsonify(rl_agent.get_stats())


# ══════════════════════════════════════════════════════════════
# ── Enhanced RL Endpoints (Adaptive Curriculum) ───────────────
# ══════════════════════════════════════════════════════════════

@app.route('/next-sign', methods=['POST'])
def next_sign():
    """
    Recommend next sign to practice using RL + spaced repetition.
    Request: { "user_id": "user123" }
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default_user')
        result = enhanced_agent.recommend_next_sign(user_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/adaptive-feedback', methods=['POST'])
def adaptive_feedback():
    """
    Get RL-adaptive feedback with curriculum and spaced repetition.
    Request: {
        "user_id": "user123",
        "sign": "අ",
        "predicted_label": "අ",
        "confidence": 0.85,
        "is_correct": true,
        "response_time_ms": 2000
    }
    """
    try:
        data = request.get_json()
        result = enhanced_agent.get_adaptive_feedback(
            user_id=data.get('user_id', 'default_user'),
            sign=data.get('sign', ''),
            predicted_label=data.get('predicted_label', ''),
            confidence=data.get('confidence', 0.0),
            is_correct=data.get('is_correct', False),
            response_time_ms=data.get('response_time_ms', 3000),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/user-progress', methods=['POST'])
def user_progress():
    """
    Get user learning progress summary.
    Request: { "user_id": "user123" }
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default_user')
        progress = enhanced_agent._get_user_progress(user_id)
        return jsonify(progress)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/enhanced-rl-stats', methods=['GET'])
def enhanced_rl_stats():
    """Return enhanced RL agent statistics."""
    return jsonify(enhanced_agent.get_stats())


@app.route('/sign-details', methods=['POST'])
def sign_details():
    """
    Return per-sign learning stats for a user.
    Request: { "user_id": "user123" }
    Response: { "signs": [ { sign, attempts, accuracy, confidence, mastery_level, streak, ... } ] }
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default_user')
        user_signs = enhanced_agent.user_states.get(user_id, {})

        signs_list = []
        for sign in class_labels:
            state = enhanced_agent.get_user_sign_state(user_id, sign)
            from rl_enhanced_agent import SIGN_TO_DIFFICULTY
            diff = SIGN_TO_DIFFICULTY.get(sign, 1)
            signs_list.append({
                "sign": sign,
                "difficulty": diff,
                "total_attempts": state.total_attempts,
                "correct_attempts": state.correct_attempts,
                "accuracy": round(state.accuracy * 100, 1),
                "avg_confidence": round(state.avg_confidence * 100, 1),
                "best_confidence": round(state.best_confidence * 100, 1),
                "current_streak": state.current_streak,
                "longest_streak": state.longest_streak,
                "mastery_level": state.mastery_level,
                "ease_factor": round(state.ease_factor, 2),
                "review_interval_days": round(state.review_interval_days, 1),
                "is_due_for_review": state.is_due_for_review,
                "days_since_practice": round(state.days_since_practice, 1) if state.last_practiced else None,
                "last_practiced": state.last_practiced,
            })

        return jsonify({
            "user_id": user_id,
            "signs": signs_list,
            "total_signs": len(class_labels),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/reset-progress', methods=['POST'])
def reset_progress():
    """
    Reset user learning progress and optionally the RL agent.
    Request: { "user_id": "user123", "reset_rl": false }
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default_user')
        reset_rl = data.get('reset_rl', False)

        if user_id in enhanced_agent.user_states:
            del enhanced_agent.user_states[user_id]
            enhanced_agent._save()

        if reset_rl:
            enhanced_agent.q_table.clear()
            enhanced_agent.epsilon = 0.25
            enhanced_agent.total_episodes = 0
            enhanced_agent.total_rewards = 0.0
            enhanced_agent.reward_history = []
            enhanced_agent._save()

        return jsonify({"status": "ok", "user_id": user_id, "rl_reset": reset_rl})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════
# ── Video Sign Endpoints (Dynamic Signs) ──────────────────────
# ══════════════════════════════════════════════════════════════

@app.route('/predict-video-sign', methods=['POST'])
def predict_video_sign():
    """
    Receive a single base64 frame from the mobile camera.
    Extracts hand landmarks, buffers them per session, and predicts
    the dynamic sign once enough frames are collected.

    Request: {
        "image": "<base64>",
        "session_id": "vid_abc123",
        "expected_label": "ඈ"  (optional),
        "is_front_camera": false
    }
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        raw = data['image']
        if ',' in raw:
            raw = raw.split(',')[1]

        image = Image.open(io.BytesIO(base64.b64decode(raw)))
        image = ImageOps.exif_transpose(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Scale down large images
        h_orig, w_orig = img_bgr.shape[:2]
        max_dim = 640
        if max(h_orig, w_orig) > max_dim:
            scale = max_dim / max(h_orig, w_orig)
            img_bgr = cv2.resize(img_bgr, (int(w_orig * scale), int(h_orig * scale)))

        # Flip front camera
        if data.get('is_front_camera', False):
            img_bgr = cv2.flip(img_bgr, 1)

        session_id = data.get('session_id', 'default_video')
        result = video_predictor.process_frame(img_bgr, session_id=session_id)

        # Build response
        expected = data.get('expected_label', None)
        if expected:
            expected = unicodedata.normalize('NFC', expected)
        predicted = result.get('prediction')

        status = result['status']
        feedback = ''
        feedback_level = 'fair'
        correction_tip = ''

        if predicted and expected:
            vid_conf = result['confidence']
            if predicted == expected and vid_conf >= 0.55:
                status = 'correct'
                feedback = 'Correct Sign'
                feedback_level = 'excellent'
            elif vid_conf < 0.45:
                status = 'incorrect'
                feedback = 'Incorrect Sign'
                feedback_level = 'poor'
                correction_tip = SIGN_CORRECTION_FEEDBACK.get(expected, '')
            else:
                status = 'try_again'
                feedback = 'Try Again \u2013 Adjust Your Hand Position'
                feedback_level = 'fair'
        elif predicted:
            status = 'predicted'
            feedback = f'Detected dynamic sign: {predicted}'
            feedback_level = 'good'
        elif result['status'] == 'buffering':
            feedback = 'Keep moving your hand... collecting frames.'
        elif result['status'] == 'low_confidence':
            feedback = 'Motion not clear enough. Try the sign again more clearly.'

        # ── Record attempt in performance history (only when prediction is final) ──
        attempt_recorded = False
        attempt_level = ""
        if predicted and expected and perf_history:
            try:
                user_id = data.get('user_id', 'default_user')
                is_correct = (predicted == expected)
                state = enhanced_agent.get_user_sign_state(user_id, expected)
                rec = perf_history.add_attempt(
                    user_id=user_id,
                    sign=expected,
                    predicted_label=predicted,
                    expected_label=expected,
                    is_correct=is_correct,
                    confidence=result['confidence'],
                    response_time_ms=data.get('response_time_ms', 0),
                    feedback=feedback,
                    mastery_level=state.mastery_level if state else 0,
                    sign_type="dynamic",
                )
                attempt_level = rec.get("level", "")
                attempt_recorded = True
            except Exception as hist_err:
                print(f"⚠️ Video history recording error: {hist_err}")

        resp = {
            "predicted_label": predicted,
            "confidence": round(result['confidence'], 4),
            "status": status,
            "feedback": feedback,
            "feedback_level": feedback_level,
            "correction_tip": correction_tip,
            "hand_detected": result['hand_detected'],
            "buffer_size": result['buffer_size'],
            "buffer_max": 30,
            "sign_type": "dynamic",
            "attempt_recorded": attempt_recorded,
            "attempt_level": attempt_level,
        }
        if not result['hand_detected']:
            resp["warning"] = (
                "No hand detected. Make sure your hand is clearly visible "
                "and well-lit inside the frame."
            )

        return jsonify(resp)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "message": "Failed to process video frame"}), 500


@app.route('/video-sign-labels', methods=['GET'])
def video_sign_labels():
    """Return all dynamic sign labels."""
    return jsonify({"labels": VIDEO_SIGN_LABELS, "total": len(VIDEO_SIGN_LABELS)})


@app.route('/clear-video-buffer', methods=['POST'])
def clear_video_buffer():
    """Clear the frame buffer for a video session."""
    data = request.get_json() or {}
    session_id = data.get('session_id', 'default_video')
    video_predictor.clear_buffer(session_id)
    return jsonify({"status": "buffer_cleared", "session_id": session_id})


@app.route('/all-labels', methods=['GET'])
def all_labels():
    """
    Return both static and dynamic sign labels.
    Used by the frontend to show what signs are available.
    """
    return jsonify({
        "static": {"labels": class_labels, "total": len(class_labels)},
        "dynamic": {"labels": VIDEO_SIGN_LABELS, "total": len(VIDEO_SIGN_LABELS)},
    })


# ══════════════════════════════════════════════════════════════
# ── Performance History & Suggestion Endpoints ────────────────
# ══════════════════════════════════════════════════════════════

@app.route('/performance-history', methods=['POST'])
def performance_history_endpoint():
    """
    Return paginated attempt history for a user.
    Request: { "user_id": "default_user", "level": "beginner", "sign": "අ",
               "sign_type": "static", "limit": 50, "offset": 0 }
    """
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id', 'default_user')
        result = perf_history.get_history(
            user_id=user_id,
            level=data.get('level'),
            sign=data.get('sign'),
            sign_type=data.get('sign_type'),
            limit=data.get('limit', 50),
            offset=data.get('offset', 0),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/level-summary', methods=['POST'])
def level_summary():
    """
    Aggregate stats per sign within a level.
    Request: { "user_id": "default_user", "level": "beginner" }
    """
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id', 'default_user')
        level = data.get('level', 'beginner')
        user_sign_states = enhanced_agent.user_states.get(user_id, {})
        result = perf_history.get_level_summary(user_id, level, user_sign_states)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/suggest-next-letter', methods=['POST'])
def suggest_next_letter():
    """
    Within a specific level, recommend which letter to practise next.
    Request: { "user_id": "default_user", "level": "beginner" }
    """
    try:
        from performance_history import suggest_next_in_level
        data = request.get_json() or {}
        user_id = data.get('user_id', 'default_user')
        level = data.get('level', 'beginner')
        user_sign_states = enhanced_agent.user_states.get(user_id, {})
        result = suggest_next_in_level(user_id, level, user_sign_states, perf_history)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/all-level-summaries', methods=['POST'])
def all_level_summaries():
    """
    Return summaries for all curriculum levels (static + dynamic).
    Request: { "user_id": "default_user" }
    """
    try:
        from rl_enhanced_agent import SIGN_CURRICULUM as CURR
        data = request.get_json() or {}
        user_id = data.get('user_id', 'default_user')
        user_sign_states = enhanced_agent.user_states.get(user_id, {})

        summaries = {}
        for level_name in CURR:
            summaries[level_name] = perf_history.get_level_summary(
                user_id, level_name, user_sign_states
            )
        return jsonify({"user_id": user_id, "levels": summaries})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/thompson-stats', methods=['GET'])
def thompson_stats():
    """Return Thompson sampling agent statistics."""
    if thompson_agent:
        return jsonify(thompson_agent.get_stats())
    return jsonify({"error": "Thompson agent not loaded"}), 404


@app.route('/level-progress', methods=['POST'])
def level_progress():
    """
    Detailed level progress: per-sign completion status + confidence category.
    Request: { "user_id": "default_user", "level": "beginner" }
    """
    try:
        from performance_history import get_level_progress
        data = request.get_json() or {}
        user_id = data.get('user_id', 'default_user')
        level = data.get('level', 'beginner')
        user_sign_states = enhanced_agent.user_states.get(user_id, {})
        result = get_level_progress(user_id, level, user_sign_states, perf_history)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/unlocked-levels', methods=['POST'])
def unlocked_levels():
    """
    Return list of unlocked level names for a user.
    Request: { "user_id": "default_user" }
    """
    try:
        from performance_history import get_unlocked_levels
        data = request.get_json() or {}
        user_id = data.get('user_id', 'default_user')
        user_sign_states = enhanced_agent.user_states.get(user_id, {})
        levels = get_unlocked_levels(user_id, user_sign_states, perf_history)
        return jsonify({"user_id": user_id, "unlocked_levels": levels})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/practice-suggestions', methods=['POST'])
def practice_suggestions():
    """
    Return signs that need re-practice (poor confidence).
    Request: { "user_id": "default_user" }
    """
    try:
        from performance_history import get_practice_suggestions
        data = request.get_json() or {}
        user_id = data.get('user_id', 'default_user')
        user_sign_states = enhanced_agent.user_states.get(user_id, {})
        suggestions = get_practice_suggestions(user_id, user_sign_states, perf_history)
        return jsonify({"user_id": user_id, "suggestions": suggestions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("🚀 Starting Sign Language Learning Backend Server...")
    print(f"📊 Static model loaded with {len(class_labels)} classes")
    print(f"📹 Video model loaded with {len(VIDEO_SIGN_LABELS)} dynamic sign classes")
    print(f"🎯 Available static signs: {', '.join(class_labels[:5])}...")
    print(f"🎯 Available dynamic signs: {', '.join(VIDEO_SIGN_LABELS[:5])}...")
    print(f"🔍 Hand detector: {'cvzone (RECOMMENDED)' if USE_CVZONE else 'MediaPipe FALLBACK'}")
    print(f"✏️  Landmark drawing: ENABLED (required for correct predictions)")
    if not USE_CVZONE:
        print("⚠️  WARNING: cvzone not installed! Predictions may differ from prediction.py.")
        print("   Install with: pip install cvzone==1.6.1")
    app.run(host='0.0.0.0', port=5000, debug=False)
