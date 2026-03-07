import os
import logging
from pathlib import Path

import numpy as np
import joblib
from flask import Flask, request, jsonify
from ai_suggestion import get_suggestions
from translate import translate_response_values_to_sinhala

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MODEL_TAG  = os.environ.get("MODEL_TAG", "xgb")
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "saved_models"))

FEATURES = [
    "age_group", "app_usage_bucket",
    "q1_familiar_instruction_response", "q2_new_instruction_response",
    "q3_response_speed_visual", "q4_basic_needs_expression",
    "q5_initiates_communication", "q6_multi_sign_combination",
    "q7_focus_duration", "q8_task_completion", "q9_reminder_dependency",
    "q10_sign_retention_next_day", "q11_repetition_needed",
    "q12_real_life_application", "q13_routine_understanding",
    "q14_routine_response_accuracy", "q15_independence_attempt",
    "q16_confidence_in_sign_usage",
]

TARGETS = [
    "communication_level",
    "instruction_understanding_level",
    "learning_retention_level",
    "attention_engagement_level",
    "independence_confidence_level",
]

TARGET_LABELS = {
    "communication_level":             {0: "Emerging",        1: "Developing",    2: "Functional",       3: "Strong"},
    "instruction_understanding_level": {0: "Limited",         1: "Improving",     2: "Consistent",       3: "Advanced"},
    "learning_retention_level":        {0: "Low",             1: "Moderate",      2: "Good",             3: "Strong"},
    "attention_engagement_level":      {0: "Very Short",      1: "Short",         2: "Stable",           3: "Sustained"},
    "independence_confidence_level":   {0: "Dependent",       1: "Assisted",      2: "Semi-Independent", 3: "Independent"},
}

_cache = {}


def _load_bundle(target):
    key = f"{MODEL_TAG}_{target}"
    if key in _cache:
        return _cache[key]
    path = MODELS_DIR / f"{MODEL_TAG}_{target}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path} — run the notebook first (SAVE_MODELS=True)")
    _cache[key] = joblib.load(path)
    log.info(f"Loaded {path}")
    return _cache[key]


def _load_nn_bundle(target):
    key = f"nn_{target}"
    if key in _cache:
        return _cache[key]
    import torch
    import torch.nn as nn

    meta_path  = MODELS_DIR / f"nn_{target}_meta.pkl"
    state_path = MODELS_DIR / f"nn_{target}.pt"
    if not meta_path.exists() or not state_path.exists():
        raise FileNotFoundError(f"NN model files missing for '{target}'")

    meta = joblib.load(meta_path)

    class BehavioralMLP(nn.Module):
        def __init__(self, in_dim, out_dim, hidden=(128, 64, 32)):
            super().__init__()
            layers, prev = [], in_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
                prev = h
            layers.append(nn.Linear(prev, out_dim))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

    model = BehavioralMLP(meta["in_dim"], meta["n_classes"])
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()
    meta["_model"] = model
    _cache[key] = meta
    log.info(f"Loaded {state_path} + {meta_path}")
    return meta


def _predict_target(target, vec):
    label_map = TARGET_LABELS[target]

    if MODEL_TAG == "nn":
        import torch
        bundle   = _load_nn_bundle(target)
        x_scaled = torch.tensor(bundle["scaler"].transform(vec), dtype=torch.float32)
        with torch.no_grad():
            probs = torch.softmax(bundle["_model"](x_scaled), dim=1).numpy()[0]
        pred_code  = int(probs.argmax())
        confidence = float(probs.max())
    else:
        bundle     = _load_bundle(target)
        pred_code  = int(bundle["model"].predict(vec)[0])
        probs      = bundle["model"].predict_proba(vec)[0]
        confidence = float(probs.max())

    pred_label = label_map[pred_code]
    return {
        "prediction":    pred_label,
        "code":          pred_code,
        "confidence":    round(confidence, 4),
        "probabilities": {label_map[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }


app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

@app.route("/child-predict", methods=["POST"])
def predict():
    data = request.get_json(force=True) or {}

    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": "Missing required fields", "missing": missing}), 400

    invalid = {f: data[f] for f in FEATURES if not (0 <= int(data[f]) <= 3)}
    if invalid:
        return jsonify({"error": "Field values must be between 0 and 3", "invalid": invalid}), 400

    vec = np.array([[float(data[f]) for f in FEATURES]], dtype=np.float32)

    predictions, errors = {}, {}
    for target in TARGETS:
        try:
            predictions[target] = _predict_target(target, vec)
        except Exception as e:
            log.error(f"{target}: {e}")
            errors[target] = str(e)

    strengths, needs_support = [], []
    for target, result in predictions.items():
        nice = target.replace("_level", "").replace("_", " ").title()
        if result["code"] >= 2:
            strengths.append(f"{nice}: {result['prediction']}")
        else:
            needs_support.append(f"{nice}: {result['prediction']}")

    child_meta = {
        "age_group":        int(data.get("age_group", 0)),
        "app_usage_bucket": int(data.get("app_usage_bucket", 0)),
    }
    ai_suggestions = get_suggestions(predictions, child_meta)
    # ai_suggestions = "Mother needs to work on improving the child's attention span and independence. Consider engaging activities that encourage focus and self-confidence."

    response = {
        "model": MODEL_TAG.upper(),
        "predictions": predictions,
        "child_profile": {
            "strengths":         strengths,
            "areas_for_support": needs_support,
            "summary": (
                f"Strengths — {', '.join(strengths) if strengths else 'Still developing'}. "
                f"Needs support — {', '.join(needs_support) if needs_support else 'All areas strong!'}."
            ),
        },
        "ai_suggestions": ai_suggestions,
    }

    if errors:
        response["errors"] = errors

    log.info(f"Response: {response}")
    response_si = translate_response_values_to_sinhala(response)
    print("Translated Response:", jsonify(response_si))
    return jsonify(response_si), 200


if __name__ == "__main__":
    log.info(f"Model    : {MODEL_TAG.upper()}")
    log.info(f"Dir      : {MODELS_DIR.resolve()}")
    if not MODELS_DIR.exists():
        log.warning(f"'{MODELS_DIR}' not found — run the notebook first!")
    
    app.run(debug=True, host="192.168.43.243", port=5000)
    # app.run(debug=True, host="0.0.0.0", port=5000)

    #192.168.43.243  192.168.8.104