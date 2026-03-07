from deep_translator import GoogleTranslator

_translator = GoogleTranslator(source="en", target="si")

# Keys whose string values should NOT be translated
# (they are codes/identifiers used programmatically)
_SKIP_KEYS = {
    "model", "prediction", "code", "confidence",
}

# Keys whose values ARE human-readable text that should be translated
_TRANSLATE_KEYS = {
    "summary", "strengths", "areas_for_support",
    "ai_suggestions", "error", "missing", "invalid",
}


def translate_text_si(text: str) -> str:
    """Translate a single English string to Sinhala."""
    try:
        return _translator.translate(text)
    except Exception:
        return text  # fallback: return original on failure


def translate_response_values_to_sinhala(data, _parent_key: str = ""):
    """
    Recursively translate only human-readable string values in the response.
    Dict keys are never translated. Values under structural/code keys are skipped.
    """
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            if k in _SKIP_KEYS:
                result[k] = v  # keep as-is
            elif k == "probabilities":
                # Translate the label keys inside probabilities dict (e.g. "Emerging" -> Sinhala)
                result[k] = {translate_text_si(label): round(prob, 4) for label, prob in v.items()}
            else:
                result[k] = translate_response_values_to_sinhala(v, _parent_key=k)
        return result

    if isinstance(data, list):
        return [translate_response_values_to_sinhala(item, _parent_key=_parent_key) for item in data]

    if isinstance(data, str):
        return translate_text_si(data)

    return data  # int, float, bool, None — unchanged