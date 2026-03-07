import os
import re
import json
import requests

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
API_URL    = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

TARGET_DISPLAY = {
    "communication_level":             "Communication",
    "instruction_understanding_level": "Instruction Understanding",
    "learning_retention_level":        "Learning & Retention",
    "attention_engagement_level":      "Attention & Engagement",
    "independence_confidence_level":   "Independence & Confidence",
}

AGE_GROUP_DISPLAY  = {0: "5-6 years", 1: "7-8 years", 2: "9-10 years"}
APP_USAGE_DISPLAY  = {0: "less than 10 min/day", 1: "10-20 min/day", 2: "more than 20 min/day"}


def _build_prompt(predictions, child_meta):
    age_str   = AGE_GROUP_DISPLAY.get(child_meta.get("age_group"), "unknown age")
    usage_str = APP_USAGE_DISPLAY.get(child_meta.get("app_usage_bucket"), "unknown usage")

    domain_lines = []
    for target, result in predictions.items():
        display = TARGET_DISPLAY.get(target, target)
        label   = result.get("prediction", "Unknown")
        conf    = result.get("confidence", 0)
        domain_lines.append(f"  - {display}: {label} (confidence {conf:.0%})")

    domains_text = "\n".join(domain_lines)

    return f"""You are a child development specialist helping parents of deaf children aged 5-10.

A parent submitted behavioral observations for their child through a sign language learning app.
The AI model assessed the child across 5 developmental domains.
Generate warm, practical, and encouraging suggestions for the parent.

CHILD PROFILE:
- Age group: {age_str}
- App usage: {usage_str}

BEHAVIORAL ASSESSMENT RESULTS:
{domains_text}

CONTEXT:
- This app uses visual-first interaction (sign language, pictures, no speech)
- Activities should be short, predictable, and child-friendly
- Suggestions must be non-diagnostic and non-medical
- Focus on what the parent can do at home, during play, and in daily routines

Respond ONLY with this JSON structure, no markdown, no extra text:
{{
  "overall_summary": "A warm 2-3 sentence summary of the child's current developmental profile",
  "priority_focus": "The single most important area to focus on right now and why",
  "domain_suggestions": {{
    "communication_level": {{
      "tip": "One specific actionable tip",
      "activity": "One simple home activity to try today"
    }},
    "instruction_understanding_level": {{
      "tip": "...",
      "activity": "..."
    }},
    "learning_retention_level": {{
      "tip": "...",
      "activity": "..."
    }},
    "attention_engagement_level": {{
      "tip": "...",
      "activity": "..."
    }},
    "independence_confidence_level": {{
      "tip": "...",
      "activity": "..."
    }}
  }},
  "weekly_plan": [
    "Day 1-2: ...",
    "Day 3-4: ...",
    "Day 5-7: ..."
  ],
  "encouragement": "A short warm message of encouragement to the parent"
}}"""


def _call_gemini(prompt):
    if not GEMINI_KEY:
        raise EnvironmentError("GEMINI_API_KEY not set. Run: export GEMINI_API_KEY='your_key'")

    response = requests.post(
        API_URL,
        headers={"Content-Type": "application/json"},
        params={"key": GEMINI_KEY},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature":     0.7,
                "topP":            0.9,
                "maxOutputTokens": 1500,
            },
        },
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Gemini API error {response.status_code}: {response.text}")

    try:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected Gemini response: {response.json()}") from e


def _parse_json(raw):
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$",          "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"raw_response": raw, "parse_error": "Could not parse JSON from response"}


def get_suggestions(predictions, child_meta=None):
    if child_meta is None:
        child_meta = {}
    try:
        prompt      = _build_prompt(predictions, child_meta)
        raw_text    = _call_gemini(prompt)
        suggestions = _parse_json(raw_text)
        return {"status": "ok", **suggestions}
    except Exception as e:
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    sample_predictions = {
        "communication_level":             {"prediction": "Developing",  "code": 1, "confidence": 0.72},
        "instruction_understanding_level": {"prediction": "Consistent",  "code": 2, "confidence": 0.65},
        "learning_retention_level":        {"prediction": "Moderate",    "code": 1, "confidence": 0.58},
        "attention_engagement_level":      {"prediction": "Short",       "code": 1, "confidence": 0.61},
        "independence_confidence_level":   {"prediction": "Assisted",    "code": 1, "confidence": 0.70},
    }
    sample_meta = {"age_group": 1, "app_usage_bucket": 2}

    result = get_suggestions(sample_predictions, sample_meta)
    print(json.dumps(result, indent=2))