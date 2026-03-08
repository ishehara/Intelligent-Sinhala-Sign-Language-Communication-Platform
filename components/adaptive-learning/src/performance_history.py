"""
Performance History & Level-Aware Next-Letter Suggestion
=========================================================
Tracks individual attempt records for each user and provides:

1. **Attempt History** — Granular per-attempt records (sign, confidence,
   correct/incorrect, timestamp, level) saved to JSON for the history page.
2. **Level-Aware Suggestion** — Within a specific level (beginner, intermediate,
   advanced), recommends which letter the user should practise next based on
   their performance history (accuracy, confidence, attempts, mastery).

Persistence: `performance_history.json` in the same directory.
"""

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from collections import defaultdict

from rl_enhanced_agent import (
    SIGN_CURRICULUM,
    SIGN_TO_DIFFICULTY,
    ALL_SIGNS,
    UserSignState,
    FRONTEND_CURRICULUM,
    FRONTEND_STATIC_ORDER,
    FRONTEND_DYNAMIC_ORDER,
)

# ══════════════════════════════════════════════════════════════
# Attempt Record
# ══════════════════════════════════════════════════════════════

@dataclass
class AttemptRecord:
    """Single practice attempt by a user."""
    user_id: str
    sign: str
    level: str                # beginner / intermediate / advanced
    sign_type: str            # static / dynamic
    predicted_label: str
    expected_label: str
    is_correct: bool
    confidence: float
    response_time_ms: float
    feedback: str
    mastery_level: int
    timestamp: float          # Unix epoch


def _level_for_sign(sign: str) -> str:
    """Return the frontend curriculum level name for a sign."""
    # Check frontend curriculum first (more granular levels)
    for level_name, data in FRONTEND_CURRICULUM.items():
        if sign in data["signs"]:
            return level_name
    # Fallback to backend curriculum
    for level_name, data in SIGN_CURRICULUM.items():
        if sign in data["signs"]:
            return level_name
    return "unknown"


# ══════════════════════════════════════════════════════════════
# Performance History Manager
# ══════════════════════════════════════════════════════════════

class PerformanceHistory:
    """Persist and query individual attempt records."""

    def __init__(self, save_path: str = None):
        self.save_path = save_path or os.path.join(
            os.path.dirname(__file__), "performance_history.json"
        )
        # user_id -> list[AttemptRecord dict]
        self.records: Dict[str, List[dict]] = {}
        self._load()
        self._fix_unknown_levels()

    # ── Record an attempt ─────────────────────────────────────

    def add_attempt(
        self,
        user_id: str,
        sign: str,
        predicted_label: str,
        expected_label: str,
        is_correct: bool,
        confidence: float,
        response_time_ms: float = 0.0,
        feedback: str = "",
        mastery_level: int = 0,
        sign_type: str = "static",
    ) -> dict:
        """Save a single attempt and return the record."""
        record = AttemptRecord(
            user_id=user_id,
            sign=sign,
            level=_level_for_sign(sign),
            sign_type=sign_type,
            predicted_label=predicted_label,
            expected_label=expected_label,
            is_correct=is_correct,
            confidence=confidence,
            response_time_ms=response_time_ms,
            feedback=feedback,
            mastery_level=mastery_level,
            timestamp=time.time(),
        )
        rec_dict = asdict(record)
        self.records.setdefault(user_id, []).append(rec_dict)
        self._save()
        return rec_dict

    # ── Query history ─────────────────────────────────────────

    def get_history(
        self,
        user_id: str,
        level: str = None,
        sign: str = None,
        sign_type: str = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """
        Return paginated attempt history for a user.
        Optionally filter by level, sign, or sign_type.
        Most recent attempts come first.
        """
        all_records = self.records.get(user_id, [])

        filtered = all_records
        if level:
            filtered = [r for r in filtered if r["level"] == level]
        if sign:
            filtered = [r for r in filtered if r["sign"] == sign]
        if sign_type:
            filtered = [r for r in filtered if r["sign_type"] == sign_type]

        # Sort newest first
        filtered.sort(key=lambda r: r["timestamp"], reverse=True)
        total = len(filtered)
        page = filtered[offset: offset + limit]

        return {
            "user_id": user_id,
            "total": total,
            "offset": offset,
            "limit": limit,
            "records": page,
        }

    def get_level_summary(
        self,
        user_id: str,
        level: str,
        user_sign_states: Dict[str, UserSignState] = None,
    ) -> dict:
        """
        Aggregate stats per sign within a level.
        Merges data from attempt records AND the RL agent's UserSignState
        so the history page is never empty if the user has practised before.
        """
        records = self.records.get(user_id, [])

        # Look up signs in FRONTEND_CURRICULUM first, then SIGN_CURRICULUM
        signs_in_level = (
            FRONTEND_CURRICULUM.get(level, {}).get("signs", [])
            or SIGN_CURRICULUM.get(level, {}).get("signs", [])
        )

        per_sign: Dict[str, dict] = {}
        for sign in signs_in_level:
            # Match records by sign character (not level name, since a sign
            # may have been recorded under a different level name)
            sign_recs = [r for r in records if r["sign"] == sign]
            total = len(sign_recs)
            correct = sum(1 for r in sign_recs if r["is_correct"])
            confidences = [r["confidence"] for r in sign_recs]
            last_attempt = max((r["timestamp"] for r in sign_recs), default=None)

            # If no attempt records but RL agent has stats, use those
            rl_state = (user_sign_states or {}).get(sign)
            if total == 0 and rl_state and rl_state.total_attempts > 0:
                total = rl_state.total_attempts
                correct = rl_state.correct_attempts
                confidences = [rl_state.avg_confidence]
                last_attempt = rl_state.last_practiced

            per_sign[sign] = {
                "sign": sign,
                "total_attempts": total,
                "correct_attempts": correct,
                "accuracy": round(correct / total * 100, 1) if total else 0.0,
                "avg_confidence": round(
                    sum(confidences) / len(confidences) * 100, 1
                ) if confidences else 0.0,
                "best_confidence": round(
                    max(confidences) * 100, 1
                ) if confidences else 0.0,
                "last_attempt": last_attempt,
                "mastery_level": rl_state.mastery_level if rl_state else 0,
                "current_streak": rl_state.current_streak if rl_state else 0,
                "longest_streak": rl_state.longest_streak if rl_state else 0,
            }

        total_attempts = sum(s["total_attempts"] for s in per_sign.values())
        total_correct = sum(s["correct_attempts"] for s in per_sign.values())

        desc = (
            FRONTEND_CURRICULUM.get(level, {}).get("description", "")
            or SIGN_CURRICULUM.get(level, {}).get("description", "")
        )
        return {
            "level": level,
            "description": desc,
            "signs_count": len(signs_in_level),
            "total_attempts": total_attempts,
            "total_correct": total_correct,
            "overall_accuracy": round(
                total_correct / total_attempts * 100, 1
            ) if total_attempts else 0.0,
            "signs": per_sign,
        }

    # ── Persistence ───────────────────────────────────────────

    def _save(self):
        try:
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump(self.records, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Failed to save performance history: {e}")

    def _load(self):
        if not os.path.exists(self.save_path):
            print("🆕 Performance history: starting fresh")
            return
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                self.records = json.load(f)
            total = sum(len(v) for v in self.records.values())
            print(f"📂 Performance history loaded ({total} records, "
                  f"{len(self.records)} users)")
        except Exception as e:
            print(f"⚠️ Failed to load performance history: {e}")

    def _fix_unknown_levels(self):
        """Re-map any records with level='unknown' now that curriculum is expanded."""
        fixed = 0
        for user_id, recs in self.records.items():
            for r in recs:
                if r.get("level") == "unknown":
                    new_level = _level_for_sign(r.get("sign", ""))
                    if new_level != "unknown":
                        r["level"] = new_level
                        fixed += 1
        if fixed:
            print(f"🔧 Fixed {fixed} records with unknown level")
            self._save()


# ══════════════════════════════════════════════════════════════
# Level-Aware Next-Letter Suggestion
# ══════════════════════════════════════════════════════════════

def suggest_next_in_level(
    user_id: str,
    level: str,
    user_sign_states: Dict[str, UserSignState],
    history: PerformanceHistory,
) -> dict:
    """
    Within a specific level, recommend which letter to practise next.

    Algorithm — each sign in the level gets a priority score:
       +10  not attempted yet (introduce new letters)
       +8   accuracy < 50 %  (struggling → needs focus)
       +6   accuracy < 70 %  (not yet proficient)
       +5   due for spaced-rep review
       +4   avg confidence < 60 %  (low confidence even if correct)
       +2   avg confidence < 80 %
       +3   mastery_level == 0 (new) or 1 (learning)
       -3   mastery_level == 3 (mastered — deprioritise)

    Tie-breaker: sign with fewer total attempts goes first (less practised).
    Returns the recommended sign plus the ranked list for the entire level.
    """
    signs_in_level = (
        FRONTEND_CURRICULUM.get(level, {}).get("signs", [])
        or SIGN_CURRICULUM.get(level, {}).get("signs", [])
    )
    if not signs_in_level:
        return {"error": f"Unknown level: {level}"}

    level_summary = history.get_level_summary(user_id, level, user_sign_states)
    candidates = []

    for sign in signs_in_level:
        state = user_sign_states.get(sign)
        hist_stats = level_summary["signs"].get(sign, {})
        total = hist_stats.get("total_attempts", 0)
        accuracy = hist_stats.get("accuracy", 0.0) / 100.0  # convert to 0-1
        avg_conf = hist_stats.get("avg_confidence", 0.0) / 100.0

        priority = 0.0
        reasons = []

        # Also check RL agent total_attempts as fallback
        rl_total = state.total_attempts if state else 0
        effective_total = max(total, rl_total)

        if effective_total == 0:
            priority += 10.0
            reasons.append("not_attempted")
        else:
            if accuracy < 0.5:
                priority += 8.0
                reasons.append("low_accuracy")
            elif accuracy < 0.7:
                priority += 6.0
                reasons.append("needs_practice")

            if avg_conf < 0.6:
                priority += 4.0
                reasons.append("low_confidence")
            elif avg_conf < 0.8:
                priority += 2.0
                reasons.append("moderate_confidence")

        # Use RL agent state if available
        if state:
            if state.is_due_for_review and effective_total > 0:
                priority += 5.0
                reasons.append("due_for_review")

            if state.mastery_level <= 1:
                priority += 3.0
                reasons.append("not_mastered")
            elif state.mastery_level == 3:
                priority -= 3.0
                reasons.append("mastered")

        # Tie-breaker: fewer attempts → higher priority
        priority += max(0.0, (10 - total) * 0.1)

        candidates.append({
            "sign": sign,
            "priority": round(priority, 2),
            "reasons": reasons,
            "total_attempts": total,
            "accuracy": round(accuracy * 100, 1),
            "avg_confidence": round(avg_conf * 100, 1),
            "mastery_level": state.mastery_level if state else 0,
            "is_due_for_review": state.is_due_for_review if state else False,
        })

    candidates.sort(key=lambda c: (-c["priority"], c["total_attempts"]))

    recommended = candidates[0] if candidates else None
    reason_msg = _build_reason_message(recommended) if recommended else ""

    return {
        "level": level,
        "level_description": (
            FRONTEND_CURRICULUM.get(level, {}).get("description", "")
            or SIGN_CURRICULUM.get(level, {}).get("description", "")
        ),
        "recommended_sign": recommended["sign"] if recommended else None,
        "recommendation_reason": reason_msg,
        "all_signs": candidates,
        "level_summary": {
            "total_signs": len(signs_in_level),
            "total_attempts": level_summary["total_attempts"],
            "overall_accuracy": level_summary["overall_accuracy"],
        },
    }


def _build_reason_message(candidate: dict) -> str:
    """Human-readable reason for the recommendation."""
    reasons = candidate.get("reasons", [])
    sign = candidate["sign"]

    if "not_attempted" in reasons:
        return f"You haven't tried '{sign}' yet — let's learn it!"
    if "low_accuracy" in reasons:
        return (
            f"'{sign}' needs more practice — your accuracy is "
            f"{candidate['accuracy']:.0f}%. Let's improve it!"
        )
    if "due_for_review" in reasons:
        return f"Time to review '{sign}' to keep it fresh in your memory!"
    if "low_confidence" in reasons:
        return (
            f"Your confidence on '{sign}' is {candidate['avg_confidence']:.0f}%. "
            f"A bit more practice will help!"
        )
    if "needs_practice" in reasons:
        return (
            f"'{sign}' is at {candidate['accuracy']:.0f}% accuracy — "
            f"a few more tries and you'll have it!"
        )
    if "not_mastered" in reasons:
        return f"Keep practising '{sign}' to reach mastery!"
    return f"Practise '{sign}' next."


# ══════════════════════════════════════════════════════════════
# Level Completion, Unlock & Practice Suggestions
# ══════════════════════════════════════════════════════════════

# Thresholds for considering a sign "completed" within a level
SIGN_COMPLETION_CONFIDENCE = 0.50   # min confidence to count as completed
LEVEL_COMPLETION_RATIO = 1.0        # all signs must be completed to unlock next

# Confidence thresholds for categorising sign quality
POOR_CONFIDENCE_THRESHOLD = 0.60    # below this → poor, needs re-practice
GOOD_CONFIDENCE_THRESHOLD = 0.80    # above this → good

# Static level progression order (frontend-aligned)
STATIC_LEVEL_ORDER = FRONTEND_STATIC_ORDER
DYNAMIC_LEVEL_ORDER = FRONTEND_DYNAMIC_ORDER


def _is_sign_completed(sign_stats: dict) -> bool:
    """A sign is completed if the user got it correct at least once
    with confidence >= SIGN_COMPLETION_CONFIDENCE."""
    return (
        sign_stats.get("correct_attempts", 0) >= 1
        and sign_stats.get("best_confidence", 0.0) >= SIGN_COMPLETION_CONFIDENCE * 100
    )


def _sign_confidence_category(sign_stats: dict) -> str:
    """Categorise sign quality: 'poor', 'moderate', 'good'."""
    avg = sign_stats.get("avg_confidence", 0.0)
    if avg < POOR_CONFIDENCE_THRESHOLD * 100:
        return "poor"
    if avg < GOOD_CONFIDENCE_THRESHOLD * 100:
        return "moderate"
    return "good"


def get_level_progress(
    user_id: str,
    level: str,
    user_sign_states: Dict[str, UserSignState],
    history: PerformanceHistory,
) -> dict:
    """
    Return detailed level progress including:
    - Per-sign completion status (highlighted when correct)
    - Per-sign confidence category (poor / moderate / good)
    - Whether the level is complete
    - Poor signs that need re-practice
    """
    signs_in_level = (
        FRONTEND_CURRICULUM.get(level, {}).get("signs", [])
        or SIGN_CURRICULUM.get(level, {}).get("signs", [])
    )
    if not signs_in_level:
        return {"error": f"Unknown level: {level}"}

    level_summary = history.get_level_summary(user_id, level, user_sign_states)

    completed_signs = []
    poor_signs = []
    sign_details = []

    for sign in signs_in_level:
        stats = level_summary["signs"].get(sign, {})
        completed = _is_sign_completed(stats)
        category = _sign_confidence_category(stats)

        detail = {
            "sign": sign,
            "completed": completed,
            "confidence_category": category,  # poor / moderate / good
            "total_attempts": stats.get("total_attempts", 0),
            "correct_attempts": stats.get("correct_attempts", 0),
            "accuracy": stats.get("accuracy", 0.0),
            "avg_confidence": stats.get("avg_confidence", 0.0),
            "best_confidence": stats.get("best_confidence", 0.0),
            "mastery_level": stats.get("mastery_level", 0),
        }
        sign_details.append(detail)

        if completed:
            completed_signs.append(sign)
        if category == "poor" and stats.get("total_attempts", 0) > 0:
            poor_signs.append(detail)

    level_complete = len(completed_signs) == len(signs_in_level) and len(signs_in_level) > 0

    desc = (
        FRONTEND_CURRICULUM.get(level, {}).get("description", "")
        or SIGN_CURRICULUM.get(level, {}).get("description", "")
    )
    return {
        "level": level,
        "description": desc,
        "total_signs": len(signs_in_level),
        "completed_count": len(completed_signs),
        "completed_signs": completed_signs,
        "level_complete": level_complete,
        "poor_signs": poor_signs,
        "signs": sign_details,
        "overall_accuracy": level_summary.get("overall_accuracy", 0.0),
        "total_attempts": level_summary.get("total_attempts", 0),
    }


def get_unlocked_levels(
    user_id: str,
    user_sign_states: Dict[str, UserSignState],
    history: PerformanceHistory,
) -> List[str]:
    """
    Return a list of unlocked level names (frontend-aligned).
    Rules:
    - First static level and first dynamic level are always unlocked.
    - Each subsequent level unlocks when ALL signs in the previous level are completed.
    """
    unlocked: List[str] = []

    for order in [STATIC_LEVEL_ORDER, DYNAMIC_LEVEL_ORDER]:
        for i, level_name in enumerate(order):
            if i == 0:
                unlocked.append(level_name)
            else:
                prev_level = order[i - 1]
                prev_signs = FRONTEND_CURRICULUM.get(prev_level, {}).get("signs", [])
                prev_states = {s: user_sign_states.get(s, UserSignState(sign=s)) for s in prev_signs}
                prev_progress = get_level_progress(user_id, prev_level, prev_states, history)
                if prev_progress.get("level_complete", False):
                    unlocked.append(level_name)

    return unlocked


def get_practice_suggestions(
    user_id: str,
    user_sign_states: Dict[str, UserSignState],
    history: PerformanceHistory,
) -> List[dict]:
    """
    Return signs that need re-practice based on poor confidence.
    These are signs the user has attempted but performed poorly on.
    Sorted by worst confidence first — highlighted in history page.
    """
    suggestions = []

    for level_name in STATIC_LEVEL_ORDER + DYNAMIC_LEVEL_ORDER:
        signs_in_level = FRONTEND_CURRICULUM.get(level_name, {}).get("signs", [])
        states = {s: user_sign_states.get(s, UserSignState(sign=s)) for s in signs_in_level}
        progress = get_level_progress(user_id, level_name, states, history)

        for poor in progress.get("poor_signs", []):
            suggestions.append({
                **poor,
                "level": level_name,
                "suggestion": f"Re-practice '{poor['sign']}' — confidence is {poor['avg_confidence']:.0f}%",
            })

    # Sort by avg_confidence ascending (worst first)
    suggestions.sort(key=lambda s: s.get("avg_confidence", 0.0))
    return suggestions
