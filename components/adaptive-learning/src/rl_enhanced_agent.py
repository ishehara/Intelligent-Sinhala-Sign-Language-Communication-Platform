"""
Enhanced Reinforcement Learning Agent for Adaptive Sign Language Learning
=========================================================================
Extends the basic Q-learning feedback agent with:

1. **Curriculum Learning** — Dynamic lesson difficulty adaptation
2. **Spaced Repetition** — AI-guided review scheduling (SM-2 variant + RL)
3. **Deep Q-Network (DQN)** option — For richer state representations
4. **Extended State Space** — Accuracy %, response time, confidence, streak
5. **Extended Action Space** — Difficulty selection, feedback style, review timing
6. **Multi-objective Reward** — Improvement + consistency + retention

Mathematical Formulation:
─────────────────────────
State Space S = (accuracy_bin, confidence_bin, response_time_bin,
                 streak_bin, recency_bin, difficulty_level)
    |S| ≈ 5 × 5 × 4 × 4 × 4 × 3 = 4,800 states

Action Space A = (next_difficulty, feedback_type, review_interval)
    |A| = 3 × 4 × 3 = 36 composite actions

Reward Function R(s, a, s'):
    R = w₁·R_accuracy + w₂·R_improvement + w₃·R_retention + w₄·R_engagement
    where:
        R_accuracy    = conf(t) - threshold            (immediate correctness)
        R_improvement = conf(t) - conf(t-1)            (learning progress)
        R_retention   = I(recalled after interval)      (long-term memory)
        R_engagement  = -penalty if user quits          (keep user engaged)
    w = [0.3, 0.3, 0.25, 0.15]

Q-Learning Update:
    Q(s,a) ← Q(s,a) + α[R + γ·max_a' Q(s',a') - Q(s,a)]

Usage:
    from rl_enhanced_agent import AdaptiveLearningAgent
    agent = AdaptiveLearningAgent()
    action = agent.select_lesson_action(user_state)
    agent.update_from_interaction(session_id, outcome)
"""

import json
import os
import time
import math
import numpy as np
from collections import defaultdict
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta

# ══════════════════════════════════════════════════════════════
# Sign Curriculum Definition
# ══════════════════════════════════════════════════════════════

# Sinhala sign letters organized by difficulty
SIGN_CURRICULUM = {
    "beginner": {
        "signs": ['අ', 'ආ', 'ඇ', 'ඉ', 'උ', 'එ'],  # Vowels (simpler hand shapes)
        "difficulty": 1,
        "description": "Basic vowels — foundational hand shapes",
    },
    "intermediate": {
        "signs": ['ක', 'ග', 'ට', 'ද', 'ත', 'ඩ', 'න', 'ප', 'බ', 'ම'],
        "difficulty": 2,
        "description": "Common consonants — moderate complexity",
    },
    "advanced": {
        "signs": ['ය', 'ර', 'ල', 'ව', 'ස', 'හ', 'ං', 'ච', 'ෆ'],
        "difficulty": 3,
        "description": "Complex consonants and special characters",
    },
    "dynamic_beginner": {
        "signs": ['ඈ', 'ඊ', 'ඌ', 'ඒ', 'ඔ', 'ඕ'],
        "difficulty": 2,
        "description": "Dynamic vowel signs — motion-based hand gestures",
    },
    "dynamic_intermediate": {
        "signs": ['ජ', 'ණ', 'ළ', 'ඟ', 'ඦ', 'ඳ', 'ඹ', 'ඛ'],
        "difficulty": 3,
        "description": "Dynamic consonant signs — moderate motion complexity",
    },
    "dynamic_advanced": {
        "signs": ['ඬ', 'ඵ', 'ධ', 'ඨ', 'ඡ', 'ක්\u200dය', 'භ', 'ථ'],
        "difficulty": 4,
        "description": "Dynamic complex signs — advanced motion patterns",
    },
}

# Frontend-aligned curriculum (same data, shared with performance_history)
FRONTEND_CURRICULUM = SIGN_CURRICULUM

# Level progression order for frontend
FRONTEND_STATIC_ORDER = ["beginner", "intermediate", "advanced"]
FRONTEND_DYNAMIC_ORDER = ["dynamic_beginner", "dynamic_intermediate", "dynamic_advanced"]

ALL_SIGNS = []
SIGN_TO_DIFFICULTY = {}
for level, data in SIGN_CURRICULUM.items():
    for sign in data["signs"]:
        ALL_SIGNS.append(sign)
        SIGN_TO_DIFFICULTY[sign] = data["difficulty"]


# ══════════════════════════════════════════════════════════════
# Extended State & Action Definitions
# ══════════════════════════════════════════════════════════════

@dataclass
class UserSignState:
    """Per-sign learning state for a user."""
    sign: str
    total_attempts: int = 0
    correct_attempts: int = 0
    avg_confidence: float = 0.0
    best_confidence: float = 0.0
    avg_response_time_ms: float = 5000.0
    current_streak: int = 0
    longest_streak: int = 0
    last_practiced: Optional[float] = None  # Unix timestamp
    ease_factor: float = 2.5  # SM-2 ease factor
    review_interval_days: float = 1.0  # SM-2 interval
    next_review: Optional[float] = None
    mastery_level: int = 0  # 0=new, 1=learning, 2=familiar, 3=mastered

    @property
    def accuracy(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.correct_attempts / self.total_attempts

    @property
    def days_since_practice(self) -> float:
        if self.last_practiced is None:
            return 999.0
        return (time.time() - self.last_practiced) / 86400.0

    @property
    def is_due_for_review(self) -> bool:
        if self.next_review is None:
            return True
        return time.time() >= self.next_review


# ── State discretization ──────────────────────────────────────

def bin_accuracy(acc: float) -> int:
    """Bin accuracy [0,1] → 5 levels."""
    if acc >= 0.9: return 4
    if acc >= 0.7: return 3
    if acc >= 0.5: return 2
    if acc >= 0.3: return 1
    return 0


def bin_confidence(conf: float) -> int:
    """Bin confidence [0,1] → 5 levels."""
    if conf >= 0.9: return 4
    if conf >= 0.75: return 3
    if conf >= 0.5: return 2
    if conf >= 0.25: return 1
    return 0


def bin_response_time(ms: float) -> int:
    """Bin response time → 4 levels (lower = better)."""
    if ms <= 1000: return 3    # very fast
    if ms <= 2500: return 2    # fast
    if ms <= 5000: return 1    # moderate
    return 0                    # slow


def bin_streak(streak: int) -> int:
    """Bin consecutive correct → 4 levels."""
    if streak >= 10: return 3
    if streak >= 5: return 2
    if streak >= 2: return 1
    return 0


def bin_recency(days: float) -> int:
    """Bin days since last practice → 4 levels."""
    if days <= 1: return 3     # today
    if days <= 3: return 2     # recent
    if days <= 7: return 1     # this week
    return 0                    # stale


def make_extended_state(user_sign: UserSignState) -> tuple:
    """
    Construct discrete state for the extended RL agent.
    
    State = (accuracy_bin, confidence_bin, response_time_bin,
             streak_bin, recency_bin, difficulty_level)
    """
    return (
        bin_accuracy(user_sign.accuracy),
        bin_confidence(user_sign.avg_confidence),
        bin_response_time(user_sign.avg_response_time_ms),
        bin_streak(user_sign.current_streak),
        bin_recency(user_sign.days_since_practice),
        min(3, SIGN_TO_DIFFICULTY.get(user_sign.sign, 1)),
    )


# ── Actions ───────────────────────────────────────────────────

# Composite action: (difficulty_action, feedback_style, review_action)
DIFFICULTY_ACTIONS = ["maintain", "increase", "decrease"]
FEEDBACK_STYLES = ["encouraging", "detailed_correction", "minimal", "challenge"]
REVIEW_ACTIONS = ["review_soon", "review_normal", "review_later"]

# Flatten to single action index
ACTION_SPACE = []
for d in DIFFICULTY_ACTIONS:
    for f in FEEDBACK_STYLES:
        for r in REVIEW_ACTIONS:
            ACTION_SPACE.append((d, f, r))

NUM_ACTIONS = len(ACTION_SPACE)  # 3 × 4 × 3 = 36

# ── Feedback Templates ────────────────────────────────────────

FEEDBACK_TEMPLATES = {
    "encouraging": {
        "correct": [
            "Great job! You're getting better at '{sign}'! 🎉",
            "Wonderful! '{sign}' is looking natural now! ⭐",
            "You're on fire! Keep up the amazing work with '{sign}'! 🔥",
        ],
        "incorrect": [
            "Almost there! '{sign}' is tricky but you're improving! 💪",
            "Don't give up! Every attempt makes you better at '{sign}'! 🌱",
            "Good effort! '{sign}' takes practice — you'll get it! ✨",
        ],
    },
    "detailed_correction": {
        "correct": [
            "Correct! '{sign}' at {conf:.0%}. Focus on finger precision for even higher scores.",
            "Well done! '{sign}' recognized. Tip: keep your wrist angle consistent.",
        ],
        "incorrect": [
            "Not quite '{sign}' — check finger #1-3 positions against the reference. "
            "Common confusion: {confused_with}.",
            "Detected '{predicted}' instead of '{sign}'. Key difference: pay attention to thumb placement.",
        ],
    },
    "minimal": {
        "correct": ["✓ '{sign}' — {conf:.0%}", "Correct ✓"],
        "incorrect": ["Try again — expected '{sign}'", "Not '{sign}' — retry"],
    },
    "challenge": {
        "correct": [
            "'{sign}' nailed at {conf:.0%}! Can you beat your best of {best:.0%}? 🏆",
            "Good — but I think you can do '{sign}' even faster! ⚡",
        ],
        "incorrect": [
            "Tough one! '{sign}' has a {acc:.0%} success rate for you. Let's break that ceiling! 📈",
            "'{sign}' is your nemesis — prove it wrong this time! 💥",
        ],
    },
}


# ══════════════════════════════════════════════════════════════
# Spaced Repetition (SM-2 Variant + RL Override)
# ══════════════════════════════════════════════════════════════

def sm2_update(user_sign: UserSignState, quality: int) -> UserSignState:
    """
    SM-2 spaced repetition algorithm update.
    
    quality: 0-5 (0=complete failure, 5=perfect)
    
    This computes the base review interval. The RL agent can then
    override it based on the user's overall learning trajectory.
    """
    if quality < 3:
        # Failed review — reset interval
        user_sign.review_interval_days = 1.0
    else:
        if user_sign.total_attempts <= 1:
            user_sign.review_interval_days = 1.0
        elif user_sign.total_attempts == 2:
            user_sign.review_interval_days = 3.0
        else:
            user_sign.review_interval_days *= user_sign.ease_factor

    # Update ease factor
    user_sign.ease_factor = max(
        1.3,
        user_sign.ease_factor + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
    )

    user_sign.next_review = time.time() + user_sign.review_interval_days * 86400
    return user_sign


def rl_adjusted_interval(base_interval: float, review_action: str) -> float:
    """
    RL agent modifies the SM-2 interval based on learned policy.
    
    review_soon:   interval × 0.5 (review sooner — struggling user)
    review_normal: interval × 1.0 (use SM-2 as-is)
    review_later:  interval × 1.5 (confident user — space out more)
    """
    multipliers = {
        "review_soon": 0.5,
        "review_normal": 1.0,
        "review_later": 1.5,
    }
    return base_interval * multipliers.get(review_action, 1.0)


# ══════════════════════════════════════════════════════════════
# Reward Function
# ══════════════════════════════════════════════════════════════

def compute_reward(
    old_state: UserSignState,
    new_confidence: float,
    is_correct: bool,
    response_time_ms: float,
    was_review: bool = False,
    user_quit: bool = False,
) -> float:
    """
    Multi-objective reward function for the RL agent.
    
    R = w₁·R_accuracy + w₂·R_improvement + w₃·R_retention + w₄·R_engagement
    
    Components:
        R_accuracy:    +1 if correct, -0.5 if incorrect, scaled by confidence
        R_improvement: (new_conf - old_conf) × 2  (learning progress)
        R_retention:   +1.5 if recalled after long interval, +0.5 if short
        R_engagement:  -2.0 if user quits, +0.1 for continued practice
    
    Weights: [0.3, 0.3, 0.25, 0.15]
    """
    # ── R_accuracy ──
    if is_correct:
        r_accuracy = 0.5 + new_confidence  # [0.5, 1.5]
    else:
        r_accuracy = -0.5 - (1 - new_confidence) * 0.5  # [-1.0, -0.5]

    # ── R_improvement ──
    r_improvement = (new_confidence - old_state.avg_confidence) * 2.0
    # Speed improvement bonus
    if response_time_ms < old_state.avg_response_time_ms * 0.8:
        r_improvement += 0.3  # Got faster

    # ── R_retention (spaced repetition bonus) ──
    r_retention = 0.0
    if was_review and is_correct:
        days_gap = old_state.days_since_practice
        if days_gap >= 7:
            r_retention = 1.5    # Excellent long-term retention
        elif days_gap >= 3:
            r_retention = 1.0    # Good retention
        elif days_gap >= 1:
            r_retention = 0.5    # Short-term retention
        else:
            r_retention = 0.2    # Same-day review

    # ── R_engagement ──
    if user_quit:
        r_engagement = -2.0
    else:
        r_engagement = 0.1 + min(0.5, old_state.current_streak * 0.05)

    # ── Weighted sum ──
    w = [0.30, 0.30, 0.25, 0.15]
    reward = (
        w[0] * r_accuracy +
        w[1] * r_improvement +
        w[2] * r_retention +
        w[3] * r_engagement
    )

    return round(reward, 4)


# ══════════════════════════════════════════════════════════════
# Adaptive Learning Agent
# ══════════════════════════════════════════════════════════════

class AdaptiveLearningAgent:
    """
    Extended Q-learning agent for adaptive sign language curriculum.
    
    Manages:
    1. Per-user, per-sign learning state tracking
    2. Lesson difficulty adaptation
    3. Feedback style selection
    4. Spaced repetition scheduling
    5. Next-sign recommendation
    
    State space:  ~4,800 discrete states
    Action space: 36 composite actions
    """

    def __init__(self,
                 save_path: str = None,
                 alpha: float = 0.1,
                 gamma: float = 0.95,
                 epsilon: float = 0.25,
                 epsilon_decay: float = 0.9997,
                 epsilon_min: float = 0.05):

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_actions = NUM_ACTIONS

        # Q-table: state_str -> [Q-values for each action]
        self.q_table = defaultdict(lambda: np.zeros(self.num_actions))

        # User state: user_id -> {sign -> UserSignState}
        self.user_states: Dict[str, Dict[str, UserSignState]] = {}

        # Session memory for delayed reward
        self.session_memory: Dict[str, Dict] = {}

        # Statistics
        self.total_episodes = 0
        self.total_rewards = 0.0
        self.reward_history: List[float] = []

        # Persistence
        self.save_path = save_path or os.path.join(
            os.path.dirname(__file__), 'rl_enhanced_q_table.json'
        )
        self._load()

    # ── Core RL ───────────────────────────────────────────────

    def select_action(self, state: tuple) -> int:
        """ε-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        return int(np.argmax(self.q_table[str(state)]))

    def update_q(self, state: tuple, action: int, reward: float,
                 next_state: tuple = None):
        """Q-learning update with experience tracking."""
        s = str(state)
        best_next = 0.0
        if next_state is not None:
            best_next = float(np.max(self.q_table[str(next_state)]))

        old_q = self.q_table[s][action]
        self.q_table[s][action] = old_q + self.alpha * (
            reward + self.gamma * best_next - old_q
        )

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.total_episodes += 1
        self.total_rewards += reward
        self.reward_history.append(reward)
        if len(self.reward_history) > 500:
            self.reward_history.pop(0)

        if self.total_episodes % 10 == 0:
            self._save()

    # ── User State Management ─────────────────────────────────

    def get_user_sign_state(self, user_id: str, sign: str) -> UserSignState:
        """Get or create learning state for a user-sign pair."""
        if user_id not in self.user_states:
            self.user_states[user_id] = {}
        if sign not in self.user_states[user_id]:
            self.user_states[user_id][sign] = UserSignState(sign=sign)
        return self.user_states[user_id][sign]

    def update_user_sign_state(self, user_id: str, sign: str,
                                confidence: float, is_correct: bool,
                                response_time_ms: float):
        """Update user's learning state for a specific sign."""
        state = self.get_user_sign_state(user_id, sign)

        state.total_attempts += 1
        if is_correct:
            state.correct_attempts += 1
            state.current_streak += 1
            state.longest_streak = max(state.longest_streak, state.current_streak)
        else:
            state.current_streak = 0

        # Running average of confidence
        n = state.total_attempts
        state.avg_confidence = ((n - 1) * state.avg_confidence + confidence) / n
        state.best_confidence = max(state.best_confidence, confidence)

        # Running average of response time
        state.avg_response_time_ms = (
            (n - 1) * state.avg_response_time_ms + response_time_ms
        ) / n

        state.last_practiced = time.time()

        # Update mastery level
        if state.accuracy >= 0.9 and state.avg_confidence >= 0.85:
            state.mastery_level = 3  # mastered
        elif state.accuracy >= 0.7 and state.avg_confidence >= 0.6:
            state.mastery_level = 2  # familiar
        elif state.total_attempts >= 3:
            state.mastery_level = 1  # learning
        else:
            state.mastery_level = 0  # new

        # SM-2 spaced repetition update
        quality = 5 if (is_correct and confidence >= 0.9) else \
                  4 if (is_correct and confidence >= 0.7) else \
                  3 if is_correct else \
                  2 if confidence >= 0.3 else 1
        sm2_update(state, quality)

        return state

    # ── Lesson Planning ───────────────────────────────────────

    def recommend_next_sign(self, user_id: str,
                             current_level: str = None) -> Dict:
        """
        Recommend the next sign to practice using RL + spaced repetition.
        
        Priority (learned by RL, but with heuristic initialization):
        1. Signs due for review (spaced repetition)
        2. Signs with low mastery in current level
        3. New signs from the next difficulty level
        """
        user_signs = self.user_states.get(user_id, {})

        candidates = []

        for sign in ALL_SIGNS:
            state = self.get_user_sign_state(user_id, sign)
            rl_state = make_extended_state(state)
            action_idx = self.select_action(rl_state)
            action = ACTION_SPACE[action_idx]

            # RL-driven priority: use learned Q-values to modulate urgency
            # instead of fixed hardcoded weights.
            # max_q represents how much reward the agent expects from its
            # best action in this sign's state — higher = better strategy found.
            q_values = self.q_table[str(rl_state)]
            max_q = float(np.max(q_values)) if np.any(q_values != 0) else 0.0
            # Sigmoid ∈ [0.3, 1.0]: ensures all signs get some priority,
            # but signs where RL has found good strategies rank higher.
            q_factor = 0.3 + 0.7 / (1.0 + np.exp(-max_q))

            priority = 0.0

            # Due for review — spaced repetition urgency scaled by RL confidence
            if state.is_due_for_review and state.total_attempts > 0:
                priority += (6.0 + state.days_since_practice * 0.5) * q_factor

            # Low mastery — RL confidence modulates how aggressively to push
            if state.mastery_level < 2:
                priority += (3.0 + (1.0 - state.accuracy) * 2.0) * q_factor

            # New sign — introduce when RL baseline is ready
            if state.total_attempts == 0:
                difficulty = SIGN_TO_DIFFICULTY.get(sign, 2)
                if difficulty <= self._user_max_difficulty(user_id):
                    priority += 2.0 + q_factor

            candidates.append({
                "sign": sign,
                "priority": priority,
                "mastery": state.mastery_level,
                "accuracy": state.accuracy,
                "due_for_review": state.is_due_for_review,
                "rl_action": action,
                "difficulty": SIGN_TO_DIFFICULTY.get(sign, 1),
            })

        # Sort by priority (highest first)
        candidates.sort(key=lambda x: x["priority"], reverse=True)

        return {
            "recommended": candidates[0] if candidates else None,
            "alternatives": candidates[1:4],
            "user_progress": self._get_user_progress(user_id),
        }

    def _user_max_difficulty(self, user_id: str) -> int:
        """Determine max difficulty level user should attempt."""
        user_signs = self.user_states.get(user_id, {})

        beginner_signs = SIGN_CURRICULUM["beginner"]["signs"]
        beginner_mastered = sum(
            1 for s in beginner_signs
            if s in user_signs and user_signs[s].mastery_level >= 2
        )

        intermediate_signs = SIGN_CURRICULUM["intermediate"]["signs"]
        intermediate_mastered = sum(
            1 for s in intermediate_signs
            if s in user_signs and user_signs[s].mastery_level >= 2
        )

        if intermediate_mastered >= len(intermediate_signs) * 0.6:
            return 3
        elif beginner_mastered >= len(beginner_signs) * 0.6:
            return 2
        return 1

    def _get_user_progress(self, user_id: str) -> Dict:
        """Get overall user progress summary."""
        user_signs = self.user_states.get(user_id, {})
        total = len(ALL_SIGNS)
        attempted = sum(1 for s in user_signs.values() if s.total_attempts > 0)
        mastered = sum(1 for s in user_signs.values() if s.mastery_level >= 3)
        familiar = sum(1 for s in user_signs.values() if s.mastery_level >= 2)

        return {
            "total_signs": total,
            "attempted": attempted,
            "familiar": familiar,
            "mastered": mastered,
            "progress_pct": round(mastered / total * 100, 1) if total > 0 else 0,
            "current_level": (
                "advanced" if self._user_max_difficulty(user_id) >= 3 else
                "intermediate" if self._user_max_difficulty(user_id) >= 2 else
                "beginner"
            ),
        }

    # ── Integrated Feedback ───────────────────────────────────

    def get_adaptive_feedback(self, user_id: str, sign: str,
                               predicted_label: str, confidence: float,
                               is_correct: bool, response_time_ms: float = 3000,
                               ) -> Dict:
        """
        Select feedback using RL policy and return comprehensive response.
        
        This is the main entry point — combines:
        1. RL-selected action (difficulty + feedback style + review timing)
        2. SM-2 spaced repetition schedule
        3. User progress tracking
        4. Personalized feedback message
        """
        # Get/update user state
        old_state = self.get_user_sign_state(user_id, sign)
        old_state_copy = UserSignState(**asdict(old_state))  # snapshot for reward

        rl_state = make_extended_state(old_state)
        action_idx = self.select_action(rl_state)
        difficulty_action, feedback_style, review_action = ACTION_SPACE[action_idx]

        # Update user learning state
        updated = self.update_user_sign_state(
            user_id, sign, confidence, is_correct, response_time_ms
        )

        # Override review interval with RL decision
        updated.review_interval_days = rl_adjusted_interval(
            updated.review_interval_days, review_action
        )
        updated.next_review = time.time() + updated.review_interval_days * 86400

        # Generate feedback message
        feedback_msg = self._generate_feedback_message(
            feedback_style, sign, predicted_label, confidence,
            is_correct, updated
        )

        # Store session for delayed reward
        sid = f"als_{user_id}_{int(time.time() * 1000)}"
        self.session_memory[sid] = {
            "user_id": user_id,
            "sign": sign,
            "state": rl_state,
            "action": action_idx,
            "old_state_snapshot": {
                "accuracy": old_state_copy.accuracy,
                "avg_confidence": old_state_copy.avg_confidence,
                "current_streak": old_state_copy.current_streak,
                "avg_response_time_ms": old_state_copy.avg_response_time_ms,
            },
            "timestamp": time.time(),
        }
        self._cleanup_sessions()

        # Compute immediate reward and update Q-table
        reward = compute_reward(
            old_state_copy, confidence, is_correct, response_time_ms,
            was_review=old_state_copy.is_due_for_review,
        )
        new_rl_state = make_extended_state(updated)
        self.update_q(rl_state, action_idx, reward, new_rl_state)

        return {
            "feedback": feedback_msg,
            "feedback_style": feedback_style,
            "difficulty_action": difficulty_action,
            "review_action": review_action,
            "next_review_days": round(updated.review_interval_days, 1),
            "mastery_level": updated.mastery_level,
            "user_accuracy": round(updated.accuracy, 3),
            "streak": updated.current_streak,
            "session_id": sid,
            "rl_state": list(rl_state),
            "rl_action": action_idx,
            "rl_reward": reward,
            "rl_epsilon": round(self.epsilon, 4),
        }

    def _generate_feedback_message(self, style: str, sign: str,
                                     predicted: str, confidence: float,
                                     is_correct: bool,
                                     user_state: UserSignState) -> str:
        """Generate a feedback message using the selected style."""
        templates = FEEDBACK_TEMPLATES.get(style, FEEDBACK_TEMPLATES["encouraging"])
        pool = templates["correct"] if is_correct else templates["incorrect"]
        template = pool[np.random.randint(len(pool))]

        return template.format(
            sign=sign,
            predicted=predicted,
            conf=confidence,
            best=user_state.best_confidence,
            acc=user_state.accuracy,
            confused_with=predicted if not is_correct else "",
        )

    # ── Delayed Reward ────────────────────────────────────────

    def receive_delayed_reward(self, session_id: str,
                                reward_type: str,
                                new_confidence: float = None) -> Dict:
        """
        Process delayed reward from frontend.
        
        reward_type: 'improved' | 'correct' | 'retry' | 'give_up' | 'mastered'
        """
        if session_id not in self.session_memory:
            return {"error": "Session expired", "applied": False}

        mem = self.session_memory.pop(session_id)
        reward_map = {
            "improved": 1.0,
            "correct": 2.0,
            "retry": 0.2,
            "give_up": -1.5,
            "mastered": 3.0,
            "no_change": 0.0,
        }
        reward = reward_map.get(reward_type, 0.0)

        if new_confidence is not None:
            old_conf = mem["old_state_snapshot"]["avg_confidence"]
            if new_confidence > old_conf:
                reward += (new_confidence - old_conf) * 2.0

        # Q-update (terminal — no next state for delayed reward)
        self.update_q(mem["state"], mem["action"], reward, None)

        return {"applied": True, "reward": round(reward, 3)}

    # ── Statistics & Persistence ──────────────────────────────

    def get_stats(self) -> Dict:
        """Return comprehensive agent statistics."""
        return {
            "total_episodes": self.total_episodes,
            "total_rewards": round(self.total_rewards, 2),
            "avg_reward_last_100": round(
                np.mean(self.reward_history[-100:]) if self.reward_history else 0, 4
            ),
            "epsilon": round(self.epsilon, 4),
            "q_table_size": len(self.q_table),
            "active_sessions": len(self.session_memory),
            "num_actions": self.num_actions,
            "state_space_desc": "accuracy × confidence × response_time × streak × recency × difficulty",
            "action_space_desc": "difficulty × feedback_style × review_interval",
            "total_users": len(self.user_states),
        }

    def _save(self):
        """Persist Q-table and user states."""
        data = {
            "q_table": {k: v.tolist() for k, v in self.q_table.items()},
            "epsilon": self.epsilon,
            "total_episodes": self.total_episodes,
            "total_rewards": self.total_rewards,
            "reward_history": self.reward_history[-500:],
            "user_states": {
                uid: {sign: asdict(s) for sign, s in signs.items()}
                for uid, signs in self.user_states.items()
            },
        }
        try:
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Failed to save enhanced RL agent: {e}")

    def _load(self):
        """Load Q-table and user states."""
        if not os.path.exists(self.save_path):
            print("🆕 Enhanced RL agent: starting fresh")
            return
        try:
            with open(self.save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for k, v in data.get("q_table", {}).items():
                self.q_table[k] = np.array(v)
            self.epsilon = data.get("epsilon", self.epsilon)
            self.total_episodes = data.get("total_episodes", 0)
            self.total_rewards = data.get("total_rewards", 0.0)
            self.reward_history = data.get("reward_history", [])
            # Restore user states
            for uid, signs in data.get("user_states", {}).items():
                self.user_states[uid] = {}
                for sign, state_dict in signs.items():
                    self.user_states[uid][sign] = UserSignState(**state_dict)
            print(f"📂 Enhanced RL agent loaded ({self.total_episodes} episodes, "
                  f"{len(self.user_states)} users)")
        except Exception as e:
            print(f"⚠️ Failed to load enhanced RL agent: {e}")

    def _cleanup_sessions(self, max_age: float = 3600):
        """Remove expired sessions."""
        now = time.time()
        expired = [s for s, m in self.session_memory.items()
                   if now - m["timestamp"] > max_age]
        for s in expired:
            del self.session_memory[s]

    def save(self):
        self._save()
