"""
Contextual Thompson Sampling RL Feedback Agent
===============================================
Based on simulation results in sinhala_sign_rl_simulator.py, which compared
5 RL strategies (Random, Rule-Based, Thompson Sampling, Epsilon-Greedy, UCB)
over 100 simulated users × 3 learning profiles.

Thompson Sampling was the top performer:
  • Highest average signs completed
  • Highest user retention (lowest quit rate)
  • Best level 2/3 progression
  • Fast contextual adaptation — no cold-start penalty

Algorithm: Contextual Bandit with Beta distributions
─────────────────────────────────────────────────────
  For each (context, action) pair we maintain Beta(α, β).
  At decision time:
    1. Determine context from user state
    2. Apply hard constraints (e.g., no 'encouraging' when frustrated)
    3. Draw sample θᵢ ~ Beta(αᵢ, βᵢ) for each valid action
    4. Select action with highest θᵢ
  On reward arrival:
    success signal → α += weight   (posterior shifts toward this action)
    failure signal → β += weight   (posterior shifts away)

Context Space (5):
  frustrated     consecutive_failures ≥ 3
  struggling     confidence < 0.40 AND not correct
  learning       attempt_count ≤ 2 (first encounters with the sign)
  confident      success_rate_last5 > 0.70 AND confidence ≥ 0.75
  normal         all other states

Action Space (6):
  encouraging      positive reinforcement messages
  corrective       specific form-correction advice
  hint             gentle directional nudge
  video_demo       reference video / detailed visual guide
  skip_option      allow user to move to next sign
  break_suggestion suggest a short rest

Hard Constraints (from simulator — proven to reduce quit rate):
  consecutive_failures ≥ 5 → only [video_demo, skip_option, break_suggestion]
  consecutive_failures ≥ 3 → exclude 'encouraging'

Reward map (from frontend signals):
  correct    → +2.0   sign matched
  improved   → +1.0   confidence increase
  retry      → +0.3   user is still trying (engagement)
  no_change  →  0.0   no measurable change
  give_up    → -1.5   skipped / gave up

Persistence: Beta parameters serialised to JSON so learning survives restarts.

Integration:
  from rl_thompson_agent import ThompsonFeedbackAgent
  agent = ThompsonFeedbackAgent()
  result = agent.get_feedback(confidence, is_correct, hand_detected,
                              attempt_count, predicted_label, expected_label)
  # later, when the user's next attempt arrives:
  agent.receive_reward(result['session_id'], 'improved',
                       new_confidence=0.92, new_is_correct=True)
"""

import json
import os

import time
import numpy as np
from typing import Dict, List, Optional, Tuple

# ══════════════════════════════════════════════════════════════
# Context Definitions
# ══════════════════════════════════════════════════════════════

CONTEXTS = ['normal', 'frustrated', 'struggling', 'learning', 'confident']

# ══════════════════════════════════════════════════════════════
# Action Definitions (6 feedback strategies → message templates)
# ══════════════════════════════════════════════════════════════

ACTIONS = [
    'encouraging',
    'corrective',
    'hint',
    'video_demo',
    'skip_option',
    'break_suggestion',
]

# Per-action feedback templates ────────────────────────────────
# Uses {sign}, {expected}, {conf_pct}, {streak} placeholders.

FEEDBACK_TEMPLATES: Dict[str, Dict] = {
    'encouraging': {
        'templates': [
            "Great effort on '{sign}'! You're at {conf_pct:.0f}% — keep going! 💪",
            "Well done! '{sign}' is looking better at {conf_pct:.0f}%! ⭐",
            "You're improving! '{sign}' at {conf_pct:.0f}% — stay consistent! 🔥",
            "Nice work on '{sign}'! {conf_pct:.0f}% confidence — almost there! ✨",
        ],
        'tip': "Keep practicing — repetition builds muscle memory.",
        'level': 'good',
    },
    'corrective': {
        'templates': [
            "'{sign}' needs adjustment — check finger positions against the reference.",
            "Focus on '{sign}': spread your fingers more clearly for better detection.",
            "For '{sign}', make sure your hand is fully upright and well-lit.",
            "'{sign}' is tricky — pay attention to your thumb and index finger angle.",
        ],
        'tip': "Compare your hand shape with the reference image side-by-side.",
        'level': 'fair',
    },
    'hint': {
        'templates': [
            "Hint for '{sign}': try rotating your wrist slightly toward the camera.",
            "Small tip for '{sign}': ensure your palm faces the camera squarely.",
            "For '{sign}', a slight adjustment to finger spacing will help.",
            "Close! For '{sign}', try a slightly slower, more deliberate movement.",
        ],
        'tip': "Subtle changes in angle make a big difference for detection.",
        'level': 'fair',
    },
    'video_demo': {
        'templates': [
            "Let me show you '{sign}' — watch the reference carefully before trying again.",
            "Watch the demonstration for '{sign}' to see the exact hand shape needed.",
            "Here's a reference for '{sign}' — focus on the finger positions shown.",
            "Take a moment to study the '{sign}' guide before your next attempt.",
        ],
        'tip': "Mirror the reference exactly — pay attention to finger gaps and wrist angle.",
        'level': 'poor',
    },
    'skip_option': {
        'templates': [
            "'{sign}' is challenging right now — you can skip and return to it later.",
            "No worries! You can move on from '{sign}' and revisit it after more practice.",
            "Feel free to skip '{sign}' for now and come back when you're ready.",
            "'{sign}' is tough! Skip it for now — it'll be easier after more practice.",
        ],
        'tip': "Skipping is smart — return to difficult signs after mastering easier ones.",
        'level': 'incorrect',
    },
    'break_suggestion': {
        'templates': [
            "You've been working hard! A short break will help your memory consolidate.",
            "Great persistence! Take a 2-minute break and come back refreshed.",
            "Learning takes energy — a brief rest now will improve your next session.",
            "Your hands need a rest too! Step away for a moment then continue.",
        ],
        'tip': "Rest is part of learning — short breaks improve long-term retention.",
        'level': 'poor',
    },
}

# ══════════════════════════════════════════════════════════════
# Hard Constraint Map (mirrors simulator logic)
# ══════════════════════════════════════════════════════════════

def _get_valid_actions(consecutive_failures: int) -> List[str]:
    """
    Return the valid action set given current frustration level.
    Matches the simulator's proven constraint rules:
      ≥ 5 failures: only supportive/escape actions
      ≥ 3 failures: no encouraging (does more harm than good)
    """
    if consecutive_failures >= 5:
        return ['video_demo', 'skip_option', 'break_suggestion']
    if consecutive_failures >= 3:
        return [a for a in ACTIONS if a != 'encouraging']
    return ACTIONS


# ══════════════════════════════════════════════════════════════
# Context Classifier
# ══════════════════════════════════════════════════════════════

def _classify_context(
    consecutive_failures: int,
    confidence: float,
    is_correct: bool,
    attempt_count: int,
    success_rate_last5: float,
) -> str:
    """Map real-time user metrics to one of the 5 context buckets."""
    if consecutive_failures >= 3:
        return 'frustrated'
    if attempt_count <= 2:
        return 'learning'
    if confidence < 0.40 and not is_correct:
        return 'struggling'
    if success_rate_last5 > 0.70 and confidence >= 0.75:
        return 'confident'
    return 'normal'


# ══════════════════════════════════════════════════════════════
# Thompson Sampling Feedback Agent
# ══════════════════════════════════════════════════════════════

class ThompsonFeedbackAgent:
    """
    Contextual Thompson Sampling agent for adaptive sign-language feedback.

    Maintains Beta(α, β) distributions over 6 feedback actions × 5 contexts.
    = 30 independent Beta distributions that are updated online.

    Compatible drop-in replacement for RLFeedbackAgent (same public API):
        get_feedback(...)   → dict with 'feedback', 'feedback_level', 'tip',
                              'session_id', 'rl_action', 'rl_state', 'rl_epsilon'
        receive_reward(...) → dict with reward applied flag and stats
    """

    # Initial Beta parameters — weak uniform prior (α=β=1 = flat distribution)
    # Prior nudges: actions that the rule-based simulator found useful get a
    # slightly higher α to jump-start good behaviour before real data arrives.
    _INITIAL_ALPHA = {
        ctx: {
            'encouraging':       2.0 if ctx in ('normal', 'learning', 'confident') else 1.0,
            'corrective':        2.0 if ctx in ('normal', 'struggling') else 1.0,
            'hint':              2.0,
            'video_demo':        2.0 if ctx in ('frustrated', 'struggling') else 1.0,
            'skip_option':       2.0 if ctx == 'frustrated' else 1.0,
            'break_suggestion':  2.0 if ctx == 'frustrated' else 1.0,
        }
        for ctx in CONTEXTS
    }
    _INITIAL_BETA = {ctx: {a: 1.0 for a in ACTIONS} for ctx in CONTEXTS}

    def __init__(self, save_path: str = None):
        # Beta distribution parameters: context → action → (α, β)
        self.alpha: Dict[str, Dict[str, float]] = {
            ctx: dict(self._INITIAL_ALPHA[ctx]) for ctx in CONTEXTS
        }
        self.beta: Dict[str, Dict[str, float]] = {
            ctx: dict(self._INITIAL_BETA[ctx]) for ctx in CONTEXTS
        }

        # Session memory for delayed reward assignment
        # session_id → {context, action, confidence, is_correct, timestamp}
        self.session_memory: Dict[str, dict] = {}

        # Running stats
        self.total_updates = 0
        self.total_rewards = 0.0
        self.reward_history: List[float] = []

        # Persistence
        self.save_path = save_path or os.path.join(
            os.path.dirname(__file__), 'rl_thompson_params.json'
        )
        self._load()

    # ─────────────────────────────────────────────────────────
    # Core Thompson Sampling
    # ─────────────────────────────────────────────────────────

    def _sample_action(self, context: str, valid_actions: List[str]) -> str:
        """
        Draw θ ~ Beta(α, β) for each valid action and return argmax.
        This is pure Thompson Sampling — no extra exploration parameter needed.
        """
        samples = {
            action: float(np.random.beta(
                self.alpha[context][action],
                self.beta[context][action],
            ))
            for action in valid_actions
        }
        return max(samples, key=samples.get)

    def _update_params(self, context: str, action: str, reward: float):
        """
        Bayesian update:
          positive reward → increase α (action was good in this context)
          negative reward → increase β (action was bad in this context)
        Reward is normalised to keep α/β values from exploding.
        """
        weight = abs(reward) / 2.0   # scale: max reward ~2.0 → weight ~1.0
        weight = max(0.05, min(1.0, weight))   # clamp to [0.05, 1.0]

        if reward >= 0:
            self.alpha[context][action] += weight
        else:
            self.beta[context][action] += weight

        self.total_updates += 1
        self.total_rewards += reward
        self.reward_history.append(reward)
        if len(self.reward_history) > 500:
            self.reward_history.pop(0)

        if self.total_updates % 10 == 0:
            self._save()

    # ─────────────────────────────────────────────────────────
    # Public API  (drop-in for RLFeedbackAgent)
    # ─────────────────────────────────────────────────────────

    def get_feedback(
        self,
        confidence: float,
        is_correct: bool,
        hand_detected: bool,
        attempt_count: int,
        predicted_label: str,
        expected_label: str = None,
        session_id: str = None,
        # Extended context inputs (optional — degrade gracefully when absent)
        consecutive_failures: int = 0,
        success_rate_last5: float = 0.5,
    ) -> dict:
        """
        Select the best feedback action for the current user state and
        return a formatted response compatible with app.py.

        Returns:
            feedback         str   Human-readable feedback message + tip
            feedback_level   str   tier label for frontend styling
            tip              str   short improvement tip
            session_id       str   ID used to send the reward signal later
            rl_action        str   which action was selected
            rl_context       str   which context was detected
            rl_state         list  raw state vector [failures, conf, correct, attempts]
            rl_epsilon       float always 0.0 (TS needs no ε — here for API compat)
        """
        # 1. Classify context
        context = _classify_context(
            consecutive_failures=consecutive_failures,
            confidence=confidence,
            is_correct=is_correct,
            attempt_count=attempt_count,
            success_rate_last5=success_rate_last5,
        )

        # 2. Apply hard constraints
        valid_actions = _get_valid_actions(consecutive_failures)

        # 3. If hand not detected, force corrective guidance
        if not hand_detected:
            action = 'corrective'
        else:
            action = self._sample_action(context, valid_actions)

        # 4. Format message
        template_group = FEEDBACK_TEMPLATES[action]
        templates = template_group['templates']
        # Pick template deterministically within the action (round-robin by attempts)
        template = templates[attempt_count % len(templates)]

        message = template.format(
            sign=expected_label or predicted_label,
            expected=expected_label or '?',
            conf_pct=confidence * 100,
            streak=consecutive_failures,  # repurposed placeholder
        )

        # Append no-hand warning
        if not hand_detected:
            message = '✋ No hand detected. ' + message

        tip = template_group['tip']
        level = template_group['level']

        # Promote poor → fair when sign is actually correct (letter match)
        if is_correct and level in ('poor', 'incorrect'):
            level = 'good'

        # 5. Store session for delayed reward
        sid = session_id or f"ts_{int(time.time() * 1000)}"
        self.session_memory[sid] = {
            'context': context,
            'action': action,
            'confidence': confidence,
            'is_correct': is_correct,
            'consecutive_failures': consecutive_failures,
            'timestamp': time.time(),
        }
        self._cleanup_sessions()

        return {
            'feedback': f"{message}  💡 {tip}",
            'feedback_level': level,
            'tip': tip,
            'session_id': sid,
            'rl_action': action,
            'rl_context': context,
            'rl_state': [consecutive_failures, round(confidence, 3),
                         int(is_correct), attempt_count],
            'rl_epsilon': 0.0,   # TS needs no ε — kept for API compatibility
        }

    def receive_reward(
        self,
        session_id: str,
        reward_type: str,
        new_confidence: float = None,
        new_is_correct: bool = None,
    ) -> dict:
        """
        Process a delayed reward signal from the frontend.

        reward_type:
            'correct'    user matched the sign          → +2.0
            'improved'   confidence went up             → +1.0 (+ improvement bonus)
            'retry'      user still trying              → +0.3
            'no_change'  no measurable change           →  0.0
            'give_up'    user skipped / gave up         → -1.5
        """
        if session_id not in self.session_memory:
            return {'error': 'session_not_found', 'applied': False}

        mem = self.session_memory.pop(session_id)
        context = mem['context']
        action = mem['action']
        old_conf = mem['confidence']

        # Base reward
        reward_map = {
            'correct':   2.0,
            'improved':  1.0,
            'retry':     0.3,
            'no_change': 0.0,
            'give_up':  -1.5,
        }
        reward = reward_map.get(reward_type, 0.0)

        # Confidence-improvement bonus (mirrors simulator emotional reward)
        if new_confidence is not None and new_confidence > old_conf:
            reward += (new_confidence - old_conf) * 1.5

        # Extra penalty if user quit while we gave the wrong feedback
        # (same logic as FeedbackSystem.calculate_reward in the simulator)
        if reward_type == 'give_up' and mem['consecutive_failures'] < 3 and action in ('skip_option', 'break_suggestion'):
            reward -= 0.5   # we gave up-option too early

        # Update Thompson parameters
        self._update_params(context, action, reward)

        return {
            'applied': True,
            'reward': round(reward, 3),
            'context': context,
            'action': action,
            'rl_alpha': round(self.alpha[context][action], 4),
            'rl_beta': round(self.beta[context][action], 4),
            'total_updates': self.total_updates,
        }

    # ─────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return agent statistics — mirrors RLFeedbackAgent.get_stats() format."""
        return {
            'agent': 'ThompsonSampling',
            'total_updates': self.total_updates,
            'total_rewards': round(self.total_rewards, 2),
            'avg_reward_last_100': round(
                float(np.mean(self.reward_history[-100:])) if self.reward_history else 0.0,
                3,
            ),
            'epsilon': 0.0,
            'active_sessions': len(self.session_memory),
            'num_actions': len(ACTIONS),
            'num_contexts': len(CONTEXTS),
            # Best action per context (highest mean of Beta = α / (α+β))
            'policy_summary': {
                ctx: max(
                    ACTIONS,
                    key=lambda a: self.alpha[ctx][a] / (self.alpha[ctx][a] + self.beta[ctx][a]),
                )
                for ctx in CONTEXTS
            },
        }

    def get_context_policy(self) -> dict:
        """
        Return the current learned policy: for each context, show the
        probability each action would be chosen (mean of Beta posterior).
        Useful for research/debugging.
        """
        policy = {}
        for ctx in CONTEXTS:
            probs = {}
            for action in ACTIONS:
                a = self.alpha[ctx][action]
                b = self.beta[ctx][action]
                probs[action] = round(a / (a + b), 4)   # Beta mean
            # Sort descending
            policy[ctx] = dict(sorted(probs.items(), key=lambda x: -x[1]))
        return policy

    # ─────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────

    def _save(self):
        data = {
            'agent': 'ThompsonSampling',
            'alpha': self.alpha,
            'beta': self.beta,
            'total_updates': self.total_updates,
            'total_rewards': self.total_rewards,
            'reward_history': self.reward_history[-500:],
        }
        try:
            with open(self.save_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️  ThompsonFeedbackAgent save failed: {e}")

    def _load(self):
        if not os.path.exists(self.save_path):
            print('🆕 ThompsonFeedbackAgent: starting fresh (no saved params)')
            return
        try:
            with open(self.save_path, 'r') as f:
                data = json.load(f)
            # Restore, filling missing keys with defaults
            for ctx in CONTEXTS:
                for action in ACTIONS:
                    self.alpha[ctx][action] = data.get('alpha', {}).get(ctx, {}).get(
                        action, self._INITIAL_ALPHA[ctx][action]
                    )
                    self.beta[ctx][action] = data.get('beta', {}).get(ctx, {}).get(
                        action, self._INITIAL_BETA[ctx][action]
                    )
            self.total_updates = data.get('total_updates', 0)
            self.total_rewards = data.get('total_rewards', 0.0)
            self.reward_history = data.get('reward_history', [])
            print(f'📂 ThompsonFeedbackAgent loaded '
                  f'({self.total_updates} updates)')
        except Exception as e:
            print(f'⚠️  ThompsonFeedbackAgent load failed: {e}')

    def save(self):
        """Force-save (public method)."""
        self._save()

    def _cleanup_sessions(self, max_age: float = 1800):
        """Remove stale sessions older than max_age seconds (default 30 min)."""
        now = time.time()
        expired = [
            sid for sid, m in self.session_memory.items()
            if now - m['timestamp'] > max_age
        ]
        for sid in expired:
            del self.session_memory[sid]
