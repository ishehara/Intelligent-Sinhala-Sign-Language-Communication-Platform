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

# Sign-specific corrective messages ───────────────────────────
# Shown when the user's sign is detected as WRONG. Each message
# gives the precise hand-shape correction needed for that letter.

SIGN_CORRECTIONS: Dict[str, str] = {
    'අ': "The palm should be with fingers together, not separated.",
    'ආ': "The thumb and other four fingers should be slightly closer.",
    'ඇ': "The index finger should be slightly bent.",
    'ඉ': "The palm should bend slightly like a cap shape.",
    'උ': "The fingers and hand should form a 90-degree position.",
    'එ': "Slightly rotate the wrist.",
    'ක': "The thumb should be positioned between the index and middle finger.",
    'ග': "The index finger should bend to approximately 90 degrees.",
    'ට': "The index finger should bend more.",
    'ද': "The hand should rotate to 90 degrees.",
    'ත': "The hand should rotate to the opposite side.",
    'ඩ': "The palm should rotate to the opposite side.",
    'න': "The palm should rotate slightly to the other side.",
    'ප': "The index finger should point downward, not upward.",
    'බ': "The palm should rotate to the other side.",
    'ම': "The palm should rotate to the other side.",
    'ය': "The middle three fingers should bend more.",
    'ර': "The index and middle fingers should stay together, not separated.",
    'ල': "The thumb and middle finger should not bend.",
    'ව': "The three fingers are positioned incorrectly.",
    'ස': "The palm should rotate to the other side.",
    'හ': "The hand position should rotate upside down.",
    'ං': "All five fingers should bend together, not just two.",
    'ච': "This is not a thumbs-up position.",
    'ෆ': "Three fingers should bend slightly.",
}

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
# Soft Constraint Mask  (replaces hard-blocking)
# ══════════════════════════════════════════════════════════════

def _compute_action_mask(consecutive_failures: int) -> Dict[str, float]:
    """
    Soft-constraint mask: instead of hard-blocking actions, each action
    gets a multiplicative weight on its Thompson sample.

    Weight = 1.0 → no penalty.
    Weight → 0   → strongly suppressed, BUT the learned Beta posterior
                    can still override if it has enough positive evidence.

    This lets the agent DISCOVER constraints from reward signals while
    still starting with sensible safety priors from simulation data.
    """
    mask = {a: 1.0 for a in ACTIONS}
    if consecutive_failures >= 5:
        mask['encouraging'] = 0.05
        mask['corrective'] = 0.20
        mask['hint'] = 0.20
    elif consecutive_failures >= 3:
        mask['encouraging'] = 0.15
    return mask


# ══════════════════════════════════════════════════════════════
# Adaptive Context Thresholds
# ══════════════════════════════════════════════════════════════

class AdaptiveContextThresholds:
    """
    Context classification with thresholds that adapt from reward signals.

    Instead of fixed rules (failures≥3 → frustrated, conf<0.40 → struggling),
    these boundaries shift when a classification consistently leads to poor
    rewards (wrong context → bad feedback → user gives up → negative reward).
    """

    def __init__(self):
        self.frustration_failures: float = 3.0
        self.struggling_conf: float = 0.40
        self.learning_max_attempts: float = 2.0
        self.confident_success_rate: float = 0.70
        self.confident_conf: float = 0.75

        self._lr = 0.01  # learning rate for threshold nudges
        self._context_rewards: Dict[str, List[float]] = {c: [] for c in CONTEXTS}

    def classify(self, consecutive_failures: int, confidence: float,
                 is_correct: bool, attempt_count: int,
                 success_rate_last5: float) -> str:
        """Classify user state using adaptive thresholds."""
        if consecutive_failures >= self.frustration_failures:
            return 'frustrated'
        if attempt_count <= self.learning_max_attempts:
            return 'learning'
        if confidence < self.struggling_conf and not is_correct:
            return 'struggling'
        if (success_rate_last5 > self.confident_success_rate
                and confidence >= self.confident_conf):
            return 'confident'
        return 'normal'

    def update(self, context: str, reward: float):
        """Nudge thresholds based on reward — gradient-free boundary adaptation."""
        self._context_rewards[context].append(reward)
        if len(self._context_rewards[context]) > 200:
            self._context_rewards[context] = self._context_rewards[context][-200:]

        recent = self._context_rewards[context][-20:]
        if len(recent) < 5:
            return

        avg = float(np.mean(recent))

        # Poor reward → relax this context (fewer users classified here)
        if avg < -0.2:
            if context == 'frustrated':
                self.frustration_failures = min(5.0, self.frustration_failures + self._lr * 3)
            elif context == 'struggling':
                self.struggling_conf = max(0.20, self.struggling_conf - self._lr * 2)
            elif context == 'confident':
                self.confident_success_rate = min(0.90, self.confident_success_rate + self._lr)
        # Good reward → widen this context (capture more users)
        elif avg > 0.5:
            if context == 'frustrated':
                self.frustration_failures = max(2.0, self.frustration_failures - self._lr)
            elif context == 'struggling':
                self.struggling_conf = min(0.55, self.struggling_conf + self._lr)
            elif context == 'confident':
                self.confident_success_rate = max(0.55, self.confident_success_rate - self._lr)

    def to_dict(self) -> dict:
        return {
            'frustration_failures': round(self.frustration_failures, 3),
            'struggling_conf': round(self.struggling_conf, 3),
            'learning_max_attempts': round(self.learning_max_attempts, 3),
            'confident_success_rate': round(self.confident_success_rate, 3),
            'confident_conf': round(self.confident_conf, 3),
            'context_avg_rewards': {
                c: round(float(np.mean(rs[-20:])), 3) if rs else 0.0
                for c, rs in self._context_rewards.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'AdaptiveContextThresholds':
        t = cls()
        t.frustration_failures = d.get('frustration_failures', 3.0)
        t.struggling_conf = d.get('struggling_conf', 0.40)
        t.learning_max_attempts = d.get('learning_max_attempts', 2.0)
        t.confident_success_rate = d.get('confident_success_rate', 0.70)
        t.confident_conf = d.get('confident_conf', 0.75)
        return t


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

        # Adaptive context thresholds (learned from reward signals)
        self.context_thresholds = AdaptiveContextThresholds()

        # Per-context per-action reward tracking
        self.context_action_rewards: Dict[str, Dict[str, List[float]]] = {
            ctx: {a: [] for a in ACTIONS} for ctx in CONTEXTS
        }

        # Policy evolution history (snapshots every 25 updates)
        self.policy_history: List[dict] = []

        # Persistence
        self.save_path = save_path or os.path.join(
            os.path.dirname(__file__), 'rl_thompson_params.json'
        )
        self._load()

    # ─────────────────────────────────────────────────────────
    # Core Thompson Sampling
    # ─────────────────────────────────────────────────────────

    def _sample_action(self, context: str,
                       action_mask: Dict[str, float] = None) -> str:
        """
        Draw θ ~ Beta(α, β) for EVERY action, apply soft-constraint mask,
        and return argmax.

        No actions are ever hard-blocked — the mask only suppresses.
        If the Beta posterior is strong enough (high α from positive rewards),
        the agent CAN override the mask, i.e. it learns its own constraints.
        """
        samples = {}
        for action in ACTIONS:
            theta = float(np.random.beta(
                self.alpha[context][action],
                self.beta[context][action],
            ))
            if action_mask:
                theta *= action_mask.get(action, 1.0)
            samples[action] = theta
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

        # Track per-context per-action rewards
        self.context_action_rewards[context][action].append(reward)
        if len(self.context_action_rewards[context][action]) > 200:
            self.context_action_rewards[context][action] = \
                self.context_action_rewards[context][action][-200:]

        # Adapt context thresholds based on reward signal
        self.context_thresholds.update(context, reward)

        # Record policy snapshot periodically
        self._record_policy_snapshot()

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
        # 1. Classify context using adaptive thresholds
        context = self.context_thresholds.classify(
            consecutive_failures=consecutive_failures,
            confidence=confidence,
            is_correct=is_correct,
            attempt_count=attempt_count,
            success_rate_last5=success_rate_last5,
        )

        # 2. Compute soft-constraint mask (no hard blocking)
        action_mask = _compute_action_mask(consecutive_failures)

        # 3. If hand not detected, force corrective guidance
        if not hand_detected:
            action = 'corrective'
        else:
            action = self._sample_action(context, action_mask)

        # 4. Format message
        sign_label = expected_label or predicted_label

        # If the sign is WRONG and we have a sign-specific correction, use it
        if not is_correct and sign_label in SIGN_CORRECTIONS:
            sign_correction = SIGN_CORRECTIONS[sign_label]
            message = f"'{sign_label}' is not quite right — {sign_correction}"
            tip = "Compare your hand shape with the reference image side-by-side."
            level = 'fair'
        else:
            template_group = FEEDBACK_TEMPLATES[action]
            templates = template_group['templates']
            # Pick template deterministically within the action (round-robin by attempts)
            template = templates[attempt_count % len(templates)]

            message = template.format(
                sign=sign_label,
                expected=expected_label or '?',
                conf_pct=confidence * 100,
                streak=consecutive_failures,  # repurposed placeholder
            )
            tip = template_group['tip']
            level = template_group['level']

        # Append no-hand warning
        if not hand_detected:
            message = '✋ No hand detected. ' + message

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
            'context_thresholds': self.context_thresholds.to_dict(),
            'policy_snapshots': len(self.policy_history),
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
    # Policy Evolution Tracking
    # ─────────────────────────────────────────────────────────

    def _record_policy_snapshot(self):
        """Snapshot current policy every 25 updates for evolution tracking."""
        if self.total_updates % 25 != 0:
            return
        snapshot = {
            'update': self.total_updates,
            'timestamp': time.time(),
            'avg_reward_last50': round(
                float(np.mean(self.reward_history[-50:]))
                if self.reward_history else 0, 4
            ),
            'thresholds': self.context_thresholds.to_dict(),
            'policy': {},
        }
        for ctx in CONTEXTS:
            means = {}
            for a in ACTIONS:
                al = self.alpha[ctx][a]
                be = self.beta[ctx][a]
                means[a] = round(al / (al + be), 4)
            best = max(means, key=means.get)
            snapshot['policy'][ctx] = {
                'best_action': best,
                'action_probs': means,
            }
        self.policy_history.append(snapshot)
        if len(self.policy_history) > 100:
            self.policy_history = self.policy_history[-100:]

    def get_policy_evolution(self) -> dict:
        """
        Return full policy evolution data for research visualization.
        Shows: how Beta distributions, context thresholds, and preferred
        actions have changed over the agent's lifetime.
        """
        return {
            'total_updates': self.total_updates,
            'total_snapshots': len(self.policy_history),
            'current_thresholds': self.context_thresholds.to_dict(),
            'current_policy': self.get_context_policy(),
            'evolution': self.policy_history,
            'context_action_stats': {
                ctx: {
                    a: {
                        'alpha': round(self.alpha[ctx][a], 2),
                        'beta': round(self.beta[ctx][a], 2),
                        'mean': round(
                            self.alpha[ctx][a]
                            / (self.alpha[ctx][a] + self.beta[ctx][a]), 4
                        ),
                        'samples': len(
                            self.context_action_rewards.get(ctx, {}).get(a, [])
                        ),
                        'avg_reward': round(
                            float(np.mean(
                                self.context_action_rewards[ctx][a][-50:]
                            )), 3
                        ) if self.context_action_rewards.get(
                            ctx, {}
                        ).get(a, []) else 0.0,
                    }
                    for a in ACTIONS
                }
                for ctx in CONTEXTS
            },
        }

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
            'policy_history': self.policy_history[-100:],
            'context_thresholds': self.context_thresholds.to_dict(),
            'context_action_rewards': {
                ctx: {a: rs[-200:] for a, rs in acts.items()}
                for ctx, acts in self.context_action_rewards.items()
            },
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
            # Restore Beta parameters, filling missing keys with defaults
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

            # Restore adaptive context thresholds
            if 'context_thresholds' in data:
                self.context_thresholds = AdaptiveContextThresholds.from_dict(
                    data['context_thresholds']
                )

            # Restore policy evolution history
            self.policy_history = data.get('policy_history', [])

            # Restore per-context-action rewards
            car_data = data.get('context_action_rewards', {})
            for ctx in CONTEXTS:
                for a in ACTIONS:
                    self.context_action_rewards[ctx][a] = \
                        car_data.get(ctx, {}).get(a, [])

            print(f'📂 ThompsonFeedbackAgent loaded '
                  f'({self.total_updates} updates, '
                  f'{len(self.policy_history)} policy snapshots)')
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
