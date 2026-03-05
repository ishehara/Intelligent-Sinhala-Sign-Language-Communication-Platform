"""
Reinforcement Learning Feedback Agent
======================================
A Q-learning agent that learns which feedback messages help users
improve their sign language gestures the most.

State:  (confidence_bin, is_correct, hand_detected, attempt_bin)
Action: index into a feedback action table (level + tip)
Reward: comes from user interaction (improvement, success, give-up)

The agent is trained online — it learns from every user interaction
and persists its Q-table to disk so learning carries across restarts.
"""

import json
import os
import time
import numpy as np
from collections import defaultdict

# ══════════════════════════════════════════════════════════════
# Feedback Action Table
# ══════════════════════════════════════════════════════════════
# Each action is (feedback_level, message_template, tip)
# The RL agent picks which action to use given the current state.

FEEDBACK_ACTIONS = [
    # ── Excellent tier ────────────────────────────────────────
    {
        "level": "excellent",
        "template": "Excellent! Perfect sign — '{label}' recognized at {pct:.1f}% confidence!",
        "tip": "You're doing great. Keep this form!",
    },
    {
        "level": "excellent",
        "template": "Outstanding! Your '{label}' is spot on ({pct:.1f}%)!",
        "tip": "Try increasing your speed while keeping accuracy.",
    },

    # ── Good tier ─────────────────────────────────────────────
    {
        "level": "good",
        "template": "Good job! '{label}' recognized with {pct:.1f}% confidence.",
        "tip": "Hold the pose a bit steadier for even higher accuracy.",
    },
    {
        "level": "good",
        "template": "Nice work! '{label}' detected at {pct:.1f}%.",
        "tip": "Try adjusting your wrist angle slightly.",
    },
    {
        "level": "good",
        "template": "Well done! '{label}' matched at {pct:.1f}% confidence.",
        "tip": "Ensure all fingers are clearly separated.",
    },

    # ── Fair tier ─────────────────────────────────────────────
    {
        "level": "fair",
        "template": "Almost there! '{label}' detected at {pct:.1f}%.",
        "tip": "Adjust your finger position and angle for a clearer gesture.",
    },
    {
        "level": "fair",
        "template": "Getting close! '{label}' recognized at {pct:.1f}%.",
        "tip": "Move your hand closer to the camera.",
    },
    {
        "level": "fair",
        "template": "On the right track — '{label}' at {pct:.1f}%.",
        "tip": "Try better lighting so the camera can see your hand clearly.",
    },

    # ── Poor tier ─────────────────────────────────────────────
    {
        "level": "poor",
        "template": "Detected '{label}' but confidence is low ({pct:.1f}%).",
        "tip": "Make sure your hand is well-lit and clearly visible.",
    },
    {
        "level": "poor",
        "template": "Hmm, '{label}' detected with only {pct:.1f}% confidence.",
        "tip": "Check your hand shape against the reference image.",
    },
    {
        "level": "poor",
        "template": "Low confidence ({pct:.1f}%) for '{label}'.",
        "tip": "Try a plain background so the camera focuses on your hand.",
    },

    # ── Incorrect tier ────────────────────────────────────────
    {
        "level": "incorrect",
        "template": "Not quite — you signed '{label}' but expected '{expected}'.",
        "tip": "Look at the reference and focus on finger positions.",
    },
    {
        "level": "incorrect",
        "template": "Oops! That looks like '{label}', not '{expected}'.",
        "tip": "Pay attention to thumb placement — it's often the key difference.",
    },
    {
        "level": "incorrect",
        "template": "Close! Detected '{label}' instead of '{expected}'.",
        "tip": "Try exaggerating the gesture a bit more.",
    },
]

NUM_ACTIONS = len(FEEDBACK_ACTIONS)

# ══════════════════════════════════════════════════════════════
# State discretisation helpers
# ══════════════════════════════════════════════════════════════

def _bin_confidence(conf: float) -> int:
    """Bin confidence [0,1] into 5 buckets."""
    if conf >= 0.90:
        return 4  # very high
    elif conf >= 0.75:
        return 3  # high
    elif conf >= 0.50:
        return 2  # medium
    elif conf >= 0.25:
        return 1  # low
    return 0       # very low


def _bin_attempts(attempts: int) -> int:
    """Bin attempt count into 4 buckets."""
    if attempts <= 1:
        return 0
    elif attempts <= 3:
        return 1
    elif attempts <= 7:
        return 2
    return 3  # many attempts


def make_state(confidence: float, is_correct: bool,
               hand_detected: bool, attempt_count: int) -> tuple:
    """Build a discrete state tuple for the Q-table."""
    return (
        _bin_confidence(confidence),
        int(is_correct),
        int(hand_detected),
        _bin_attempts(attempt_count),
    )


# ══════════════════════════════════════════════════════════════
# RL Feedback Agent (Q-learning)
# ══════════════════════════════════════════════════════════════

class RLFeedbackAgent:
    """
    Tabular Q-learning agent for adaptive feedback selection.

    - State space: ~5×2×2×4 = 80 states
    - Action space: len(FEEDBACK_ACTIONS) ≈ 14 actions
    - Uses ε-greedy exploration
    - Persists Q-table to JSON
    """

    def __init__(self, save_path: str = None,
                 alpha: float = 0.1,
                 gamma: float = 0.95,
                 epsilon: float = 0.2,
                 epsilon_decay: float = 0.9995,
                 epsilon_min: float = 0.05):
        self.alpha = alpha          # learning rate
        self.gamma = gamma          # discount factor
        self.epsilon = epsilon      # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_actions = NUM_ACTIONS

        # Q-table: state_str -> [Q-values for each action]
        self.q_table = defaultdict(lambda: np.zeros(self.num_actions))

        # Session memory: session_id -> {state, action, timestamp}
        # Used to assign delayed rewards
        self.session_memory = {}

        # Statistics
        self.total_episodes = 0
        self.total_rewards = 0.0
        self.reward_history = []  # last 100 rewards

        # Persistence
        self.save_path = save_path or os.path.join(
            os.path.dirname(__file__), 'rl_q_table.json'
        )
        self._load()

    # ── Core RL methods ───────────────────────────────────────

    def select_action(self, state: tuple) -> int:
        """ε-greedy action selection."""
        state_key = str(state)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        return int(np.argmax(self.q_table[state_key]))

    def update(self, state: tuple, action: int, reward: float,
               next_state: tuple = None):
        """Q-learning update rule."""
        s = str(state)
        if next_state is not None:
            ns = str(next_state)
            best_next = float(np.max(self.q_table[ns]))
        else:
            best_next = 0.0

        old_q = self.q_table[s][action]
        self.q_table[s][action] = old_q + self.alpha * (
            reward + self.gamma * best_next - old_q
        )

        # Decay exploration
        self.epsilon = max(self.epsilon_min,
                          self.epsilon * self.epsilon_decay)

        # Track stats
        self.total_episodes += 1
        self.total_rewards += reward
        self.reward_history.append(reward)
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)

        # Auto-save every 10 episodes
        if self.total_episodes % 10 == 0:
            self._save()

    # ── High-level API ────────────────────────────────────────

    def get_feedback(self, confidence: float, is_correct: bool,
                     hand_detected: bool, attempt_count: int,
                     predicted_label: str, expected_label: str = None,
                     session_id: str = None) -> dict:
        """
        Pick a feedback action and return the formatted feedback.
        Stores the (state, action) in session memory for delayed reward.
        """
        state = make_state(confidence, is_correct, hand_detected, attempt_count)
        action_idx = self.select_action(state)
        action = FEEDBACK_ACTIONS[action_idx]

        pct = confidence * 100
        message = action["template"].format(
            label=predicted_label,
            expected=expected_label or "?",
            pct=pct,
        )
        tip = action["tip"]

        # Store for delayed reward
        sid = session_id or f"s_{int(time.time()*1000)}"
        self.session_memory[sid] = {
            "state": state,
            "action": action_idx,
            "confidence": confidence,
            "is_correct": is_correct,
            "timestamp": time.time(),
        }
        # Cleanup old sessions (>30 min)
        self._cleanup_sessions()

        return {
            "feedback": f"{message} 💡 Tip: {tip}",
            "feedback_level": action["level"],
            "tip": tip,
            "session_id": sid,
            "rl_action": action_idx,
            "rl_state": list(state),
            "rl_epsilon": round(self.epsilon, 4),
        }

    def receive_reward(self, session_id: str, reward_type: str,
                       new_confidence: float = None,
                       new_is_correct: bool = None) -> dict:
        """
        Process a reward signal from the frontend.

        reward_type:
          'improved'     — user's confidence went up (+1.0)
          'correct'      — user got the sign right (+2.0)
          'retry'        — user is trying again (+0.2 for persistence)
          'give_up'      — user skipped/gave up (-1.0)
          'no_change'    — no improvement (0.0)
        """
        if session_id not in self.session_memory:
            return {"error": "Session not found", "applied": False}

        mem = self.session_memory.pop(session_id)
        old_state = mem["state"]
        action = mem["action"]
        old_conf = mem["confidence"]

        # Calculate reward
        reward_map = {
            "improved": 1.0,
            "correct": 2.0,
            "retry": 0.2,
            "give_up": -1.0,
            "no_change": 0.0,
        }
        base_reward = reward_map.get(reward_type, 0.0)

        # Bonus for measurable improvement
        if new_confidence is not None and new_confidence > old_conf:
            improvement_bonus = (new_confidence - old_conf) * 2.0
            base_reward += improvement_bonus

        # Build next state (if available)
        next_state = None
        if new_confidence is not None and new_is_correct is not None:
            next_state = make_state(
                new_confidence, new_is_correct,
                True,  # assume hand detected if they're sending reward
                mem.get("attempt_count", 1) + 1,
            )

        # Update Q-table
        self.update(old_state, action, base_reward, next_state)

        return {
            "applied": True,
            "reward": round(base_reward, 3),
            "old_state": list(old_state),
            "action": action,
            "epsilon": round(self.epsilon, 4),
            "total_episodes": self.total_episodes,
        }

    # ── Stats ─────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return agent learning statistics."""
        return {
            "total_episodes": self.total_episodes,
            "total_rewards": round(self.total_rewards, 2),
            "avg_reward_last_100": round(
                np.mean(self.reward_history) if self.reward_history else 0, 3
            ),
            "epsilon": round(self.epsilon, 4),
            "q_table_size": len(self.q_table),
            "active_sessions": len(self.session_memory),
            "num_actions": self.num_actions,
        }

    # ── Persistence ───────────────────────────────────────────

    def _save(self):
        """Save Q-table and stats to JSON file."""
        data = {
            "q_table": {k: v.tolist() for k, v in self.q_table.items()},
            "epsilon": self.epsilon,
            "total_episodes": self.total_episodes,
            "total_rewards": self.total_rewards,
            "reward_history": self.reward_history,
        }
        try:
            with open(self.save_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"💾 RL agent saved ({self.total_episodes} episodes)")
        except Exception as e:
            print(f"⚠️ Failed to save RL agent: {e}")

    def _load(self):
        """Load Q-table and stats from JSON file."""
        if not os.path.exists(self.save_path):
            print("🆕 RL agent: starting fresh (no saved Q-table)")
            return
        try:
            with open(self.save_path, 'r') as f:
                data = json.load(f)
            for k, v in data.get("q_table", {}).items():
                self.q_table[k] = np.array(v)
            self.epsilon = data.get("epsilon", self.epsilon)
            self.total_episodes = data.get("total_episodes", 0)
            self.total_rewards = data.get("total_rewards", 0.0)
            self.reward_history = data.get("reward_history", [])
            print(f"📂 RL agent loaded ({self.total_episodes} episodes, "
                  f"ε={self.epsilon:.4f})")
        except Exception as e:
            print(f"⚠️ Failed to load RL agent: {e}")

    def _cleanup_sessions(self, max_age: float = 1800):
        """Remove sessions older than max_age seconds."""
        now = time.time()
        expired = [
            sid for sid, m in self.session_memory.items()
            if now - m["timestamp"] > max_age
        ]
        for sid in expired:
            del self.session_memory[sid]

    def save(self):
        """Public save method."""
        self._save()
