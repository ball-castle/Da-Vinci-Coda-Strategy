from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, replace
from math import ceil, exp, log2, sqrt
from pathlib import Path
from random import Random
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.core.state import (
    CARD_COLORS,
    JOKER,
    MAX_CARD_VALUE,
    Card,
    CardSlot,
    GameState,
    GuessAction,
    PlayerState,
)

ProbabilityMatrix = Dict[int, Dict[Card, float]]
FullProbabilityMatrix = Dict[str, ProbabilityMatrix]
SlotKey = Tuple[str, int]

CARD_COLOR_RANK = {"B": 0, "W": 1}
SOFT_BEHAVIOR_BLEND = 0.28


@dataclass(frozen=True)
class HiddenPosition:
    player_id: str
    slot_index: int
    order_index: int
    color: Optional[str]


@dataclass
class HardConstraintSet:
    fixed_by_slot: Dict[SlotKey, Card]
    forbidden_by_slot: Dict[SlotKey, Set[Card]]


@dataclass(frozen=True)
class GuessSignal:
    action_index: int
    guesser_id: str
    target_player_id: str
    target_slot_index: int
    guessed_card: Card
    result: bool
    continued_turn: Optional[bool]


@dataclass
class SearchTreeNode:
    label: str
    prior: float
    rollout_value: float
    move: Optional[Dict[str, Any]] = None
    is_terminal: bool = False
    visits: float = 0.0
    value_sum: float = 0.0
    peak_value: float = 0.0
    positive_value_count: float = 0.0
    children: Optional[List["SearchTreeNode"]] = None
    success_matrix: Optional[FullProbabilityMatrix] = None
    my_hidden_count: int = 0
    hidden_index_by_player: Optional[Dict[str, Dict[int, int]]] = None
    behavior_model: Optional["BehavioralLikelihoodModel"] = None
    guess_signals_by_player: Optional[Dict[str, Sequence[GuessSignal]]] = None
    acting_player_id: Optional[str] = None
    behavior_guidance_profile: Optional[Dict[str, float]] = None
    game_state: Optional[GameState] = None
    behavior_map_hypothesis: Optional[Dict[str, Dict[int, Card]]] = None
    rollout_depth_remaining: int = 0
    perspective_sign: float = 1.0
    search_mode: str = "mcts"


def numeric_card_value(card: Optional[Card]) -> Optional[int]:
    if card is None:
        return None
    value = card[1]
    if value == JOKER or not isinstance(value, int):
        return None
    return int(value)


def card_sort_key(card: Card) -> Tuple[int, int]:
    color, value = card
    if value == JOKER:
        return (MAX_CARD_VALUE + 1, CARD_COLOR_RANK[color])
    return (int(value), CARD_COLOR_RANK[color])


def serialize_card(card: Card) -> List[Any]:
    return [card[0], card[1]]


def normalize_card_distribution(weights: Dict[Card, float]) -> Dict[Card, float]:
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {card: weight / total for card, weight in weights.items()}


def slot_key(player_id: str, slot_index: int) -> SlotKey:
    return (player_id, slot_index)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class HardConstraintCompiler:
    """Compile hard public evidence into fixed/forbidden slot constraints."""

    def __init__(self, game_state: GameState):
        self.game_state = game_state

    def compile(self) -> HardConstraintSet:
        fixed_by_slot: Dict[SlotKey, Card] = {}
        forbidden_by_slot: Dict[SlotKey, Set[Card]] = defaultdict(set)

        for player_id in self.game_state.inference_player_ids():
            for slot in self.game_state.resolved_ordered_slots(player_id):
                known = slot.known_card()
                if known is not None:
                    self._fix_card(
                        fixed_by_slot,
                        forbidden_by_slot,
                        slot_key(player_id, slot.slot_index),
                        known,
                    )

        for action in getattr(self.game_state, "actions", ()):
            if getattr(action, "action_type", None) != "guess":
                continue

            guessed_card = action.guessed_card()
            target_player_id = getattr(action, "target_player_id", None)
            target_slot_index = getattr(action, "target_slot_index", None)
            if guessed_card is not None and target_player_id is not None and target_slot_index is not None:
                key = slot_key(target_player_id, target_slot_index)
                if getattr(action, "result", False):
                    self._fix_card(fixed_by_slot, forbidden_by_slot, key, guessed_card)
                else:
                    forbidden_by_slot[key].add(guessed_card)

            revealed_card = self._revealed_card_from_action(action)
            revealed_player_id = getattr(action, "revealed_player_id", None)
            revealed_slot_index = getattr(action, "revealed_slot_index", None)
            if revealed_card is not None and revealed_player_id is not None and revealed_slot_index is not None:
                self._fix_card(
                    fixed_by_slot,
                    forbidden_by_slot,
                    slot_key(revealed_player_id, revealed_slot_index),
                    revealed_card,
                )

        return HardConstraintSet(
            fixed_by_slot=fixed_by_slot,
            forbidden_by_slot={key: set(value) for key, value in forbidden_by_slot.items()},
        )

    def _fix_card(
        self,
        fixed_by_slot: Dict[SlotKey, Card],
        forbidden_by_slot: Dict[SlotKey, Set[Card]],
        key: SlotKey,
        card: Card,
    ) -> None:
        current = fixed_by_slot.get(key)
        if current is not None and current != card:
            raise ValueError(
                f"Conflicting hard evidence for slot {key}: {current} vs {card}"
            )
        if card in forbidden_by_slot.get(key, set()):
            raise ValueError(
                f"Slot {key} cannot be both fixed to and forbidden from {card}"
            )
        fixed_by_slot[key] = card

    def _revealed_card_from_action(self, action: Any) -> Optional[Card]:
        if hasattr(action, "revealed_card"):
            revealed = action.revealed_card()
            if revealed is not None:
                return revealed

        color = getattr(action, "revealed_color", None)
        value = getattr(action, "revealed_value", None)
        if color is None or value is None:
            return None
        return (color, value)


class BehavioralLikelihoodModel:
    """Structured soft action likelihood model, separate from hard constraints."""

    # Theory of Mind (ToM) / Advanced Mind-Reading Parameters
    # Extremely strong penalty for a player guessing a card they actually hold.
    # We leave 5% (0.05) to account for high-level bluffing behavior.
    SELF_EXACT_GUESS_PENALTY = 0.05
    SELF_ADJACENT_ANCHOR_BONUS = 1.15

    TARGET_PLAYER_BEST_MATCH_BONUS = 1.07
    TARGET_PLAYER_CLOSE_MATCH_BONUS = 1.03
    TARGET_PLAYER_WEAK_CHOICE_PENALTY = 0.94
    TARGET_PLAYER_REPEAT_FOCUS_BONUS = 1.04
    TARGET_PLAYER_CONFIDENT_CHAIN_BONUS = 1.05
    TARGET_PLAYER_BREAK_CONFIDENT_CHAIN_PENALTY = 0.95
    TARGET_PLAYER_SWITCH_AFTER_FAILURE_BONUS = 1.04
    TARGET_PLAYER_SWITCH_FAILURE_CONTINUITY_BONUS = 1.03
    TARGET_PLAYER_RECENT_COLLAPSE_BONUS = 1.04
    TARGET_PLAYER_COLLAPSE_STREAK_BONUS = 1.05
    TARGET_PLAYER_PUBLIC_BRIDGE_BONUS = 1.04
    TARGET_PLAYER_FINISH_CHAIN_BONUS = 1.08
    TARGET_PLAYER_RECENT_TARGET_CHAIN_BONUS = 1.06
    PLAYER_FINISH_PRESSURE_REFERENCE = 3.0
    PLAYER_RECENT_PUBLIC_REVEAL_BONUS = 0.08
    PLAYER_RECENT_FAILED_GUESS_BONUS = 0.10
    PLAYER_ATTACK_WINDOW_BONUS = 0.14
    PLAYER_GLOBAL_COLLAPSE_BONUS = 0.10

    TARGET_SLOT_BEST_MATCH_BONUS = 1.08
    TARGET_SLOT_CLOSE_MATCH_BONUS = 1.03
    TARGET_SLOT_WEAK_CHOICE_PENALTY = 0.91
    TARGET_SLOT_RETRY_AFTER_FAILURE_BONUS = 1.125
    TARGET_SLOT_CONFIDENT_ADJACENT_FOLLOW_BONUS = 1.04
    TARGET_SLOT_FAILURE_ADJACENT_PROBE_BONUS = 1.03
    TARGET_SLOT_ATTACK_WINDOW_BONUS = 1.05
    SLOT_EDGE_PRESSURE_BONUS = 1.05
    SLOT_RECENT_REVEAL_NEIGHBOR_BONUS = 1.06
    SLOT_FAILURE_NEIGHBOR_ATTACK_BONUS = 1.04
    SLOT_ATTACK_WINDOW_BONUS = 1.08
    PLAYER_SECONDARY_ATTACKABILITY_BLEND = 0.20
    PLAYER_FINISH_PRESSURE_BONUS = 0.26
    MATRIX_SECONDARY_ATTACKABILITY_BLEND = 0.22

    TARGET_IN_INTERVAL_BONUS = 1.15
    TARGET_NARROW_INTERVAL_BONUS = 1.10
    TARGET_OUTSIDE_INTERVAL_PENALTY = 0.80
    TARGET_NEIGHBOR_BONUS = 1.10
    TARGET_CLOSE_BONUS = 1.04
    TARGET_FAR_PENALTY = 0.97
    TARGET_VALUE_PROGRESSIVE_STEP_BONUS = 1.08
    TARGET_VALUE_DIRECTIONAL_BONUS = 1.03
    TARGET_VALUE_STALLED_PENALTY = 0.90
    TARGET_VALUE_WRONG_DIRECTION_PENALTY = 0.92
    TARGET_VALUE_CONFIDENT_CHAIN_LOCAL_STEP_BONUS = 1.04
    TARGET_VALUE_CONFIDENT_CHAIN_JUMP_PENALTY = 0.95
    TARGET_VALUE_SANDWICH_EXACT_BONUS = 1.12
    TARGET_VALUE_SANDWICH_FILL_BONUS = 1.06
    TARGET_VALUE_WIDE_GAP_CENTER_BONUS = 1.05
    TARGET_VALUE_WIDE_GAP_EDGE_PENALTY = 0.96
    TARGET_VALUE_NARROW_BOUNDARY_PROBE_BONUS = 1.03
    WRONG_COLOR_SLOT_PENALTY = 0.88

    CONTINUE_HIGH_ATTACKABILITY_BONUS = 1.10
    CONTINUE_LOW_ATTACKABILITY_PENALTY = 0.93
    STOP_LOW_ATTACKABILITY_BONUS = 1.04
    STOP_HIGH_ATTACKABILITY_PENALTY = 0.97
    CONTINUE_TARGET_FOLLOWUP_BONUS = 1.06
    STOP_TARGET_FOLLOWUP_PENALTY = 0.94
    CONTINUE_TARGET_FINISH_BONUS = 1.05
    CONTINUE_TARGET_FINISH_CHAIN_BONUS = 1.12
    CONTINUE_TARGET_CHAIN_BONUS = 1.08
    STOP_TARGET_FINISH_PENALTY = 0.96
    STOP_TARGET_FINISH_CHAIN_PENALTY = 0.92
    STOP_TARGET_CHAIN_PENALTY = 0.94
    CONTINUE_SELF_EXPOSURE_PENALTY = 0.93
    CONTINUE_NEW_DRAWN_EXPOSURE_PENALTY = 0.91
    CONTINUE_FINISH_FRAGILITY_PENALTY = 0.92
    CONTINUE_FAILURE_RECOVERY_BONUS = 1.05
    CONTINUE_JOINT_COLLAPSE_BONUS = 1.04
    STOP_SELF_EXPOSURE_BONUS = 1.05
    STOP_NEW_DRAWN_EXPOSURE_BONUS = 1.07
    STOP_FINISH_FRAGILITY_BONUS = 1.06
    STOP_FAILURE_RECOVERY_PENALTY = 0.95
    STOP_JOINT_COLLAPSE_PENALTY = 0.96
    PUBLIC_SELF_EXPOSURE_SECONDARY_BLEND = 0.35
    PUBLIC_SELF_EXPOSURE_FINISH_REFERENCE = 3.0
    PUBLIC_SELF_EXPOSURE_SAME_COLOR_ANCHOR_BONUS = 1.08
    PUBLIC_SELF_EXPOSURE_DOUBLE_COLOR_ANCHOR_BONUS = 1.12

    ATTACKABILITY_TIGHT_THRESHOLD = 0.34
    CONTINUATION_PRIOR_BASE = 0.52
    CONTINUATION_ATTACKABILITY_GAIN = 1.35
    CONTINUATION_TARGET_FOLLOWUP_BLEND = 0.32
    CONTINUATION_MIN = 0.08
    CONTINUATION_MAX = 0.95
    CONTINUATION_RECENT_CONTINUE_BONUS = 0.04
    CONTINUATION_RECENT_STOP_PENALTY = 0.04
    CONTINUATION_CONFIDENT_STREAK_BONUS = 0.03
    CONTINUATION_TARGET_FINISH_CHAIN_BONUS = 1.08
    CONTINUATION_TARGET_CHAIN_HISTORY_BONUS = 1.06
    PLAYER_GLOBAL_PROPAGATION_BONUS = 0.14
    JOINT_ACTION_PLAN_BONUS = 1.10
    JOINT_ACTION_PLAN_PENALTY = 0.92
    JOINT_ACTION_COMPONENT_REFERENCE = 0.16
    JOINT_ACTION_TEMPERATURE = 3.4
    JOINT_ACTION_STRUCTURED_BLEND = 0.22
    JOINT_ACTION_GENERATION_BLEND = 0.35
    JOINT_ACTION_GENERATION_PRIOR_BLEND = 0.26
    JOINT_ACTION_GENERATION_MAX_WEIGHT = 2.8
    JOINT_ACTION_GENERATIVE_TEMPERATURE = 2.2
    JOINT_ACTION_GENERATIVE_CONDITIONAL_BLEND = 0.58
    JOINT_ACTION_GENERATIVE_PRIOR_BLEND = 0.24
    JOINT_ACTION_GENERATIVE_STRUCTURAL_BLEND = 0.18
    JOINT_ACTION_GENERATIVE_BLEND = 0.40
    JOINT_ACTION_GENERATIVE_MAX_WEIGHT = 3.2
    JOINT_ACTION_POSTERIOR_BLEND = 0.58
    JOINT_ACTION_POSTERIOR_CONTEXT_BLEND = 0.30
    JOINT_ACTION_POSTERIOR_TRAJECTORY_BLEND = 0.24
    JOINT_ACTION_POSTERIOR_GAP_CREDIT = 0.14
    JOINT_ACTION_POSTERIOR_ENTROPY_CREDIT = 0.12
    JOINT_ACTION_POSTERIOR_RETRY_PROGRESSION_CREDIT = 0.14
    JOINT_ACTION_POSTERIOR_RETRY_ATTACK_WINDOW_CREDIT = 0.06
    JOINT_ACTION_POSTERIOR_RETRY_STALLED_PENALTY = 0.92
    JOINT_ACTION_POSTERIOR_RETRY_WIDE_PENALTY = 0.96
    JOINT_ACTION_SEQUENCE_RECENCY_DECAY = 0.72
    JOINT_ACTION_SEQUENCE_CONDITIONAL_BLEND = 0.16
    JOINT_ACTION_SEQUENCE_POSTERIOR_BLEND = 0.14
    JOINT_ACTION_PARAMETRIC_BLEND = 0.32
    JOINT_ACTION_POSTERIOR_PARAMETRIC_BLEND = 0.22
    PARAMETRIC_GENERATIVE_BIAS = -0.18
    PARAMETRIC_POSTERIOR_BIAS = -0.10
    PARAMETRIC_GENERATIVE_FEATURE_WEIGHTS: Dict[str, float] = {
        "player_selection": 0.55,
        "slot_selection": 0.72,
        "value_selection": 0.90,
        "slot_fit": 0.68,
        "continue_fit": 0.46,
        "attackability": 0.52,
        "attack_window": 0.40,
        "public_bridge": 0.22,
        "target_chain": 0.24,
        "finish_chain": 0.28,
        "global_propagation": 0.20,
        "same_player_transition": 0.16,
        "same_slot_transition": 0.24,
        "continue_choice": 0.10,
        "conditional_probability": 0.78,
        "prior_probability": 0.38,
        "structural_probability": 0.60,
        "sequence_probability": 0.54,
        "sequence_profile_stability": 0.22,
        "sequence_profile_density": 0.14,
        "base_probability": 0.70,
    }
    PARAMETRIC_POSTERIOR_FEATURE_WEIGHTS: Dict[str, float] = {
        "player_selection": 0.36,
        "slot_selection": 0.44,
        "value_selection": 0.58,
        "continue_fit": 0.30,
        "sequence_probability": 0.40,
        "base_probability": 0.54,
        "posterior_structured_fit": 0.72,
        "posterior_contextual_prior": 0.48,
        "posterior_trajectory_prior": 0.52,
        "posterior_sequence_prior": 0.46,
        "sequence_profile_stability": 0.18,
        "sequence_profile_density": 0.12,
    }
    JOINT_ACTION_POSTERIOR_MAX_WEIGHT = 4.0
    EPSILON = 1e-9

    def __init__(
        self,
        *,
        load_trained_weights: bool = True,
        parametric_weights_path: Optional[Path] = None,
    ) -> None:
        self.PARAMETRIC_GENERATIVE_FEATURE_WEIGHTS = dict(
            self.PARAMETRIC_GENERATIVE_FEATURE_WEIGHTS
        )
        self.PARAMETRIC_POSTERIOR_FEATURE_WEIGHTS = dict(
            self.PARAMETRIC_POSTERIOR_FEATURE_WEIGHTS
        )
        self.PARAMETRIC_GENERATIVE_BIAS = float(self.PARAMETRIC_GENERATIVE_BIAS)
        self.PARAMETRIC_POSTERIOR_BIAS = float(self.PARAMETRIC_POSTERIOR_BIAS)
        self.parametric_weights_source = "default"
        if load_trained_weights:
            self._load_trained_parametric_weights(
                parametric_weights_path=parametric_weights_path
            )

    def _load_trained_parametric_weights(
        self,
        *,
        parametric_weights_path: Optional[Path] = None,
    ) -> None:
        weights_path = parametric_weights_path or Path(__file__).with_name(
            "parametric_weights.json"
        )
        if not weights_path.exists():
            return
        try:
            payload = json.loads(weights_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        generative = payload.get("generative", {})
        posterior = payload.get("posterior", {})
        generative_weights = generative.get("weights", {})
        posterior_weights = posterior.get("weights", {})
        if isinstance(generative_weights, dict):
            self.PARAMETRIC_GENERATIVE_FEATURE_WEIGHTS.update(
                {
                    str(key): float(value)
                    for key, value in generative_weights.items()
                }
            )
        if isinstance(posterior_weights, dict):
            self.PARAMETRIC_POSTERIOR_FEATURE_WEIGHTS.update(
                {
                    str(key): float(value)
                    for key, value in posterior_weights.items()
                }
            )
        if "bias" in generative:
            self.PARAMETRIC_GENERATIVE_BIAS = float(generative["bias"])
        if "bias" in posterior:
            self.PARAMETRIC_POSTERIOR_BIAS = float(posterior["bias"])
        self.parametric_weights_source = str(weights_path)

    def build_guess_signals(
        self,
        game_state: GameState,
    ) -> Dict[str, List[GuessSignal]]:
        signals_by_player: Dict[str, List[GuessSignal]] = defaultdict(list)

        guess_action_index = 0
        for action in getattr(game_state, "actions", ()):
            if getattr(action, "action_type", None) != "guess":
                continue

            guessed_card = action.guessed_card()
            guesser_id = getattr(action, "guesser_id", None)
            target_player_id = getattr(action, "target_player_id", None)
            target_slot_index = getattr(action, "target_slot_index", None)
            if (
                guessed_card is None
                or guesser_id is None
                or target_player_id is None
                or target_slot_index is None
            ):
                continue

            signals_by_player[guesser_id].append(
                GuessSignal(
                    action_index=guess_action_index,
                    guesser_id=guesser_id,
                    target_player_id=target_player_id,
                    target_slot_index=target_slot_index,
                    guessed_card=guessed_card,
                    result=bool(getattr(action, "result", False)),
                    continued_turn=getattr(action, "continued_turn", None),
                )
            )
            guess_action_index += 1

        return signals_by_player

    def continuation_profile(
        self,
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        acting_player_id: Optional[str],
    ) -> Dict[str, float]:
        if acting_player_id is None:
            return {
                "continue_rate": self.CONTINUATION_PRIOR_BASE,
                "observations": 0.0,
                "history_blend": 0.0,
            }

        signals = guess_signals_by_player.get(acting_player_id, ())
        observed = [
            signal
            for signal in signals
            if signal.result and signal.continued_turn is not None
        ]
        observations = len(observed)
        if observations == 0:
            return {
                "continue_rate": self.CONTINUATION_PRIOR_BASE,
                "observations": 0.0,
                "history_blend": 0.0,
            }

        continue_count = sum(1 for signal in observed if signal.continued_turn)
        continue_rate = (continue_count + 1.0) / (observations + 2.0)
        recent_signal = observed[-1]
        if recent_signal.continued_turn:
            continue_rate += self.CONTINUATION_RECENT_CONTINUE_BONUS
        else:
            continue_rate -= self.CONTINUATION_RECENT_STOP_PENALTY
        if observations >= 2 and observed[-1].continued_turn and observed[-2].continued_turn:
            continue_rate += self.CONTINUATION_CONFIDENT_STREAK_BONUS
        continue_rate = clamp(continue_rate, self.CONTINUATION_MIN, self.CONTINUATION_MAX)
        history_blend = min(0.5, 0.18 * observations)
        return {
            "continue_rate": continue_rate,
            "observations": float(observations),
            "history_blend": history_blend,
        }

    def estimate_continue_likelihood(
        self,
        full_probability_matrix: FullProbabilityMatrix,
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        acting_player_id: Optional[str],
        *,
        exclude_slot: Optional[SlotKey] = None,
    ) -> Dict[str, float]:
        attackability = self.estimate_matrix_attackability(
            full_probability_matrix,
            acting_player_id=acting_player_id,
            exclude_slot=exclude_slot,
        )
        attackability_prior = clamp(
            0.5 + self.CONTINUATION_ATTACKABILITY_GAIN * (attackability - self.ATTACKABILITY_TIGHT_THRESHOLD),
            self.CONTINUATION_MIN,
            self.CONTINUATION_MAX,
        )
        target_followup_attackability = 0.0
        target_finish_chain_signal = 0.0
        target_chain_history_signal = 0.0
        if exclude_slot is not None:
            target_player_id = exclude_slot[0]
            target_followup_attackability = self._estimate_player_matrix_attackability(
                target_player_id,
                full_probability_matrix.get(target_player_id, {}),
                exclude_slot=exclude_slot,
            )
            target_followup_prior = clamp(
                0.5 + self.CONTINUATION_ATTACKABILITY_GAIN * (
                    target_followup_attackability - self.ATTACKABILITY_TIGHT_THRESHOLD
                ),
                self.CONTINUATION_MIN,
                self.CONTINUATION_MAX,
            )
            attackability_prior = (
                ((1.0 - self.CONTINUATION_TARGET_FOLLOWUP_BLEND) * attackability_prior)
                + (self.CONTINUATION_TARGET_FOLLOWUP_BLEND * target_followup_prior)
            )
            target_finish_chain_signal = self._matrix_target_finish_chain_signal(
                target_player_id,
                full_probability_matrix.get(target_player_id, {}),
                exclude_slot=exclude_slot,
            )
            target_chain_history_signal = self._continuation_target_chain_history_signal(
                guess_signals_by_player,
                acting_player_id,
                target_player_id,
            )

        profile = self.continuation_profile(guess_signals_by_player, acting_player_id)
        history_blend = profile["history_blend"]
        continue_likelihood = ((1.0 - history_blend) * attackability_prior) + (history_blend * profile["continue_rate"])

        if attackability >= self.ATTACKABILITY_TIGHT_THRESHOLD and profile["continue_rate"] >= 0.60:
            continue_likelihood *= 1.03
        elif attackability < self.ATTACKABILITY_TIGHT_THRESHOLD and profile["continue_rate"] <= 0.45:
            continue_likelihood *= 0.98
        if target_finish_chain_signal > 0.0:
            continue_likelihood *= 1.0 + (
                (self.CONTINUATION_TARGET_FINISH_CHAIN_BONUS - 1.0)
                * target_finish_chain_signal
            )
        if target_chain_history_signal > 0.0:
            continue_likelihood *= 1.0 + (
                (self.CONTINUATION_TARGET_CHAIN_HISTORY_BONUS - 1.0)
                * target_chain_history_signal
            )

        continue_likelihood = clamp(continue_likelihood, self.CONTINUATION_MIN, self.CONTINUATION_MAX)
        return {
            "continue_likelihood": continue_likelihood,
            "attackability": attackability,
            "target_followup_attackability": target_followup_attackability,
            "target_finish_chain_signal": target_finish_chain_signal,
            "target_chain_history_signal": target_chain_history_signal,
            "history_continue_rate": profile["continue_rate"],
            "history_observations": profile["observations"],
        }

    def _matrix_target_finish_chain_signal(
        self,
        player_id: str,
        probability_matrix: ProbabilityMatrix,
        *,
        exclude_slot: Optional[SlotKey] = None,
    ) -> float:
        remaining_hidden_slots = 0
        for slot_index, slot_distribution in probability_matrix.items():
            if not slot_distribution:
                continue
            key = slot_key(player_id, slot_index)
            if exclude_slot is not None and key == exclude_slot:
                continue
            remaining_hidden_slots += 1
        return clamp(
            (self.PLAYER_FINISH_PRESSURE_REFERENCE - float(remaining_hidden_slots))
            / self.PLAYER_FINISH_PRESSURE_REFERENCE,
            0.0,
            1.0,
        )

    def _continuation_target_chain_history_signal(
        self,
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        acting_player_id: Optional[str],
        target_player_id: str,
    ) -> float:
        if acting_player_id is None:
            return 0.0
        signals = guess_signals_by_player.get(acting_player_id, ())
        chain_score = 0.0
        recency_weight = 1.0
        for signal in reversed(signals):
            if signal.target_player_id != target_player_id:
                continue
            if signal.result:
                chain_score += 0.65 * recency_weight
                if signal.continued_turn is True:
                    chain_score += 0.20 * recency_weight
            recency_weight *= 0.62
            if recency_weight < 0.15:
                break
        return clamp(chain_score, 0.0, 1.0)

    def score_hypothesis(
        self,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        game_state: GameState,
    ) -> float:
        weight = 1.0

        for signals in guess_signals_by_player.values():
            for signal in signals:
                weight *= self._score_signal(game_state, hypothesis_by_player, signal)
                if weight <= 0.0:
                    return 0.0

        return weight

    def estimate_attackability(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        *,
        acting_player_id: Optional[str] = None,
        exclude_slot: Optional[SlotKey] = None,
    ) -> float:
        best = 0.0
        for player_id in game_state.inference_player_ids():
            if acting_player_id is not None and player_id == acting_player_id:
                continue
            best = max(
                best,
                self._player_attackability(
                    game_state,
                    hypothesis_by_player,
                    player_id,
                    exclude_slot=exclude_slot,
                ),
            )
        return best

    def estimate_matrix_attackability(
        self,
        full_probability_matrix: FullProbabilityMatrix,
        *,
        acting_player_id: Optional[str] = None,
        exclude_slot: Optional[SlotKey] = None,
    ) -> float:
        best_player_pressure = 0.0
        for player_id, probability_matrix in full_probability_matrix.items():
            if acting_player_id is not None and player_id == acting_player_id:
                continue
            player_pressure = self._estimate_player_matrix_attackability(
                player_id,
                probability_matrix,
                exclude_slot=exclude_slot,
            )
            best_player_pressure = max(best_player_pressure, player_pressure)
        return clamp(best_player_pressure, 0.0, 1.0)

    def _estimate_player_matrix_attackability(
        self,
        player_id: str,
        probability_matrix: ProbabilityMatrix,
        *,
        exclude_slot: Optional[SlotKey] = None,
    ) -> float:
        slot_certainties: List[float] = []
        for slot_index, slot_distribution in probability_matrix.items():
            key = slot_key(player_id, slot_index)
            if exclude_slot is not None and key == exclude_slot:
                continue
            if not slot_distribution:
                continue
            max_probability = max(slot_distribution.values())
            concentration = sum(probability * probability for probability in slot_distribution.values())
            effective_support = 1.0 / max(self.EPSILON, concentration)
            certainty = max_probability / max(1.0, effective_support ** 0.5)
            if len(slot_distribution) <= 2:
                certainty *= 1.05
            slot_certainties.append(certainty)

        if not slot_certainties:
            return 0.0
        slot_certainties.sort(reverse=True)
        player_pressure = slot_certainties[0]
        if len(slot_certainties) >= 2:
            player_pressure += self.MATRIX_SECONDARY_ATTACKABILITY_BLEND * slot_certainties[1]
        return clamp(player_pressure, 0.0, 1.0)

    def _score_signal(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
    ) -> float:
        target_card = self._resolve_slot_card(
            game_state,
            hypothesis_by_player,
            signal.target_player_id,
            signal.target_slot_index,
        )
        if target_card is None:
            return 1.0

        guesser_cards = self._player_cards(
            game_state,
            signal.guesser_id,
            hypothesis_by_player.get(signal.guesser_id, {}),
        )
        self_hand_weight = self._score_self_hand(guesser_cards, signal.guessed_card)
        target_player_weight = self._score_target_player_selection(
            game_state,
            hypothesis_by_player,
            signal,
        )
        target_slot_selection_weight = self._score_target_slot_selection(
            game_state,
            hypothesis_by_player,
            signal,
        )
        value_selection = self._value_selection_breakdown(
            game_state,
            hypothesis_by_player,
            signal,
            target_card,
            guesser_cards,
        )
        target_slot_weight = self._score_target_slot(
            game_state,
            hypothesis_by_player,
            signal,
            target_card,
        )
        continue_weight = self._score_continue_decision(
            game_state,
            hypothesis_by_player,
            signal,
        )
        joint_action_probability = self._joint_action_probability_breakdown(
            game_state,
            hypothesis_by_player,
            signal,
            target_card=target_card,
            guesser_cards=guesser_cards,
        )
        joint_action_generation = self._joint_action_generation_breakdown(
            game_state,
            hypothesis_by_player,
            signal,
            joint_action_probability=joint_action_probability,
        )
        joint_action_generative_probability = (
            self._joint_action_generative_probability_breakdown(
                game_state,
                hypothesis_by_player,
                signal,
                guesser_cards=guesser_cards,
            )
        )
        joint_action_fit = self._joint_action_fit_breakdown(
            game_state,
            hypothesis_by_player,
            signal,
            target_player_weight=target_player_weight,
            target_slot_selection_weight=target_slot_selection_weight,
            value_selection=value_selection,
            continue_weight=continue_weight,
        )
        structured_fit = max(
            self.EPSILON,
            target_player_weight
            * target_slot_selection_weight
            * value_selection["total_weight"]
            * target_slot_weight
            * continue_weight,
        )
        weight = self_hand_weight
        weight *= max(self.EPSILON, joint_action_probability["joint_probability"]) ** 0.35
        weight *= max(self.EPSILON, joint_action_generation["weight"])
        weight *= max(
            self.EPSILON,
            joint_action_generative_probability.get(
                "posterior_weight",
                joint_action_generative_probability["weight"],
            ),
        )
        weight *= joint_action_fit["weight"]
        weight *= structured_fit ** self.JOINT_ACTION_STRUCTURED_BLEND
        return max(self.EPSILON, weight)

    def _player_cards(
        self,
        game_state: GameState,
        player_id: str,
        hidden_cards_by_slot: Dict[int, Card],
    ) -> Sequence[Card]:
        cards: List[Card] = []
        for slot in game_state.resolved_ordered_slots(player_id):
            known = slot.known_card()
            if known is not None:
                cards.append(known)
                continue
            inferred = hidden_cards_by_slot.get(slot.slot_index)
            if inferred is not None:
                cards.append(inferred)
        return cards

    def _score_self_hand(
        self,
        player_cards: Sequence[Card],
        guessed_card: Card,
    ) -> float:
        weight = 1.0
        if guessed_card in player_cards:
            weight *= self.SELF_EXACT_GUESS_PENALTY

        guessed_numeric = numeric_card_value(guessed_card)
        if guessed_numeric is None:
            return weight

        same_color_values = {
            numeric_card_value(card)
            for card in player_cards
            if card[0] == guessed_card[0] and numeric_card_value(card) is not None
        }
        if (guessed_numeric - 1) in same_color_values or (guessed_numeric + 1) in same_color_values:
            weight *= self.SELF_ADJACENT_ANCHOR_BONUS
        return weight

    def _score_target_player_selection(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
    ) -> float:
        candidate_scores: Dict[str, float] = {}
        for player_id in game_state.players:
            if player_id == signal.guesser_id:
                continue
            player_score = self._player_attackability(
                game_state,
                hypothesis_by_player,
                player_id,
            )
            if player_score > 0.0:
                candidate_scores[player_id] = player_score

        if len(candidate_scores) <= 1:
            return 1.0

        chosen_score = candidate_scores.get(signal.target_player_id)
        if chosen_score is None:
            return 1.0

        best_score = max(candidate_scores.values())
        weight = 1.0
        if chosen_score >= best_score - self.EPSILON:
            weight *= self.TARGET_PLAYER_BEST_MATCH_BONUS
        elif chosen_score >= best_score * 0.85:
            weight *= self.TARGET_PLAYER_CLOSE_MATCH_BONUS
        else:
            weight *= self.TARGET_PLAYER_WEAK_CHOICE_PENALTY

        recent_collapse_pressure = self._recent_public_reveal_pressure(
            game_state,
            signal.target_player_id,
        )
        if recent_collapse_pressure > 0.0:
            weight *= 1.0 + (
                (self.TARGET_PLAYER_RECENT_COLLAPSE_BONUS - 1.0)
                * recent_collapse_pressure
            )
        collapse_streak_pressure = self._recent_player_collapse_streak_pressure(
            game_state,
            signal.target_player_id,
        )
        if collapse_streak_pressure > 0.0:
            weight *= 1.0 + (
                (self.TARGET_PLAYER_COLLAPSE_STREAK_BONUS - 1.0)
                * collapse_streak_pressure
            )
        public_bridge_signal = self._recent_public_bridge_signal_for_guess(
            game_state,
            signal,
        )
        if public_bridge_signal > 0.0:
            weight *= 1.0 + (
                (self.TARGET_PLAYER_PUBLIC_BRIDGE_BONUS - 1.0)
                * public_bridge_signal
            )
        finish_chain_signal = self._slot_finish_chain_pressure_after_hit(
            game_state,
            signal.target_player_id,
            signal.target_slot_index,
        )
        if finish_chain_signal > 0.0:
            weight *= 1.0 + (
                (self.TARGET_PLAYER_FINISH_CHAIN_BONUS - 1.0)
                * finish_chain_signal
            )
        target_chain_pressure = self._recent_target_chain_pressure(
            game_state,
            guesser_id=signal.guesser_id,
            target_player_id=signal.target_player_id,
        )
        if target_chain_pressure > 0.0:
            weight *= 1.0 + (
                (self.TARGET_PLAYER_RECENT_TARGET_CHAIN_BONUS - 1.0)
                * target_chain_pressure
            )

        previous_signal = self._previous_guess_signal(game_state, signal)
        if previous_signal is not None and previous_signal.target_player_id == signal.target_player_id:
            if previous_signal.result and previous_signal.continued_turn:
                weight *= self.TARGET_PLAYER_REPEAT_FOCUS_BONUS
                weight *= self.TARGET_PLAYER_CONFIDENT_CHAIN_BONUS
            elif previous_signal.result:
                weight *= self.TARGET_PLAYER_REPEAT_FOCUS_BONUS
        elif (
            previous_signal is not None
            and not previous_signal.result
        ):
            previous_target_score = candidate_scores.get(previous_signal.target_player_id, 0.0)
            if chosen_score > (previous_target_score + 0.08):
                weight *= self.TARGET_PLAYER_SWITCH_AFTER_FAILURE_BONUS
            if self._has_failure_switch_guess_continuity(previous_signal, signal):
                weight *= self.TARGET_PLAYER_SWITCH_FAILURE_CONTINUITY_BONUS
        elif (
            previous_signal is not None
            and previous_signal.result
            and previous_signal.continued_turn
        ):
            weight *= self.TARGET_PLAYER_BREAK_CONFIDENT_CHAIN_PENALTY
        return weight

    def _normalized_joint_action_component_fit(
        self,
        weight: float,
        *,
        reference: Optional[float] = None,
    ) -> float:
        fit_reference = (
            self.JOINT_ACTION_COMPONENT_REFERENCE
            if reference is None
            else max(reference, self.EPSILON)
        )
        return clamp((float(weight) - 1.0) / fit_reference, -1.0, 1.0)

    def _joint_action_fit_breakdown(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
        *,
        target_player_weight: float,
        target_slot_selection_weight: float,
        value_selection: Dict[str, Any],
        continue_weight: float,
    ) -> Dict[str, float]:
        target_player_fit = self._normalized_joint_action_component_fit(
            target_player_weight,
        )
        target_slot_fit = self._normalized_joint_action_component_fit(
            target_slot_selection_weight,
        )
        value_fit = self._normalized_joint_action_component_fit(
            float(value_selection.get("total_weight", 1.0)),
            reference=0.20,
        )
        continue_fit = self._normalized_joint_action_component_fit(
            continue_weight,
        )
        target_player_attackability = clamp(
            self._player_attackability(
                game_state,
                hypothesis_by_player,
                signal.target_player_id,
                exclude_slot=slot_key(signal.target_player_id, signal.target_slot_index),
            ),
            0.0,
            1.0,
        )
        target_slot_attackability = clamp(
            self._slot_attackability(
                game_state,
                hypothesis_by_player,
                signal.target_player_id,
                signal.target_slot_index,
            ),
            0.0,
            1.0,
        )
        attack_window_signal = self._slot_attack_window_pressure(
            game_state,
            signal.target_player_id,
            signal.target_slot_index,
        )
        joint_collapse_signal = self._continue_joint_collapse_signal(
            game_state,
            signal,
        )
        global_propagation_signal = self._global_public_propagation_pressure(game_state)
        public_bridge_signal = self._recent_public_bridge_signal_for_guess(
            game_state,
            signal,
        )
        target_chain_signal = self._recent_target_chain_pressure(
            game_state,
            guesser_id=signal.guesser_id,
            target_player_id=signal.target_player_id,
        )
        finish_chain_signal = self._slot_finish_chain_pressure_after_hit(
            game_state,
            signal.target_player_id,
            signal.target_slot_index,
        )

        local_alignment = clamp(
            (0.26 * max(0.0, target_player_fit))
            + (0.20 * max(0.0, target_slot_fit))
            + (0.24 * max(0.0, value_fit))
            + (0.18 * max(0.0, continue_fit))
            + (
                0.12
                * clamp(
                    (0.65 * target_player_attackability)
                    + (0.35 * target_slot_attackability),
                    0.0,
                    1.0,
                )
            ),
            0.0,
            1.0,
        )
        propagated_alignment = clamp(
            (0.28 * attack_window_signal)
            + (0.24 * joint_collapse_signal)
            + (0.22 * global_propagation_signal)
            + (0.14 * public_bridge_signal)
            + (0.12 * max(target_chain_signal, finish_chain_signal)),
            0.0,
            1.0,
        )
        penalty_signal = clamp(
            (0.34 * max(0.0, -target_player_fit))
            + (0.22 * max(0.0, -target_slot_fit))
            + (0.28 * max(0.0, -value_fit))
            + (0.16 * max(0.0, -continue_fit)),
            0.0,
            1.0,
        )
        joint_signal = clamp(
            (0.70 * local_alignment) + (0.30 * propagated_alignment),
            0.0,
            1.0,
        )
        weight = 1.0 + ((self.JOINT_ACTION_PLAN_BONUS - 1.0) * joint_signal)
        weight *= 1.0 - (
            (1.0 - self.JOINT_ACTION_PLAN_PENALTY) * penalty_signal
        )
        return {
            "weight": max(self.EPSILON, weight),
            "signal": joint_signal,
            "penalty_signal": penalty_signal,
            "local_alignment": local_alignment,
            "propagated_alignment": propagated_alignment,
            "target_player_fit": target_player_fit,
            "target_slot_fit": target_slot_fit,
            "value_fit": value_fit,
            "continue_fit": continue_fit,
            "target_player_attackability": target_player_attackability,
            "target_slot_attackability": target_slot_attackability,
            "attack_window_signal": attack_window_signal,
            "joint_collapse_signal": joint_collapse_signal,
            "global_propagation_signal": global_propagation_signal,
            "public_bridge_signal": public_bridge_signal,
            "target_chain_signal": target_chain_signal,
            "finish_chain_signal": finish_chain_signal,
        }

    def _softmax_probabilities(
        self,
        raw_scores: Dict[Any, float],
        *,
        temperature: Optional[float] = None,
    ) -> Dict[Any, float]:
        if not raw_scores:
            return {}
        max_score = max(float(score) for score in raw_scores.values())
        temperature = (
            self.JOINT_ACTION_TEMPERATURE
            if temperature is None
            else float(temperature)
        )
        exp_scores = {
            key: exp(
                temperature * (float(score) - max_score)
            )
            for key, score in raw_scores.items()
        }
        total = sum(exp_scores.values())
        if total <= 0.0:
            uniform = 1.0 / float(len(raw_scores))
            return {key: uniform for key in raw_scores}
        return {
            key: value / total
            for key, value in exp_scores.items()
        }

    def _representative_hidden_slot_index(
        self,
        game_state: GameState,
        player_id: str,
        *,
        preferred_slot_index: Optional[int] = None,
    ) -> Optional[int]:
        hidden_slots = [
            slot.slot_index
            for slot in game_state.resolved_ordered_slots(player_id)
            if slot.known_card() is None
        ]
        if not hidden_slots:
            return None
        if preferred_slot_index in hidden_slots:
            return preferred_slot_index
        return hidden_slots[0]

    def _joint_action_player_probabilities(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
    ) -> Dict[str, float]:
        raw_scores: Dict[str, float] = {}
        for player_id in game_state.players:
            if player_id == signal.guesser_id:
                continue
            representative_slot_index = self._representative_hidden_slot_index(
                game_state,
                player_id,
                preferred_slot_index=signal.target_slot_index,
            )
            if representative_slot_index is None:
                continue
            candidate_signal = replace(
                signal,
                target_player_id=player_id,
                target_slot_index=representative_slot_index,
            )
            raw_scores[player_id] = self._score_target_player_selection(
                game_state,
                hypothesis_by_player,
                candidate_signal,
            )
        return self._softmax_probabilities(raw_scores)

    def _joint_action_slot_probabilities(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
    ) -> Dict[int, float]:
        raw_scores: Dict[int, float] = {}
        for slot in game_state.resolved_ordered_slots(signal.target_player_id):
            if slot.known_card() is not None:
                continue
            candidate_signal = replace(
                signal,
                target_slot_index=slot.slot_index,
            )
            raw_scores[slot.slot_index] = self._score_target_slot_selection(
                game_state,
                hypothesis_by_player,
                candidate_signal,
            )
        return self._softmax_probabilities(raw_scores)

    def _joint_action_value_candidates(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
        *,
        target_card: Card,
    ) -> List[Card]:
        slot = game_state.get_slot(signal.target_player_id, signal.target_slot_index)
        target_numeric = numeric_card_value(target_card)
        guessed_numeric = numeric_card_value(signal.guessed_card)
        low, high, width = self._slot_numeric_interval(
            game_state,
            hypothesis_by_player,
            signal.target_player_id,
            signal.target_slot_index,
        )
        candidate_values: Set[int] = set()
        if guessed_numeric is not None:
            candidate_values.add(guessed_numeric)
        if target_numeric is not None:
            candidate_values.add(target_numeric)
            for value in range(
                max(0, target_numeric - 2),
                min(MAX_CARD_VALUE, target_numeric + 2) + 1,
            ):
                candidate_values.add(value)
        if low is not None and high is not None and width <= 6:
            for value in range(max(0, low), min(MAX_CARD_VALUE, high) + 1):
                candidate_values.add(value)
        for prior_failed in self._prior_failed_numeric_guesses_on_slot(
            game_state,
            signal,
        ):
            candidate_values.add(prior_failed)
            if prior_failed > 0:
                candidate_values.add(prior_failed - 1)
            if prior_failed < MAX_CARD_VALUE:
                candidate_values.add(prior_failed + 1)

        colors = (
            [slot.color]
            if getattr(slot, "color", None) in CARD_COLORS
            else list(CARD_COLORS)
        )
        candidates: Set[Card] = set()
        if signal.guessed_card[1] == JOKER:
            candidates.add(signal.guessed_card)
        if target_card[1] == JOKER:
            candidates.add(target_card)
        for color in colors:
            for value in sorted(candidate_values):
                candidates.add((color, value))
        candidates.add(signal.guessed_card)
        candidates.add(target_card)
        return sorted(candidates, key=card_sort_key)

    def _joint_action_value_probabilities(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
        *,
        target_card: Card,
        guesser_cards: Sequence[Card],
    ) -> Dict[Card, float]:
        raw_scores: Dict[Card, float] = {}
        for candidate_card in self._joint_action_value_candidates(
            game_state,
            hypothesis_by_player,
            signal,
            target_card=target_card,
        ):
            candidate_signal = replace(signal, guessed_card=candidate_card)
            value_breakdown = self._value_selection_breakdown(
                game_state,
                hypothesis_by_player,
                candidate_signal,
                target_card,
                guesser_cards,
            )
            raw_scores[candidate_card] = (
                value_breakdown["total_weight"]
                * self._score_target_slot(
                    game_state,
                    hypothesis_by_player,
                    candidate_signal,
                    target_card,
                )
            )
        return self._softmax_probabilities(raw_scores)

    def _joint_action_continue_probabilities(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
    ) -> Dict[bool, float]:
        if not signal.result or signal.continued_turn is None:
            return {True: 1.0, False: 1.0}
        raw_scores = {
            True: self._score_continue_decision(
                game_state,
                hypothesis_by_player,
                replace(signal, continued_turn=True),
            ),
            False: self._score_continue_decision(
                game_state,
                hypothesis_by_player,
                replace(signal, continued_turn=False),
            ),
        }
        return self._softmax_probabilities(raw_scores)

    def _joint_action_probability_breakdown(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
        *,
        target_card: Card,
        guesser_cards: Sequence[Card],
    ) -> Dict[str, Any]:
        player_probabilities = self._joint_action_player_probabilities(
            game_state,
            hypothesis_by_player,
            signal,
        )
        slot_probabilities = self._joint_action_slot_probabilities(
            game_state,
            hypothesis_by_player,
            signal,
        )
        value_probabilities = self._joint_action_value_probabilities(
            game_state,
            hypothesis_by_player,
            signal,
            target_card=target_card,
            guesser_cards=guesser_cards,
        )
        continue_probabilities = self._joint_action_continue_probabilities(
            game_state,
            hypothesis_by_player,
            signal,
        )
        player_probability = player_probabilities.get(signal.target_player_id, 1.0)
        slot_probability = slot_probabilities.get(signal.target_slot_index, 1.0)
        value_probability = value_probabilities.get(signal.guessed_card, 1.0)
        continue_probability = (
            continue_probabilities.get(bool(signal.continued_turn), 1.0)
            if signal.continued_turn is not None
            else 1.0
        )
        return {
            "player_probability": player_probability,
            "slot_probability": slot_probability,
            "value_probability": value_probability,
            "continue_probability": continue_probability,
            "joint_probability": max(
                self.EPSILON,
                player_probability
                * slot_probability
                * value_probability
                * continue_probability,
            ),
            "player_candidate_count": float(len(player_probabilities)),
            "slot_candidate_count": float(len(slot_probabilities)),
            "value_candidate_count": float(len(value_probabilities)),
            "continue_candidate_count": float(len(continue_probabilities)),
        }

    def _joint_action_generation_breakdown(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
        *,
        joint_action_probability: Dict[str, Any],
    ) -> Dict[str, float]:
        candidate_space = max(
            1.0,
            float(joint_action_probability.get("player_candidate_count", 1.0))
            * float(joint_action_probability.get("slot_candidate_count", 1.0))
            * float(joint_action_probability.get("value_candidate_count", 1.0))
            * float(joint_action_probability.get("continue_candidate_count", 1.0)),
        )
        uniform_probability = 1.0 / candidate_space
        probability_advantage = (
            float(joint_action_probability.get("joint_probability", self.EPSILON))
            / max(self.EPSILON, uniform_probability)
        )
        target_player_attackability = self._player_attackability(
            game_state,
            hypothesis_by_player,
            signal.target_player_id,
        )
        target_slot_attackability = self._slot_attackability(
            game_state,
            hypothesis_by_player,
            signal.target_player_id,
            signal.target_slot_index,
        )
        attack_window_signal = self._slot_attack_window_pressure(
            game_state,
            signal.target_player_id,
            signal.target_slot_index,
        )
        global_propagation_signal = self._global_public_propagation_pressure(game_state)
        public_bridge_signal = self._recent_public_bridge_signal_for_guess(
            game_state,
            signal,
        )
        target_chain_signal = self._recent_target_chain_pressure(
            game_state,
            guesser_id=signal.guesser_id,
            target_player_id=signal.target_player_id,
        )
        finish_chain_signal = self._slot_finish_chain_pressure_after_hit(
            game_state,
            signal.target_player_id,
            signal.target_slot_index,
        )
        generative_prior = clamp(
            0.42
            + (0.20 * target_player_attackability)
            + (0.14 * target_slot_attackability)
            + (0.10 * attack_window_signal)
            + (0.05 * global_propagation_signal)
            + (0.04 * public_bridge_signal)
            + (0.03 * target_chain_signal)
            + (0.02 * finish_chain_signal),
            0.35,
            1.35,
        )
        normalized_generation = max(
            self.EPSILON,
            probability_advantage ** self.JOINT_ACTION_GENERATION_BLEND,
        )
        weight = clamp(
            normalized_generation
            * (generative_prior ** self.JOINT_ACTION_GENERATION_PRIOR_BLEND),
            self.EPSILON,
            self.JOINT_ACTION_GENERATION_MAX_WEIGHT,
        )
        return {
            "candidate_space": candidate_space,
            "uniform_probability": uniform_probability,
            "probability_advantage": probability_advantage,
            "target_player_attackability": target_player_attackability,
            "target_slot_attackability": target_slot_attackability,
            "attack_window_signal": attack_window_signal,
            "global_propagation_signal": global_propagation_signal,
            "public_bridge_signal": public_bridge_signal,
            "target_chain_signal": target_chain_signal,
            "finish_chain_signal": finish_chain_signal,
            "generative_prior": generative_prior,
            "weight": weight,
        }

    def _normalize_action_feature_score(
        self,
        value: float,
        *,
        center: float = 1.0,
        radius: float = 0.30,
    ) -> float:
        return clamp(
            0.5 + ((float(value) - center) / max(self.EPSILON, 2.0 * radius)),
            0.0,
            1.0,
        )

    def _action_feature_vector(
        self,
        *,
        player_selection_score: float,
        slot_selection_score: float,
        value_selection_score: float,
        local_slot_fit: float,
        continue_score: float,
        slot_attackability: float,
        attack_window_signal: float,
        public_bridge_signal: float,
        target_chain_signal: float,
        finish_chain_signal: float,
        global_propagation_signal: float,
        action_key: Tuple[str, int, Card, Optional[bool]],
        previous_signal: Optional[GuessSignal],
    ) -> Dict[str, float]:
        same_player_transition = 0.0
        same_slot_transition = 0.0
        if previous_signal is not None:
            same_player_transition = (
                1.0
                if action_key[0] == previous_signal.target_player_id
                else 0.0
            )
            same_slot_transition = (
                1.0
                if same_player_transition > 0.0
                and action_key[1] == previous_signal.target_slot_index
                else 0.0
            )
        continued_turn = action_key[3]
        continue_choice = (
            0.5
            if continued_turn is None
            else 1.0
            if continued_turn
            else 0.0
        )
        return {
            "player_selection": self._normalize_action_feature_score(
                player_selection_score
            ),
            "slot_selection": self._normalize_action_feature_score(
                slot_selection_score
            ),
            "value_selection": self._normalize_action_feature_score(
                value_selection_score
            ),
            "slot_fit": self._normalize_action_feature_score(local_slot_fit),
            "continue_fit": self._normalize_action_feature_score(continue_score),
            "attackability": clamp(slot_attackability, 0.0, 1.0),
            "attack_window": clamp(attack_window_signal, 0.0, 1.0),
            "public_bridge": clamp(public_bridge_signal, 0.0, 1.0),
            "target_chain": clamp(target_chain_signal, 0.0, 1.0),
            "finish_chain": clamp(finish_chain_signal, 0.0, 1.0),
            "global_propagation": clamp(global_propagation_signal, 0.0, 1.0),
            "same_player_transition": same_player_transition,
            "same_slot_transition": same_slot_transition,
            "continue_choice": continue_choice,
        }

    def _observed_action_feature_vector(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
        *,
        guesser_cards: Sequence[Card],
    ) -> Optional[Dict[str, float]]:
        target_card = self._resolve_slot_card(
            game_state,
            hypothesis_by_player,
            signal.target_player_id,
            signal.target_slot_index,
        )
        if target_card is None:
            return None
        previous_signal = self._previous_guess_signal(game_state, signal)
        player_selection_score = self._score_target_player_selection(
            game_state,
            hypothesis_by_player,
            signal,
        )
        slot_selection_score = self._score_target_slot_selection(
            game_state,
            hypothesis_by_player,
            signal,
        )
        value_breakdown = self._value_selection_breakdown(
            game_state,
            hypothesis_by_player,
            signal,
            target_card,
            guesser_cards,
        )
        local_slot_fit = self._score_target_slot(
            game_state,
            hypothesis_by_player,
            signal,
            target_card,
        )
        continue_score = (
            self._score_continue_decision(
                game_state,
                hypothesis_by_player,
                signal,
            )
            if signal.continued_turn is not None
            else 1.0
        )
        slot_attackability = self._slot_attackability(
            game_state,
            hypothesis_by_player,
            signal.target_player_id,
            signal.target_slot_index,
        )
        attack_window_signal = self._slot_attack_window_pressure(
            game_state,
            signal.target_player_id,
            signal.target_slot_index,
        )
        public_bridge_signal = self._recent_public_bridge_signal_for_guess(
            game_state,
            signal,
        )
        target_chain_signal = self._recent_target_chain_pressure(
            game_state,
            guesser_id=signal.guesser_id,
            target_player_id=signal.target_player_id,
        )
        finish_chain_signal = self._slot_finish_chain_pressure_after_hit(
            game_state,
            signal.target_player_id,
            signal.target_slot_index,
        )
        action_key = (
            signal.target_player_id,
            signal.target_slot_index,
            signal.guessed_card,
            signal.continued_turn if signal.continued_turn is not None else None,
        )
        return self._action_feature_vector(
            player_selection_score=player_selection_score,
            slot_selection_score=slot_selection_score,
            value_selection_score=float(value_breakdown["total_weight"]),
            local_slot_fit=local_slot_fit,
            continue_score=continue_score,
            slot_attackability=slot_attackability,
            attack_window_signal=attack_window_signal,
            public_bridge_signal=public_bridge_signal,
            target_chain_signal=target_chain_signal,
            finish_chain_signal=finish_chain_signal,
            global_propagation_signal=self._global_public_propagation_pressure(
                game_state
            ),
            action_key=action_key,
            previous_signal=previous_signal,
        )

    def _sequential_action_feature_profile(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
        *,
        guesser_cards: Sequence[Card],
    ) -> Dict[str, Any]:
        if signal.action_index <= 0:
            return {
                "count": 0.0,
                "total_weight": 0.0,
                "stability": 0.0,
                "means": {},
                "variances": {},
            }

        feature_sums: DefaultDict[str, float] = defaultdict(float)
        feature_square_sums: DefaultDict[str, float] = defaultdict(float)
        total_weight = 0.0
        feature_count = 0.0
        guess_action_index = 0

        for action in getattr(game_state, "actions", ()):
            if getattr(action, "action_type", None) != "guess":
                continue
            guessed_card = action.guessed_card()
            guesser_id = getattr(action, "guesser_id", None)
            target_player_id = getattr(action, "target_player_id", None)
            target_slot_index = getattr(action, "target_slot_index", None)
            if (
                guessed_card is None
                or guesser_id is None
                or target_player_id is None
                or target_slot_index is None
            ):
                guess_action_index += 1
                continue

            observed_signal = GuessSignal(
                action_index=guess_action_index,
                guesser_id=guesser_id,
                target_player_id=target_player_id,
                target_slot_index=target_slot_index,
                guessed_card=guessed_card,
                result=bool(getattr(action, "result", False)),
                continued_turn=getattr(action, "continued_turn", None),
            )
            guess_action_index += 1
            if observed_signal.action_index >= signal.action_index:
                break
            if observed_signal.guesser_id != signal.guesser_id:
                continue

            feature_vector = self._observed_action_feature_vector(
                game_state,
                hypothesis_by_player,
                observed_signal,
                guesser_cards=guesser_cards,
            )
            if not feature_vector:
                continue

            recency_distance = max(
                0.0,
                float(signal.action_index - observed_signal.action_index - 1),
            )
            history_weight = self.JOINT_ACTION_SEQUENCE_RECENCY_DECAY ** recency_distance
            for feature_name, feature_value in feature_vector.items():
                feature_sums[feature_name] += history_weight * float(feature_value)
                feature_square_sums[feature_name] += (
                    history_weight * (float(feature_value) ** 2)
                )
            total_weight += history_weight
            feature_count += 1.0

        if total_weight <= 0.0 or feature_count <= 0.0:
            return {
                "count": 0.0,
                "total_weight": 0.0,
                "stability": 0.0,
                "means": {},
                "variances": {},
            }

        means = {
            feature_name: feature_sums[feature_name] / total_weight
            for feature_name in feature_sums
        }
        variances = {
            feature_name: max(
                self.EPSILON,
                (feature_square_sums[feature_name] / total_weight)
                - (means[feature_name] ** 2),
            )
            for feature_name in means
        }
        average_stddev = (
            sum(sqrt(variances[feature_name]) for feature_name in variances)
            / max(1.0, float(len(variances)))
        )
        stability = clamp(1.0 - (1.6 * average_stddev), 0.0, 1.0)
        return {
            "count": feature_count,
            "total_weight": total_weight,
            "stability": stability,
            "means": means,
            "variances": variances,
        }

    def _sequence_feature_similarity(
        self,
        candidate_features: Dict[str, float],
        profile: Dict[str, Any],
    ) -> float:
        means = profile.get("means", {})
        if not means:
            return 1.0
        variances = profile.get("variances", {})
        weighted_distance = 0.0
        total_feature_weight = 0.0
        for feature_name, mean in means.items():
            candidate_value = float(candidate_features.get(feature_name, 0.5))
            variance = float(variances.get(feature_name, 0.0))
            scale = max(0.12, sqrt(max(self.EPSILON, variance)) + 0.08)
            feature_weight = (
                1.25
                if feature_name in {"same_player_transition", "same_slot_transition"}
                else 1.15
                if feature_name in {"value_selection", "continue_fit", "continue_choice"}
                else 1.0
            )
            weighted_distance += (
                feature_weight * abs(candidate_value - float(mean)) / scale
            )
            total_feature_weight += feature_weight
        normalized_distance = weighted_distance / max(
            self.EPSILON,
            total_feature_weight,
        )
        stability = clamp(float(profile.get("stability", 0.0)), 0.0, 1.0)
        return clamp(
            exp(-normalized_distance * (1.0 + (0.55 * stability))),
            self.EPSILON,
            1.0,
        )

    def _parametric_action_feature_bundle(
        self,
        *,
        action_key: Tuple[str, int, Card, Optional[bool]],
        metadata: Dict[str, Any],
        base_probability: float,
        sequence_profile: Dict[str, Any],
    ) -> Dict[str, float]:
        action_features = dict(metadata.get("action_features", {}))
        sequence_count = float(sequence_profile.get("count", 0.0))
        sequence_total_weight = float(sequence_profile.get("total_weight", 0.0))
        density = clamp(
            sequence_total_weight / max(1.0, sequence_count + 1.0),
            0.0,
            1.0,
        )
        feature_bundle = {
            **action_features,
            "conditional_probability": clamp(
                float(
                    metadata.get(
                        "conditional_probability",
                        base_probability,
                    )
                ),
                0.0,
                1.0,
            ),
            "prior_probability": clamp(
                float(metadata.get("prior_probability", base_probability)),
                0.0,
                1.0,
            ),
            "structural_probability": clamp(
                float(metadata.get("structural_probability", base_probability)),
                0.0,
                1.0,
            ),
            "sequence_probability": clamp(
                float(metadata.get("sequence_probability", base_probability)),
                0.0,
                1.0,
            ),
            "base_probability": clamp(float(base_probability), 0.0, 1.0),
            "sequence_profile_stability": clamp(
                float(sequence_profile.get("stability", 0.0)),
                0.0,
                1.0,
            ),
            "sequence_profile_density": density,
        }
        if "continue_choice" not in feature_bundle:
            continued_turn = action_key[3]
            feature_bundle["continue_choice"] = (
                0.5
                if continued_turn is None
                else 1.0
                if continued_turn
                else 0.0
            )
        return feature_bundle

    def _parametric_action_logit(
        self,
        features: Dict[str, float],
        *,
        weights: Dict[str, float],
        bias: float,
    ) -> float:
        logit = float(bias)
        for feature_name, feature_weight in weights.items():
            feature_value = clamp(float(features.get(feature_name, 0.0)), 0.0, 1.0)
            logit += float(feature_weight) * (feature_value - 0.5)
        return logit

    def _joint_action_generative_probability_breakdown(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
        *,
        guesser_cards: Sequence[Card],
        include_candidate_catalog: bool = False,
    ) -> Dict[str, Any]:
        raw_scores: Dict[Tuple[str, int, Card, Optional[bool]], float] = {}
        metadata: Dict[Tuple[str, int, Card, Optional[bool]], Dict[str, Any]] = {}
        global_propagation_signal = self._global_public_propagation_pressure(game_state)
        previous_signal = self._previous_guess_signal(game_state, signal)
        sequence_profile = self._sequential_action_feature_profile(
            game_state,
            hypothesis_by_player,
            signal,
            guesser_cards=guesser_cards,
        )

        for player_id in game_state.players:
            if player_id == signal.guesser_id:
                continue
            hidden_slots = [
                slot.slot_index
                for slot in game_state.resolved_ordered_slots(player_id)
                if slot.known_card() is None
            ]
            for slot_index in hidden_slots:
                target_card = hypothesis_by_player.get(player_id, {}).get(slot_index)
                if target_card is None:
                    continue
                base_signal = replace(
                    signal,
                    target_player_id=player_id,
                    target_slot_index=slot_index,
                )
                player_selection_score = self._score_target_player_selection(
                    game_state,
                    hypothesis_by_player,
                    base_signal,
                )
                slot_selection_score = self._score_target_slot_selection(
                    game_state,
                    hypothesis_by_player,
                    base_signal,
                )
                slot_attackability = self._slot_attackability(
                    game_state,
                    hypothesis_by_player,
                    player_id,
                    slot_index,
                )
                attack_window_signal = self._slot_attack_window_pressure(
                    game_state,
                    player_id,
                    slot_index,
                )
                public_bridge_signal = self._recent_public_bridge_signal_for_guess(
                    game_state,
                    base_signal,
                )
                target_chain_signal = self._recent_target_chain_pressure(
                    game_state,
                    guesser_id=signal.guesser_id,
                    target_player_id=player_id,
                )
                finish_chain_signal = self._slot_finish_chain_pressure_after_hit(
                    game_state,
                    player_id,
                    slot_index,
                )
                for candidate_card in self._joint_action_value_candidates(
                    game_state,
                    hypothesis_by_player,
                    base_signal,
                    target_card=target_card,
                ):
                    candidate_signal = replace(base_signal, guessed_card=candidate_card)
                    value_breakdown = self._value_selection_breakdown(
                        game_state,
                        hypothesis_by_player,
                        candidate_signal,
                        target_card,
                        guesser_cards,
                    )
                    local_slot_fit = self._score_target_slot(
                        game_state,
                        hypothesis_by_player,
                        candidate_signal,
                        target_card,
                    )
                    continue_options: Sequence[Optional[bool]]
                    if signal.result and signal.continued_turn is not None:
                        continue_options = (True, False)
                    else:
                        continue_options = (None,)
                    for continued_turn in continue_options:
                        action_signal = (
                            replace(candidate_signal, continued_turn=continued_turn)
                            if continued_turn is not None
                            else candidate_signal
                        )
                        continue_score = (
                            self._score_continue_decision(
                                game_state,
                                hypothesis_by_player,
                                action_signal,
                            )
                            if continued_turn is not None
                            else 1.0
                        )
                        same_card_bonus = 0.20 if candidate_card == target_card else -0.05
                        self_card_penalty = (
                            0.18 if candidate_card in guesser_cards else 0.0
                        )
                        raw_score = (
                            (0.23 * player_selection_score)
                            + (0.18 * slot_selection_score)
                            + (0.22 * value_breakdown["total_weight"])
                            + (0.11 * local_slot_fit)
                            + (0.10 * continue_score)
                            + (0.06 * slot_attackability)
                            + (0.04 * attack_window_signal)
                            + (0.02 * public_bridge_signal)
                            + (0.02 * target_chain_signal)
                            + (0.02 * finish_chain_signal)
                            + (0.02 * global_propagation_signal)
                            + same_card_bonus
                            - self_card_penalty
                        )
                        action_key = (
                            player_id,
                            slot_index,
                            candidate_card,
                            continued_turn,
                        )
                        action_features = self._action_feature_vector(
                            player_selection_score=player_selection_score,
                            slot_selection_score=slot_selection_score,
                            value_selection_score=float(
                                value_breakdown["total_weight"]
                            ),
                            local_slot_fit=local_slot_fit,
                            continue_score=continue_score,
                            slot_attackability=slot_attackability,
                            attack_window_signal=attack_window_signal,
                            public_bridge_signal=public_bridge_signal,
                            target_chain_signal=target_chain_signal,
                            finish_chain_signal=finish_chain_signal,
                            global_propagation_signal=global_propagation_signal,
                            action_key=action_key,
                            previous_signal=previous_signal,
                        )
                        sequence_similarity = self._sequence_feature_similarity(
                            action_features,
                            sequence_profile,
                        )
                        raw_scores[action_key] = raw_score
                        metadata[action_key] = {
                            "player_selection_score": player_selection_score,
                            "slot_selection_score": slot_selection_score,
                            "value_selection_score": value_breakdown["total_weight"],
                            "local_slot_fit": local_slot_fit,
                            "continue_score": continue_score,
                            "slot_attackability": slot_attackability,
                            "attack_window_signal": attack_window_signal,
                            "public_bridge_signal": public_bridge_signal,
                            "target_chain_signal": target_chain_signal,
                            "finish_chain_signal": finish_chain_signal,
                            "global_propagation_signal": global_propagation_signal,
                            "action_features": action_features,
                            "sequence_similarity": sequence_similarity,
                        }

        if not raw_scores:
            return {
                "candidate_count": 1.0,
                "uniform_probability": 1.0,
                "probability": 1.0,
                "conditional_probability": 1.0,
                "prior_probability": 1.0,
                "structural_probability": 1.0,
                "probability_advantage": 1.0,
                "weight": 1.0,
                "observed_rank": 1.0,
                "top_actions": [],
            }

        structural_probabilities = self._softmax_probabilities(
            raw_scores,
            temperature=self.JOINT_ACTION_GENERATIVE_TEMPERATURE,
        )
        player_raw_scores: Dict[str, float] = {}
        slot_raw_scores: DefaultDict[str, Dict[int, float]] = defaultdict(dict)
        value_raw_scores: DefaultDict[Tuple[str, int], Dict[Card, float]] = defaultdict(dict)
        continue_raw_scores: DefaultDict[
            Tuple[str, int, Card],
            Dict[Optional[bool], float],
        ] = defaultdict(dict)
        prior_raw_scores: Dict[Tuple[str, int, Card, Optional[bool]], float] = {}

        for action_key, meta in metadata.items():
            player_id, slot_index, candidate_card, continued_turn = action_key
            player_raw_scores[player_id] = max(
                float(meta.get("player_selection_score", 0.0)),
                player_raw_scores.get(player_id, float("-inf")),
            )
            slot_raw_scores[player_id][slot_index] = max(
                float(meta.get("slot_selection_score", 0.0)),
                slot_raw_scores[player_id].get(slot_index, float("-inf")),
            )
            value_raw_scores[(player_id, slot_index)][candidate_card] = max(
                (
                    0.72 * float(meta.get("value_selection_score", 0.0))
                    + 0.28 * float(meta.get("local_slot_fit", 0.0))
                ),
                value_raw_scores[(player_id, slot_index)].get(
                    candidate_card,
                    float("-inf"),
                ),
            )
            continue_raw_scores[(player_id, slot_index, candidate_card)][
                continued_turn
            ] = float(meta.get("continue_score", 1.0))
            prior_raw_scores[action_key] = (
                (0.34 * float(meta.get("slot_attackability", 0.0)))
                + (0.18 * float(meta.get("attack_window_signal", 0.0)))
                + (0.12 * float(meta.get("global_propagation_signal", 0.0)))
                + (0.10 * float(meta.get("public_bridge_signal", 0.0)))
                + (0.10 * float(meta.get("target_chain_signal", 0.0)))
                + (0.08 * float(meta.get("finish_chain_signal", 0.0)))
                + (0.08 * float(meta.get("value_selection_score", 0.0)))
            )

        player_probabilities = self._softmax_probabilities(
            player_raw_scores,
            temperature=self.JOINT_ACTION_GENERATIVE_TEMPERATURE,
        )
        slot_probabilities = {
            player_id: self._softmax_probabilities(
                raw_slot_scores,
                temperature=self.JOINT_ACTION_GENERATIVE_TEMPERATURE,
            )
            for player_id, raw_slot_scores in slot_raw_scores.items()
        }
        value_probabilities = {
            slot_key: self._softmax_probabilities(
                raw_value_scores,
                temperature=self.JOINT_ACTION_GENERATIVE_TEMPERATURE,
            )
            for slot_key, raw_value_scores in value_raw_scores.items()
        }
        continue_probabilities = {
            continue_key: self._softmax_probabilities(
                raw_continue_scores,
                temperature=self.JOINT_ACTION_GENERATIVE_TEMPERATURE,
            )
            for continue_key, raw_continue_scores in continue_raw_scores.items()
        }
        prior_probabilities = self._softmax_probabilities(
            prior_raw_scores,
            temperature=0.7 * self.JOINT_ACTION_GENERATIVE_TEMPERATURE,
        )
        sequence_raw_scores = {
            action_key: max(
                self.EPSILON,
                float(meta.get("sequence_similarity", 1.0)),
            )
            for action_key, meta in metadata.items()
        }
        sequence_probabilities = self._softmax_probabilities(
            sequence_raw_scores,
            temperature=0.85 * self.JOINT_ACTION_GENERATIVE_TEMPERATURE,
        )
        action_probability_mass: Dict[Tuple[str, int, Card, Optional[bool]], float] = {}
        for action_key in raw_scores:
            player_id, slot_index, candidate_card, continued_turn = action_key
            conditional_probability = (
                float(player_probabilities.get(player_id, self.EPSILON))
                * float(
                    slot_probabilities.get(player_id, {}).get(
                        slot_index,
                        self.EPSILON,
                    )
                )
                * float(
                    value_probabilities.get((player_id, slot_index), {}).get(
                        candidate_card,
                        self.EPSILON,
                    )
                )
                * float(
                    continue_probabilities.get(
                        (player_id, slot_index, candidate_card),
                        {},
                    ).get(continued_turn, 1.0)
                )
            )
            prior_probability = float(
                prior_probabilities.get(action_key, self.EPSILON)
            )
            structural_probability = float(
                structural_probabilities.get(action_key, self.EPSILON)
            )
            sequence_probability = float(
                sequence_probabilities.get(action_key, self.EPSILON)
            )
            action_probability_mass[action_key] = (
                max(self.EPSILON, conditional_probability)
                ** self.JOINT_ACTION_GENERATIVE_CONDITIONAL_BLEND
                * max(self.EPSILON, prior_probability)
                ** self.JOINT_ACTION_GENERATIVE_PRIOR_BLEND
                * max(self.EPSILON, structural_probability)
                ** self.JOINT_ACTION_GENERATIVE_STRUCTURAL_BLEND
                * max(self.EPSILON, sequence_probability)
                ** self.JOINT_ACTION_SEQUENCE_CONDITIONAL_BLEND
            )
            metadata[action_key]["conditional_probability"] = conditional_probability
            metadata[action_key]["prior_probability"] = prior_probability
            metadata[action_key]["structural_probability"] = structural_probability
            metadata[action_key]["sequence_probability"] = sequence_probability

        total_probability_mass = sum(action_probability_mass.values())
        if total_probability_mass <= 0.0:
            uniform_probability = 1.0 / float(len(action_probability_mass))
            base_action_probabilities = {
                action_key: uniform_probability
                for action_key in action_probability_mass
            }
        else:
            base_action_probabilities = {
                action_key: probability_mass / total_probability_mass
                for action_key, probability_mass in action_probability_mass.items()
            }
        parametric_raw_scores: Dict[Tuple[str, int, Card, Optional[bool]], float] = {}
        for action_key, base_probability in base_action_probabilities.items():
            meta = metadata.get(action_key, {})
            parametric_features = self._parametric_action_feature_bundle(
                action_key=action_key,
                metadata=meta,
                base_probability=base_probability,
                sequence_profile=sequence_profile,
            )
            parametric_logit = self._parametric_action_logit(
                parametric_features,
                weights=self.PARAMETRIC_GENERATIVE_FEATURE_WEIGHTS,
                bias=self.PARAMETRIC_GENERATIVE_BIAS,
            )
            parametric_raw_scores[action_key] = parametric_logit
            meta["parametric_features"] = parametric_features
            meta["parametric_logit"] = parametric_logit
        parametric_probabilities = self._softmax_probabilities(
            parametric_raw_scores,
            temperature=1.0,
        )
        blended_action_mass: Dict[
            Tuple[str, int, Card, Optional[bool]],
            float,
        ] = {}
        for action_key, base_probability in base_action_probabilities.items():
            parametric_probability = float(
                parametric_probabilities.get(action_key, self.EPSILON)
            )
            blended_action_mass[action_key] = (
                max(self.EPSILON, base_probability)
                ** (1.0 - self.JOINT_ACTION_PARAMETRIC_BLEND)
                * max(self.EPSILON, parametric_probability)
                ** self.JOINT_ACTION_PARAMETRIC_BLEND
            )
            metadata[action_key]["base_probability"] = base_probability
            metadata[action_key]["parametric_probability"] = parametric_probability
        total_blended_action_mass = sum(blended_action_mass.values())
        if total_blended_action_mass <= 0.0:
            uniform_probability = 1.0 / float(max(1, len(blended_action_mass)))
            action_probabilities = {
                action_key: uniform_probability
                for action_key in blended_action_mass
            }
        else:
            action_probabilities = {
                action_key: probability_mass / total_blended_action_mass
                for action_key, probability_mass in blended_action_mass.items()
            }
        posterior_action_mass: Dict[
            Tuple[str, int, Card, Optional[bool]],
            float,
        ] = {}
        for action_key, action_probability in action_probabilities.items():
            meta = metadata.get(action_key, {})
            structured_fit = max(
                self.EPSILON,
                float(meta.get("player_selection_score", 1.0))
                * float(meta.get("slot_selection_score", 1.0))
                * float(meta.get("value_selection_score", 1.0))
                * max(self.EPSILON, float(meta.get("continue_score", 1.0))),
            )
            contextual_prior = clamp(
                0.55
                + (0.16 * float(meta.get("slot_attackability", 0.0)))
                + (0.10 * float(meta.get("attack_window_signal", 0.0)))
                + (0.06 * float(meta.get("global_propagation_signal", 0.0)))
                + (0.05 * float(meta.get("public_bridge_signal", 0.0)))
                + (0.04 * float(meta.get("target_chain_signal", 0.0)))
                + (0.04 * float(meta.get("finish_chain_signal", 0.0))),
                0.35,
                1.55,
            )
            sequence_prior = clamp(
                0.65
                + (0.20 * float(meta.get("sequence_probability", 0.0)))
                + (0.10 * float(sequence_profile.get("stability", 0.0))),
                0.35,
                1.55,
            )
            trajectory_prior = 1.0
            if previous_signal is not None:
                same_player = action_key[0] == previous_signal.target_player_id
                same_slot = same_player and action_key[1] == previous_signal.target_slot_index
                if previous_signal.result and previous_signal.continued_turn:
                    if same_player:
                        trajectory_prior *= 1.0 + (
                            0.14 * float(meta.get("target_chain_signal", 0.0))
                        )
                    if same_slot:
                        trajectory_prior *= 1.0 + (
                            0.12 * float(meta.get("finish_chain_signal", 0.0))
                        )
                    if action_key[3] is True:
                        trajectory_prior *= 1.0 + (
                            0.10 * float(meta.get("slot_attackability", 0.0))
                        )
                elif not previous_signal.result:
                    if same_player:
                        trajectory_prior *= 1.0 + (
                            0.12 * float(meta.get("public_bridge_signal", 0.0))
                        )
                    if same_slot:
                        trajectory_prior *= self._retry_after_failure_trajectory_prior(
                            previous_signal=previous_signal,
                            candidate_card=action_key[2],
                            slot_attackability=float(
                                meta.get("slot_attackability", 0.0)
                            ),
                            attack_window_signal=float(
                                meta.get("attack_window_signal", 0.0)
                            ),
                        )
            posterior_action_mass[action_key] = (
                max(self.EPSILON, action_probability)
                * (structured_fit ** self.JOINT_ACTION_STRUCTURED_BLEND)
                * (contextual_prior ** self.JOINT_ACTION_POSTERIOR_CONTEXT_BLEND)
                * (trajectory_prior ** self.JOINT_ACTION_POSTERIOR_TRAJECTORY_BLEND)
                * (sequence_prior ** self.JOINT_ACTION_SEQUENCE_POSTERIOR_BLEND)
            )
            metadata[action_key]["posterior_structured_fit"] = structured_fit
            metadata[action_key]["posterior_contextual_prior"] = contextual_prior
            metadata[action_key]["posterior_trajectory_prior"] = trajectory_prior
            metadata[action_key]["posterior_sequence_prior"] = sequence_prior
        total_posterior_mass = sum(posterior_action_mass.values())
        if total_posterior_mass <= 0.0:
            posterior_uniform_probability = 1.0 / float(
                max(1, len(posterior_action_mass))
            )
            posterior_probabilities = {
                action_key: posterior_uniform_probability
                for action_key in posterior_action_mass
            }
        else:
            posterior_probabilities = {
                action_key: posterior_mass / total_posterior_mass
                for action_key, posterior_mass in posterior_action_mass.items()
            }
        posterior_parametric_raw_scores: Dict[
            Tuple[str, int, Card, Optional[bool]],
            float,
        ] = {}
        for action_key, posterior_probability in posterior_probabilities.items():
            meta = metadata.get(action_key, {})
            posterior_parametric_features = self._parametric_action_feature_bundle(
                action_key=action_key,
                metadata=meta,
                base_probability=posterior_probability,
                sequence_profile=sequence_profile,
            )
            posterior_parametric_features.update(
                {
                    "posterior_structured_fit": self._normalize_action_feature_score(
                        float(meta.get("posterior_structured_fit", 1.0)),
                        radius=0.45,
                    ),
                    "posterior_contextual_prior": self._normalize_action_feature_score(
                        float(meta.get("posterior_contextual_prior", 1.0)),
                        radius=0.35,
                    ),
                    "posterior_trajectory_prior": self._normalize_action_feature_score(
                        float(meta.get("posterior_trajectory_prior", 1.0)),
                        radius=0.35,
                    ),
                    "posterior_sequence_prior": self._normalize_action_feature_score(
                        float(meta.get("posterior_sequence_prior", 1.0)),
                        radius=0.35,
                    ),
                }
            )
            posterior_parametric_logit = self._parametric_action_logit(
                posterior_parametric_features,
                weights=self.PARAMETRIC_POSTERIOR_FEATURE_WEIGHTS,
                bias=self.PARAMETRIC_POSTERIOR_BIAS,
            )
            posterior_parametric_raw_scores[action_key] = posterior_parametric_logit
            meta["posterior_parametric_features"] = posterior_parametric_features
            meta["posterior_parametric_logit"] = posterior_parametric_logit
        posterior_parametric_probabilities = self._softmax_probabilities(
            posterior_parametric_raw_scores,
            temperature=1.0,
        )
        blended_posterior_mass: Dict[
            Tuple[str, int, Card, Optional[bool]],
            float,
        ] = {}
        for action_key, posterior_probability in posterior_probabilities.items():
            posterior_parametric_probability = float(
                posterior_parametric_probabilities.get(action_key, self.EPSILON)
            )
            blended_posterior_mass[action_key] = (
                max(self.EPSILON, posterior_probability)
                ** (1.0 - self.JOINT_ACTION_POSTERIOR_PARAMETRIC_BLEND)
                * max(self.EPSILON, posterior_parametric_probability)
                ** self.JOINT_ACTION_POSTERIOR_PARAMETRIC_BLEND
            )
            metadata[action_key]["posterior_parametric_probability"] = (
                posterior_parametric_probability
            )
        total_blended_posterior_mass = sum(blended_posterior_mass.values())
        if total_blended_posterior_mass > 0.0:
            posterior_probabilities = {
                action_key: blended_mass / total_blended_posterior_mass
                for action_key, blended_mass in blended_posterior_mass.items()
            }
        posterior_player_probabilities: DefaultDict[str, float] = defaultdict(float)
        posterior_slot_probabilities: DefaultDict[str, Dict[int, float]] = defaultdict(
            dict
        )
        posterior_value_probabilities: DefaultDict[
            Tuple[str, int],
            Dict[Card, float],
        ] = defaultdict(dict)
        posterior_continue_probabilities: DefaultDict[
            Tuple[str, int, Card],
            Dict[Optional[bool], float],
        ] = defaultdict(dict)
        for action_key, posterior_probability in posterior_probabilities.items():
            player_id, slot_index, candidate_card, continued_turn = action_key
            posterior_player_probabilities[player_id] += posterior_probability
            posterior_slot_probabilities[player_id][slot_index] = (
                posterior_slot_probabilities[player_id].get(slot_index, 0.0)
                + posterior_probability
            )
            posterior_value_probabilities[(player_id, slot_index)][candidate_card] = (
                posterior_value_probabilities[(player_id, slot_index)].get(
                    candidate_card,
                    0.0,
                )
                + posterior_probability
            )
            posterior_continue_probabilities[(player_id, slot_index, candidate_card)][
                continued_turn
            ] = (
                posterior_continue_probabilities[
                    (player_id, slot_index, candidate_card)
                ].get(continued_turn, 0.0)
                + posterior_probability
            )
        refined_posterior_mass: Dict[
            Tuple[str, int, Card, Optional[bool]],
            float,
        ] = {}
        for action_key, posterior_probability in posterior_probabilities.items():
            player_id, slot_index, candidate_card, continued_turn = action_key
            player_mass = max(
                self.EPSILON,
                posterior_player_probabilities.get(player_id, self.EPSILON),
            )
            slot_mass = max(
                self.EPSILON,
                posterior_slot_probabilities.get(player_id, {}).get(
                    slot_index,
                    self.EPSILON,
                ),
            )
            value_mass = max(
                self.EPSILON,
                posterior_value_probabilities.get((player_id, slot_index), {}).get(
                    candidate_card,
                    self.EPSILON,
                ),
            )
            continue_mass = max(
                self.EPSILON,
                posterior_continue_probabilities.get(
                    (player_id, slot_index, candidate_card),
                    {},
                ).get(continued_turn, 1.0),
            )
            refined_posterior_mass[action_key] = (
                max(self.EPSILON, posterior_probability)
                * (player_mass ** 0.16)
                * (slot_mass ** 0.16)
                * (value_mass ** 0.18)
                * (continue_mass ** 0.10)
                * (
                    max(
                        self.EPSILON,
                        float(meta.get("sequence_probability", self.EPSILON)),
                    )
                    ** 0.10
                )
            )
            metadata[action_key]["posterior_player_mass"] = player_mass
            metadata[action_key]["posterior_slot_mass"] = slot_mass
            metadata[action_key]["posterior_value_mass"] = value_mass
            metadata[action_key]["posterior_continue_mass"] = continue_mass
        total_refined_posterior_mass = sum(refined_posterior_mass.values())
        if total_refined_posterior_mass > 0.0:
            posterior_probabilities = {
                action_key: refined_mass / total_refined_posterior_mass
                for action_key, refined_mass in refined_posterior_mass.items()
            }
        observed_key = (
            signal.target_player_id,
            signal.target_slot_index,
            signal.guessed_card,
            signal.continued_turn if signal.continued_turn is not None else None,
        )
        candidate_count = float(len(action_probabilities))
        uniform_probability = 1.0 / candidate_count
        observed_probability = max(
            self.EPSILON,
            action_probabilities.get(observed_key, self.EPSILON),
        )
        probability_advantage = observed_probability / max(
            self.EPSILON,
            uniform_probability,
        )
        weight = clamp(
            probability_advantage ** self.JOINT_ACTION_GENERATIVE_BLEND,
            self.EPSILON,
            self.JOINT_ACTION_GENERATIVE_MAX_WEIGHT,
        )
        posterior_uniform_probability = 1.0 / candidate_count
        observed_posterior_probability = max(
            self.EPSILON,
            posterior_probabilities.get(observed_key, self.EPSILON),
        )
        posterior_probability_advantage = observed_posterior_probability / max(
            self.EPSILON,
            posterior_uniform_probability,
        )
        posterior_weight = clamp(
            posterior_probability_advantage ** self.JOINT_ACTION_POSTERIOR_BLEND,
            self.EPSILON,
            self.JOINT_ACTION_POSTERIOR_MAX_WEIGHT,
        )
        ranked_actions = sorted(
            action_probabilities.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1], card_sort_key(item[0][2])),
        )
        posterior_ranked_actions = sorted(
            posterior_probabilities.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1], card_sort_key(item[0][2])),
        )
        observed_rank = float(
            next(
                (
                    index + 1
                    for index, (action_key, _) in enumerate(ranked_actions)
                    if action_key == observed_key
                ),
                len(ranked_actions),
            )
        )
        posterior_observed_rank = float(
            next(
                (
                    index + 1
                    for index, (action_key, _) in enumerate(posterior_ranked_actions)
                    if action_key == observed_key
                ),
                len(posterior_ranked_actions),
            )
        )
        posterior_rank_credit = 1.0 + (
            0.08
            * clamp(
                (
                    (candidate_count - posterior_observed_rank)
                    / max(1.0, candidate_count - 1.0)
                ),
                0.0,
                1.0,
            )
        )
        posterior_top_probability = float(
            posterior_ranked_actions[0][1] if posterior_ranked_actions else 0.0
        )
        posterior_second_probability = float(
            posterior_ranked_actions[1][1]
            if len(posterior_ranked_actions) > 1
            else 0.0
        )
        posterior_top_gap_signal = clamp(
            (
                posterior_top_probability - posterior_second_probability
            )
            / max(self.EPSILON, posterior_top_probability),
            0.0,
            1.0,
        )
        posterior_entropy = -sum(
            probability * log2(max(self.EPSILON, probability))
            for probability in posterior_probabilities.values()
        )
        normalized_posterior_entropy = clamp(
            posterior_entropy / max(1.0, log2(max(2.0, candidate_count))),
            0.0,
            1.0,
        )
        posterior_entropy_credit = 1.0 + (
            self.JOINT_ACTION_POSTERIOR_ENTROPY_CREDIT
            * (1.0 - normalized_posterior_entropy)
        )
        posterior_gap_credit = 1.0 + (
            self.JOINT_ACTION_POSTERIOR_GAP_CREDIT
            * posterior_top_gap_signal
            * clamp(
                (
                    (candidate_count - posterior_observed_rank)
                    / max(1.0, candidate_count - 1.0)
                ),
                0.0,
                1.0,
            )
        )
        posterior_weight = clamp(
            posterior_weight
            * posterior_rank_credit
            * posterior_gap_credit
            * posterior_entropy_credit,
            self.EPSILON,
            self.JOINT_ACTION_POSTERIOR_MAX_WEIGHT,
        )
        top_actions = []
        for action_key, probability in ranked_actions[:3]:
            meta = metadata.get(action_key, {})
            top_actions.append(
                {
                    "target_player_id": action_key[0],
                    "target_slot_index": action_key[1],
                    "guess_card": serialize_card(action_key[2]),
                    "continued_turn": action_key[3],
                    "probability": float(probability),
                    "conditional_probability": float(
                        meta.get("conditional_probability", probability)
                    ),
                    "prior_probability": float(
                        meta.get("prior_probability", probability)
                    ),
                    "structural_probability": float(
                        meta.get("structural_probability", probability)
                    ),
                    "base_probability": float(
                        meta.get("base_probability", probability)
                    ),
                    "parametric_probability": float(
                        meta.get("parametric_probability", probability)
                    ),
                    "parametric_logit": float(meta.get("parametric_logit", 0.0)),
                    "sequence_probability": float(
                        meta.get("sequence_probability", probability)
                    ),
                    "sequence_similarity": float(
                        meta.get("sequence_similarity", 1.0)
                    ),
                    "player_selection_score": float(
                        meta.get("player_selection_score", 0.0)
                    ),
                    "slot_selection_score": float(
                        meta.get("slot_selection_score", 0.0)
                    ),
                    "value_selection_score": float(
                        meta.get("value_selection_score", 0.0)
                    ),
                    "continue_score": float(meta.get("continue_score", 0.0)),
                }
            )
        posterior_top_actions = []
        for action_key, probability in posterior_ranked_actions[:3]:
            meta = metadata.get(action_key, {})
            posterior_top_actions.append(
                {
                    "target_player_id": action_key[0],
                    "target_slot_index": action_key[1],
                    "guess_card": serialize_card(action_key[2]),
                    "continued_turn": action_key[3],
                    "probability": float(probability),
                    "generative_probability": float(
                        action_probabilities.get(action_key, probability)
                    ),
                    "conditional_probability": float(
                        meta.get(
                            "conditional_probability",
                            action_probabilities.get(action_key, probability),
                        )
                    ),
                    "prior_probability": float(
                        meta.get(
                            "prior_probability",
                            action_probabilities.get(action_key, probability),
                        )
                    ),
                    "structural_probability": float(
                        meta.get(
                            "structural_probability",
                            action_probabilities.get(action_key, probability),
                        )
                    ),
                    "base_probability": float(
                        meta.get(
                            "base_probability",
                            action_probabilities.get(action_key, probability),
                        )
                    ),
                    "parametric_probability": float(
                        meta.get(
                            "parametric_probability",
                            action_probabilities.get(action_key, probability),
                        )
                    ),
                    "posterior_parametric_probability": float(
                        meta.get(
                            "posterior_parametric_probability",
                            probability,
                        )
                    ),
                    "parametric_logit": float(meta.get("parametric_logit", 0.0)),
                    "posterior_parametric_logit": float(
                        meta.get("posterior_parametric_logit", 0.0)
                    ),
                    "sequence_probability": float(
                        meta.get(
                            "sequence_probability",
                            action_probabilities.get(action_key, probability),
                        )
                    ),
                    "sequence_similarity": float(
                        meta.get("sequence_similarity", 1.0)
                    ),
                    "posterior_structured_fit": float(
                        meta.get("posterior_structured_fit", 1.0)
                    ),
                    "posterior_contextual_prior": float(
                        meta.get("posterior_contextual_prior", 1.0)
                    ),
                    "posterior_sequence_prior": float(
                        meta.get("posterior_sequence_prior", 1.0)
                    ),
                }
            )
        observed_meta = metadata.get(observed_key, {})
        result = {
            "candidate_count": candidate_count,
            "uniform_probability": uniform_probability,
            "probability": observed_probability,
            "conditional_probability": float(
                observed_meta.get("conditional_probability", observed_probability)
            ),
            "prior_probability": float(
                observed_meta.get("prior_probability", observed_probability)
            ),
            "structural_probability": float(
                observed_meta.get("structural_probability", observed_probability)
            ),
            "base_probability": float(
                observed_meta.get("base_probability", observed_probability)
            ),
            "parametric_probability": float(
                observed_meta.get("parametric_probability", observed_probability)
            ),
            "parametric_logit": float(observed_meta.get("parametric_logit", 0.0)),
            "sequence_probability": float(
                observed_meta.get("sequence_probability", observed_probability)
            ),
            "sequence_similarity": float(
                observed_meta.get("sequence_similarity", 1.0)
            ),
            "probability_advantage": probability_advantage,
            "weight": weight,
            "posterior_uniform_probability": posterior_uniform_probability,
            "posterior_probability": observed_posterior_probability,
            "posterior_probability_advantage": posterior_probability_advantage,
            "posterior_weight": posterior_weight,
            "posterior_parametric_probability": float(
                observed_meta.get(
                    "posterior_parametric_probability",
                    observed_posterior_probability,
                )
            ),
            "posterior_parametric_logit": float(
                observed_meta.get("posterior_parametric_logit", 0.0)
            ),
            "observed_rank": observed_rank,
            "posterior_observed_rank": posterior_observed_rank,
            "posterior_top_gap_signal": posterior_top_gap_signal,
            "posterior_entropy": posterior_entropy,
            "normalized_posterior_entropy": normalized_posterior_entropy,
            "posterior_gap_credit": posterior_gap_credit,
            "posterior_entropy_credit": posterior_entropy_credit,
            "player_selection_score": float(
                observed_meta.get("player_selection_score", 0.0)
            ),
            "slot_selection_score": float(
                observed_meta.get("slot_selection_score", 0.0)
            ),
            "value_selection_score": float(
                observed_meta.get("value_selection_score", 0.0)
            ),
            "continue_score": float(observed_meta.get("continue_score", 0.0)),
            "top_actions": top_actions,
            "posterior_structured_fit": float(
                observed_meta.get("posterior_structured_fit", 1.0)
            ),
            "posterior_contextual_prior": float(
                observed_meta.get("posterior_contextual_prior", 1.0)
            ),
            "posterior_sequence_prior": float(
                observed_meta.get("posterior_sequence_prior", 1.0)
            ),
            "sequence_profile_count": float(sequence_profile.get("count", 0.0)),
            "sequence_profile_total_weight": float(
                sequence_profile.get("total_weight", 0.0)
            ),
            "sequence_profile_stability": float(
                sequence_profile.get("stability", 0.0)
            ),
            "posterior_top_actions": posterior_top_actions,
        }
        if include_candidate_catalog:
            result["candidate_catalog"] = [
                {
                    "action_key": action_key,
                    "observed": action_key == observed_key,
                    "probability": float(action_probabilities.get(action_key, 0.0)),
                    "posterior_probability": float(
                        posterior_probabilities.get(action_key, 0.0)
                    ),
                    "parametric_features": dict(
                        metadata.get(action_key, {}).get("parametric_features", {})
                    ),
                    "posterior_parametric_features": dict(
                        metadata.get(action_key, {}).get(
                            "posterior_parametric_features",
                            {},
                        )
                    ),
                }
                for action_key in posterior_probabilities
            ]
        return result

    def _has_failure_switch_guess_continuity(
        self,
        previous_signal: GuessSignal,
        signal: GuessSignal,
    ) -> bool:
        if previous_signal.result or previous_signal.target_player_id == signal.target_player_id:
            return False

        previous_card = previous_signal.guessed_card
        current_card = signal.guessed_card
        if previous_card[0] != current_card[0]:
            return False

        previous_value = numeric_card_value(previous_card)
        current_value = numeric_card_value(current_card)
        if previous_value is None or current_value is None:
            return False
        return abs(previous_value - current_value) <= 2

    def _recent_target_chain_pressure(
        self,
        game_state: GameState,
        *,
        guesser_id: str,
        target_player_id: str,
    ) -> float:
        chain_score = 0.0
        recency_weight = 1.0
        for action in reversed(getattr(game_state, "actions", ())):
            if getattr(action, "action_type", None) != "guess":
                continue
            if getattr(action, "guesser_id", None) != guesser_id:
                continue
            if getattr(action, "target_player_id", None) != target_player_id:
                continue
            if getattr(action, "result", False):
                chain_score += 0.65 * recency_weight
                if getattr(action, "continued_turn", None) is True:
                    chain_score += 0.20 * recency_weight
            if getattr(action, "revealed_player_id", None) == target_player_id:
                chain_score += 0.35 * recency_weight
            recency_weight *= 0.62
            if recency_weight < 0.15:
                break
        return clamp(chain_score, 0.0, 1.0)

    def _score_target_slot_selection(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
    ) -> float:
        candidate_scores: Dict[int, float] = {}
        for slot in game_state.resolved_ordered_slots(signal.target_player_id):
            if slot.known_card() is not None:
                continue
            score = self._slot_attackability(
                game_state,
                hypothesis_by_player,
                signal.target_player_id,
                slot.slot_index,
            )
            if score > 0.0:
                candidate_scores[slot.slot_index] = score

        if len(candidate_scores) <= 1:
            return 1.0

        chosen_score = candidate_scores.get(signal.target_slot_index)
        if chosen_score is None:
            return 1.0

        best_score = max(candidate_scores.values())
        weight = 1.0
        if chosen_score >= best_score - self.EPSILON:
            weight *= self.TARGET_SLOT_BEST_MATCH_BONUS
        elif chosen_score >= best_score * 0.85:
            weight *= self.TARGET_SLOT_CLOSE_MATCH_BONUS
        else:
            weight *= self.TARGET_SLOT_WEAK_CHOICE_PENALTY

        if self._has_prior_failed_guess_on_slot(game_state, signal):
            weight *= self.TARGET_SLOT_RETRY_AFTER_FAILURE_BONUS
        if self._is_adjacent_slot_follow_after_confident_hit(game_state, signal):
            weight *= self.TARGET_SLOT_CONFIDENT_ADJACENT_FOLLOW_BONUS
        if self._is_adjacent_slot_probe_after_failure(game_state, signal):
            weight *= self.TARGET_SLOT_FAILURE_ADJACENT_PROBE_BONUS
        attack_window_pressure = self._slot_attack_window_pressure(
            game_state,
            signal.target_player_id,
            signal.target_slot_index,
        )
        if attack_window_pressure > 0.0:
            weight *= 1.0 + (
                (self.TARGET_SLOT_ATTACK_WINDOW_BONUS - 1.0)
                * attack_window_pressure
            )
        return weight

    def _is_adjacent_slot_follow_after_confident_hit(
        self,
        game_state: GameState,
        signal: GuessSignal,
    ) -> bool:
        previous_signal = self._previous_guess_signal(game_state, signal)
        if previous_signal is None:
            return False
        if not previous_signal.result or not previous_signal.continued_turn:
            return False
        if previous_signal.target_player_id != signal.target_player_id:
            return False
        return abs(previous_signal.target_slot_index - signal.target_slot_index) == 1

    def _is_adjacent_slot_probe_after_failure(
        self,
        game_state: GameState,
        signal: GuessSignal,
    ) -> bool:
        previous_signal = self._previous_guess_signal(game_state, signal)
        if previous_signal is None:
            return False
        if previous_signal.result:
            return False
        if previous_signal.target_player_id != signal.target_player_id:
            return False
        return abs(previous_signal.target_slot_index - signal.target_slot_index) == 1

    def _retry_after_failure_trajectory_prior(
        self,
        *,
        previous_signal: GuessSignal,
        candidate_card: Card,
        slot_attackability: float,
        attack_window_signal: float,
    ) -> float:
        previous_card = previous_signal.guessed_card
        if previous_card[0] != candidate_card[0]:
            return self.JOINT_ACTION_POSTERIOR_RETRY_WIDE_PENALTY

        previous_numeric = numeric_card_value(previous_card)
        candidate_numeric = numeric_card_value(candidate_card)
        if previous_numeric is None or candidate_numeric is None:
            return 1.0

        step = abs(candidate_numeric - previous_numeric)
        if step == 0:
            return self.JOINT_ACTION_POSTERIOR_RETRY_STALLED_PENALTY
        if step > 2:
            return self.JOINT_ACTION_POSTERIOR_RETRY_WIDE_PENALTY

        progression_signal = 1.0 if step == 1 else 0.55
        return 1.0 + (
            self.JOINT_ACTION_POSTERIOR_RETRY_PROGRESSION_CREDIT
            * progression_signal
        ) + (
            self.JOINT_ACTION_POSTERIOR_RETRY_ATTACK_WINDOW_CREDIT
            * clamp(
                0.6 * float(slot_attackability) + 0.4 * float(attack_window_signal),
                0.0,
                1.0,
            )
        )

    def _score_target_slot(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
        target_card: Card,
    ) -> float:
        guessed_card = signal.guessed_card
        weight = 1.0

        try:
            slot = game_state.get_slot(signal.target_player_id, signal.target_slot_index)
        except ValueError:
            return 1.0

        known_slot_color = getattr(slot, "color", None)
        if known_slot_color is not None and guessed_card[0] != known_slot_color:
            weight *= self.WRONG_COLOR_SLOT_PENALTY

        guessed_numeric = numeric_card_value(guessed_card)
        target_numeric = numeric_card_value(target_card)
        if guessed_numeric is None or target_numeric is None:
            return weight

        low, high, width = self._slot_numeric_interval(
            game_state,
            hypothesis_by_player,
            signal.target_player_id,
            signal.target_slot_index,
        )

        if low is not None and high is not None:
            if low <= guessed_numeric <= high:
                weight *= self.TARGET_IN_INTERVAL_BONUS
                if width <= 2:
                    weight *= self.TARGET_NARROW_INTERVAL_BONUS
            else:
                weight *= self.TARGET_OUTSIDE_INTERVAL_PENALTY

        distance = abs(guessed_numeric - target_numeric)
        if distance == 0:
            return weight
        if distance == 1:
            return weight * self.TARGET_NEIGHBOR_BONUS
        if distance <= 2:
            return weight * self.TARGET_CLOSE_BONUS
        return weight * (self.TARGET_FAR_PENALTY ** max(1, distance - 2))

    def _score_target_value_selection(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
        target_card: Card,
        guesser_cards: Sequence[Card],
    ) -> float:
        return self._value_selection_breakdown(
            game_state,
            hypothesis_by_player,
            signal,
            target_card,
            guesser_cards,
        )["total_weight"]

    def _value_selection_breakdown(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
        target_card: Card,
        guesser_cards: Sequence[Card],
    ) -> Dict[str, Any]:
        guessed_numeric = numeric_card_value(signal.guessed_card)
        target_numeric = numeric_card_value(target_card)
        if guessed_numeric is None or target_numeric is None:
            neutral = {"weight": 1.0, "reason": "neutral"}
            return {
                "total_weight": 1.0,
                "signal_tags": [],
                "dominant_signal": {
                    "source": "neutral",
                    "reason": "neutral",
                    "weight": 1.0,
                },
                "progressive": neutral,
                "anchor": neutral,
                "boundary": neutral,
            }

        low, high, width = self._slot_numeric_interval(
            game_state,
            hypothesis_by_player,
            signal.target_player_id,
            signal.target_slot_index,
        )
        progressive = self._progressive_value_component(
            guessed_numeric,
            target_numeric,
            self._prior_failed_numeric_guesses_on_slot(game_state, signal),
            self._previous_confident_success_numeric_guess(game_state, signal),
        )
        anchor = self._same_color_anchor_value_component(
            guessed_numeric,
            self._same_color_numeric_values(guesser_cards, signal.guessed_card[0]),
        )
        boundary = self._local_boundary_value_component(
            guessed_numeric,
            low,
            high,
            width,
        )
        signal_tags = [
            component["reason"]
            for component in (progressive, anchor, boundary)
            if component["reason"] != "neutral"
        ]
        return {
            "total_weight": progressive["weight"] * anchor["weight"] * boundary["weight"],
            "signal_tags": signal_tags,
            "dominant_signal": self._dominant_signal(
                {
                    "progressive": progressive,
                    "same_color_anchor": anchor,
                    "local_boundary": boundary,
                }
            ),
            "progressive": progressive,
            "anchor": anchor,
            "boundary": boundary,
        }

    def _progressive_value_component(
        self,
        guessed_numeric: int,
        target_numeric: int,
        prior_failed_values: Sequence[int],
        previous_confident_value: Optional[int],
    ) -> Dict[str, Any]:
        component: Dict[str, Any] = {
            "weight": 1.0,
            "reason": "neutral",
            "latest_failed_value": None,
            "expected_progress": None,
            "direction": 0,
            "previous_confident_value": previous_confident_value,
        }
        if not prior_failed_values:
            if previous_confident_value is None:
                return component
            delta_from_confident = abs(guessed_numeric - previous_confident_value)
            if delta_from_confident == 1:
                component["weight"] = self.TARGET_VALUE_CONFIDENT_CHAIN_LOCAL_STEP_BONUS
                component["reason"] = "confident_chain_local_step"
            elif delta_from_confident >= 4:
                component["weight"] = self.TARGET_VALUE_CONFIDENT_CHAIN_JUMP_PENALTY
                component["reason"] = "confident_chain_jump"
            return component

        latest_failed = prior_failed_values[-1]
        direction = 0
        if target_numeric > latest_failed:
            direction = 1
        elif target_numeric < latest_failed:
            direction = -1

        component["latest_failed_value"] = latest_failed
        component["direction"] = direction
        if direction == 0:
            return component

        expected_progress = latest_failed + direction
        delta = guessed_numeric - latest_failed
        component["expected_progress"] = expected_progress
        if guessed_numeric == expected_progress:
            component["weight"] = self.TARGET_VALUE_PROGRESSIVE_STEP_BONUS
            component["reason"] = "progressive_step"
        elif delta == 0:
            component["weight"] = self.TARGET_VALUE_STALLED_PENALTY
            component["reason"] = "stalled_after_failure"
        elif delta * direction > 0:
            component["weight"] = self.TARGET_VALUE_DIRECTIONAL_BONUS
            component["reason"] = "directional_progress"
        else:
            component["weight"] = self.TARGET_VALUE_WRONG_DIRECTION_PENALTY
            component["reason"] = "wrong_direction"
        return component

    def _previous_confident_success_numeric_guess(
        self,
        game_state: GameState,
        signal: GuessSignal,
    ) -> Optional[int]:
        previous_signal = self._previous_guess_signal(game_state, signal)
        if previous_signal is None:
            return None
        if not previous_signal.result or not previous_signal.continued_turn:
            return None
        if previous_signal.target_player_id != signal.target_player_id:
            return None
        if previous_signal.guessed_card[0] != signal.guessed_card[0]:
            return None
        return numeric_card_value(previous_signal.guessed_card)

    def explain_guess_signals(
        self,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        game_state: GameState,
    ) -> List[Dict[str, Any]]:
        explanations: List[Dict[str, Any]] = []
        for player_id in sorted(guess_signals_by_player):
            for signal in guess_signals_by_player[player_id]:
                explanations.append(
                    self.explain_signal(
                        hypothesis_by_player,
                        game_state,
                        signal,
                    )
                )
        return explanations

    def explain_signal(
        self,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        game_state: GameState,
        signal: GuessSignal,
    ) -> Dict[str, Any]:
        target_card = self._resolve_slot_card(
            game_state,
            hypothesis_by_player,
            signal.target_player_id,
            signal.target_slot_index,
        )
        base_explanation = {
            "action_index": signal.action_index,
            "guesser_id": signal.guesser_id,
            "target_player_id": signal.target_player_id,
            "target_slot_index": signal.target_slot_index,
            "guessed_card": serialize_card(signal.guessed_card),
            "result": signal.result,
            "continued_turn": signal.continued_turn,
        }
        if target_card is None:
            return {
                **base_explanation,
                "hypothesis_target_card": None,
                "total_weight": 1.0,
                "component_weights": {},
                "joint_action": {
                    "weight": 1.0,
                    "signal": 0.0,
                    "penalty_signal": 0.0,
                    "local_alignment": 0.0,
                    "propagated_alignment": 0.0,
                },
                "joint_action_probability": {
                    "player_probability": 1.0,
                    "slot_probability": 1.0,
                    "value_probability": 1.0,
                    "continue_probability": 1.0,
                    "joint_probability": 1.0,
                },
                "joint_action_generation": {
                    "candidate_space": 1.0,
                    "uniform_probability": 1.0,
                    "probability_advantage": 1.0,
                    "generative_prior": 1.0,
                    "weight": 1.0,
                },
                "joint_action_generative_probability": {
                    "candidate_count": 1.0,
                    "uniform_probability": 1.0,
                    "probability": 1.0,
                    "probability_advantage": 1.0,
                    "weight": 1.0,
                    "posterior_uniform_probability": 1.0,
                    "posterior_probability": 1.0,
                    "posterior_probability_advantage": 1.0,
                    "posterior_weight": 1.0,
                    "observed_rank": 1.0,
                    "posterior_observed_rank": 1.0,
                    "top_actions": [],
                    "posterior_top_actions": [],
                },
                "joint_action_posterior": {
                    "uniform_probability": 1.0,
                    "probability": 1.0,
                    "probability_advantage": 1.0,
                    "weight": 1.0,
                    "observed_rank": 1.0,
                    "top_actions": [],
                },
                "value_selection": {
                    "total_weight": 1.0,
                    "signal_tags": [],
                    "dominant_signal": {
                        "source": "neutral",
                        "reason": "neutral",
                        "weight": 1.0,
                    },
                },
            }

        guesser_cards = self._player_cards(
            game_state,
            signal.guesser_id,
            hypothesis_by_player.get(signal.guesser_id, {}),
        )
        self_hand_weight = self._score_self_hand(guesser_cards, signal.guessed_card)
        target_player_weight = self._score_target_player_selection(
            game_state,
            hypothesis_by_player,
            signal,
        )
        target_slot_selection_weight = self._score_target_slot_selection(
            game_state,
            hypothesis_by_player,
            signal,
        )
        value_selection = self._value_selection_breakdown(
            game_state,
            hypothesis_by_player,
            signal,
            target_card,
            guesser_cards,
        )
        target_slot_weight = self._score_target_slot(
            game_state,
            hypothesis_by_player,
            signal,
            target_card,
        )
        continue_weight = self._score_continue_decision(
            game_state,
            hypothesis_by_player,
            signal,
        )
        joint_action_probability = self._joint_action_probability_breakdown(
            game_state,
            hypothesis_by_player,
            signal,
            target_card=target_card,
            guesser_cards=guesser_cards,
        )
        joint_action_generation = self._joint_action_generation_breakdown(
            game_state,
            hypothesis_by_player,
            signal,
            joint_action_probability=joint_action_probability,
        )
        joint_action_generative_probability = (
            self._joint_action_generative_probability_breakdown(
                game_state,
                hypothesis_by_player,
                signal,
                guesser_cards=guesser_cards,
            )
        )
        joint_action = self._joint_action_fit_breakdown(
            game_state,
            hypothesis_by_player,
            signal,
            target_player_weight=target_player_weight,
            target_slot_selection_weight=target_slot_selection_weight,
            value_selection=value_selection,
            continue_weight=continue_weight,
        )
        component_weights = {
            "self_hand": self_hand_weight,
            "target_player_selection": target_player_weight,
            "target_slot_selection": target_slot_selection_weight,
            "target_value_selection": value_selection["total_weight"],
            "target_slot_fit": target_slot_weight,
            "continue_decision_fit": continue_weight,
            "joint_action_probability": joint_action_probability["joint_probability"],
            "joint_action_generation": joint_action_generation["weight"],
            "joint_action_posterior": (
                joint_action_generative_probability.get(
                    "posterior_weight",
                    joint_action_generative_probability["weight"],
                )
            ),
            "joint_action_fit": joint_action["weight"],
        }
        total_weight = 1.0
        for weight in component_weights.values():
            total_weight *= weight

        return {
            **base_explanation,
            "hypothesis_target_card": serialize_card(target_card),
            "total_weight": max(self.EPSILON, total_weight),
            "component_weights": component_weights,
            "joint_action": joint_action,
            "joint_action_probability": joint_action_probability,
            "joint_action_generation": joint_action_generation,
            "joint_action_generative_probability": (
                joint_action_generative_probability
            ),
            "joint_action_posterior": {
                "uniform_probability": float(
                    joint_action_generative_probability.get(
                        "posterior_uniform_probability",
                        joint_action_generative_probability.get(
                            "uniform_probability",
                            1.0,
                        ),
                    )
                ),
                "probability": float(
                    joint_action_generative_probability.get(
                        "posterior_probability",
                        joint_action_generative_probability.get("probability", 1.0),
                    )
                ),
                "probability_advantage": float(
                    joint_action_generative_probability.get(
                        "posterior_probability_advantage",
                        joint_action_generative_probability.get(
                            "probability_advantage",
                            1.0,
                        ),
                    )
                ),
                "weight": float(
                    joint_action_generative_probability.get(
                        "posterior_weight",
                        joint_action_generative_probability.get("weight", 1.0),
                    )
                ),
                "observed_rank": float(
                    joint_action_generative_probability.get(
                        "posterior_observed_rank",
                        joint_action_generative_probability.get("observed_rank", 1.0),
                    )
                ),
                "top_gap_signal": float(
                    joint_action_generative_probability.get(
                        "posterior_top_gap_signal",
                        0.0,
                    )
                ),
                "normalized_entropy": float(
                    joint_action_generative_probability.get(
                        "normalized_posterior_entropy",
                        1.0,
                    )
                ),
                "gap_credit": float(
                    joint_action_generative_probability.get(
                        "posterior_gap_credit",
                        1.0,
                    )
                ),
                "entropy_credit": float(
                    joint_action_generative_probability.get(
                        "posterior_entropy_credit",
                        1.0,
                    )
                ),
                "top_actions": list(
                    joint_action_generative_probability.get(
                        "posterior_top_actions",
                        joint_action_generative_probability.get("top_actions", []),
                    )
                ),
            },
            "value_selection": value_selection,
        }

    def describe_candidate_value_signal(
        self,
        *,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        guesser_id: str,
        target_player_id: str,
        target_slot_index: int,
        guessed_card: Card,
    ) -> Dict[str, Any]:
        guessed_numeric = numeric_card_value(guessed_card)
        if guessed_numeric is None:
            neutral = {"weight": 1.0, "reason": "neutral"}
            return {
                "signal_tags": [],
                "dominant_signal": {
                    "source": "neutral",
                    "reason": "neutral",
                    "weight": 1.0,
                },
                "progressive": neutral,
                "anchor": neutral,
                "boundary": neutral,
            }

        guesser_cards = self._player_cards(
            game_state,
            guesser_id,
            hypothesis_by_player.get(guesser_id, {}),
        )
        low, high, width = self._slot_numeric_interval(
            game_state,
            hypothesis_by_player,
            target_player_id,
            target_slot_index,
        )
        progressive = self._candidate_progressive_probe_component(
            game_state,
            guesser_id=guesser_id,
            target_player_id=target_player_id,
            target_slot_index=target_slot_index,
            guessed_numeric=guessed_numeric,
        )
        anchor = self._same_color_anchor_value_component(
            guessed_numeric,
            self._same_color_numeric_values(guesser_cards, guessed_card[0]),
        )
        boundary = self._local_boundary_value_component(
            guessed_numeric,
            low,
            high,
            width,
        )
        signal_tags = [
            component["reason"]
            for component in (progressive, anchor, boundary)
            if component["reason"] != "neutral"
        ]
        return {
            "signal_tags": signal_tags,
            "dominant_signal": self._dominant_signal(
                {
                    "progressive": progressive,
                    "same_color_anchor": anchor,
                    "local_boundary": boundary,
                }
            ),
            "progressive": progressive,
            "anchor": anchor,
            "boundary": boundary,
        }

    def _same_color_anchor_value_component(
        self,
        guessed_numeric: int,
        same_color_values: Sequence[int],
    ) -> Dict[str, Any]:
        component: Dict[str, Any] = {
            "weight": 1.0,
            "reason": "neutral",
            "same_color_values": list(same_color_values),
            "left_anchor": None,
            "right_anchor": None,
            "anchor_gap": None,
            "nearest_anchor_distance": None,
        }
        if not same_color_values:
            return component

        nearest_anchor_distance = min(abs(guessed_numeric - value) for value in same_color_values)
        component["nearest_anchor_distance"] = nearest_anchor_distance
        if nearest_anchor_distance >= 4:
            component["weight"] *= 0.97
            component["reason"] = "far_from_same_color_anchor"

        left_anchor = max((value for value in same_color_values if value < guessed_numeric), default=None)
        right_anchor = min((value for value in same_color_values if value > guessed_numeric), default=None)
        component["left_anchor"] = left_anchor
        component["right_anchor"] = right_anchor

        if left_anchor is None or right_anchor is None:
            return component

        anchor_gap = right_anchor - left_anchor
        component["anchor_gap"] = anchor_gap
        midpoint = (left_anchor + right_anchor) / 2.0
        if anchor_gap == 2 and guessed_numeric == left_anchor + 1:
            component["weight"] *= self.TARGET_VALUE_SANDWICH_EXACT_BONUS
            component["reason"] = "same_color_sandwich_exact"
        elif 2 < anchor_gap <= 4 and left_anchor < guessed_numeric < right_anchor:
            component["weight"] *= self.TARGET_VALUE_SANDWICH_FILL_BONUS
            component["reason"] = "same_color_gap_fill"
        elif anchor_gap >= 6 and abs(guessed_numeric - midpoint) <= 1.0:
            component["weight"] *= self.TARGET_VALUE_WIDE_GAP_CENTER_BONUS
            component["reason"] = "same_color_wide_gap_center"
        return component

    def _local_boundary_value_component(
        self,
        guessed_numeric: int,
        low: Optional[int],
        high: Optional[int],
        width: int,
    ) -> Dict[str, Any]:
        component: Dict[str, Any] = {
            "weight": 1.0,
            "reason": "neutral",
            "low": low,
            "high": high,
            "width": width,
        }
        if low is None and high is None:
            return component

        if low is not None and high is not None:
            if width <= 3 and guessed_numeric in {low + 1, high - 1}:
                component["weight"] *= self.TARGET_VALUE_NARROW_BOUNDARY_PROBE_BONUS
                component["reason"] = "narrow_boundary_probe"
            elif width >= 6:
                midpoint = (low + high) / 2.0
                component["midpoint"] = midpoint
                if abs(guessed_numeric - midpoint) <= 1.0:
                    component["weight"] *= self.TARGET_VALUE_WIDE_GAP_CENTER_BONUS
                    component["reason"] = "wide_gap_center_probe"
                elif guessed_numeric in {low + 1, high - 1}:
                    component["weight"] *= self.TARGET_VALUE_WIDE_GAP_EDGE_PENALTY
                    component["reason"] = "wide_gap_edge_hug"
            return component

        if low is not None and guessed_numeric == low + 1:
            component["weight"] *= self.TARGET_VALUE_NARROW_BOUNDARY_PROBE_BONUS
            component["reason"] = "single_sided_low_boundary_probe"
        if high is not None and guessed_numeric == high - 1:
            component["weight"] *= self.TARGET_VALUE_NARROW_BOUNDARY_PROBE_BONUS
            component["reason"] = "single_sided_high_boundary_probe"
        return component

    def _dominant_signal(
        self,
        components_by_source: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        dominant_source = "neutral"
        dominant_reason = "neutral"
        dominant_weight = 1.0
        dominant_strength = 0.0

        for source, component in components_by_source.items():
            weight = float(component.get("weight", 1.0))
            strength = abs(weight - 1.0)
            if strength <= dominant_strength + self.EPSILON:
                continue
            dominant_source = source
            dominant_reason = str(component.get("reason", "neutral"))
            dominant_weight = weight
            dominant_strength = strength

        return {
            "source": dominant_source,
            "reason": dominant_reason,
            "weight": dominant_weight,
        }

    def _candidate_progressive_probe_component(
        self,
        game_state: GameState,
        *,
        guesser_id: str,
        target_player_id: str,
        target_slot_index: int,
        guessed_numeric: int,
    ) -> Dict[str, Any]:
        component: Dict[str, Any] = {
            "weight": 1.0,
            "reason": "neutral",
            "latest_failed_value": None,
        }
        latest_failed = self._latest_failed_numeric_guess_on_slot(
            game_state,
            guesser_id=guesser_id,
            target_player_id=target_player_id,
            target_slot_index=target_slot_index,
        )
        if latest_failed is None:
            return component

        component["latest_failed_value"] = latest_failed
        delta = guessed_numeric - latest_failed
        if abs(delta) == 1:
            component["weight"] = 1.05
            component["reason"] = "retry_step_probe"
        elif abs(delta) == 2:
            component["weight"] = 1.02
            component["reason"] = "retry_directional_probe"
        elif delta == 0:
            component["weight"] = 0.97
            component["reason"] = "retry_stalled_probe"
        return component

    def _same_color_numeric_values(
        self,
        player_cards: Sequence[Card],
        color: str,
    ) -> List[int]:
        values = sorted(
            numeric_card_value(card)
            for card in player_cards
            if card[0] == color and numeric_card_value(card) is not None
        )
        return [value for value in values if value is not None]

    def _score_continue_decision(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
    ) -> float:
        if not signal.result or signal.continued_turn is None:
            return 1.0

        attackability = self.estimate_attackability(
            game_state,
            hypothesis_by_player,
            acting_player_id=signal.guesser_id,
            exclude_slot=slot_key(signal.target_player_id, signal.target_slot_index),
        )
        target_followup_attackability = self._player_attackability(
            game_state,
            hypothesis_by_player,
            signal.target_player_id,
            exclude_slot=slot_key(signal.target_player_id, signal.target_slot_index),
        )
        self_exposure_profile = self._public_hand_exposure_profile(
            game_state,
            signal.guesser_id,
        )
        failure_recovery_signal = self._continue_failure_recovery_signal(
            game_state,
            hypothesis_by_player,
            signal,
        )
        target_finish_chain_signal = self._slot_finish_chain_pressure_after_hit(
            game_state,
            signal.target_player_id,
            signal.target_slot_index,
        )
        target_chain_pressure = self._recent_target_chain_pressure(
            game_state,
            guesser_id=signal.guesser_id,
            target_player_id=signal.target_player_id,
        )
        joint_collapse_signal = self._continue_joint_collapse_signal(
            game_state,
            signal,
        )
        if signal.continued_turn:
            weight = (
                self.CONTINUE_HIGH_ATTACKABILITY_BONUS
                if attackability >= self.ATTACKABILITY_TIGHT_THRESHOLD
                else self.CONTINUE_LOW_ATTACKABILITY_PENALTY
            )
            weight *= 1.0 - (
                (1.0 - self.CONTINUE_SELF_EXPOSURE_PENALTY)
                * self_exposure_profile["total_exposure"]
            )
            weight *= 1.0 - (
                (1.0 - self.CONTINUE_NEW_DRAWN_EXPOSURE_PENALTY)
                * self_exposure_profile["newly_drawn_exposure"]
            )
            weight *= 1.0 - (
                (1.0 - self.CONTINUE_FINISH_FRAGILITY_PENALTY)
                * self_exposure_profile["finish_fragility"]
            )
            if target_followup_attackability >= self.ATTACKABILITY_TIGHT_THRESHOLD:
                weight *= self.CONTINUE_TARGET_FOLLOWUP_BONUS
            if self._remaining_hidden_on_target_after_hit(game_state, signal) <= 1:
                weight *= self.CONTINUE_TARGET_FINISH_BONUS
            weight *= 1.0 + (
                (self.CONTINUE_TARGET_FINISH_CHAIN_BONUS - 1.0)
                * target_finish_chain_signal
            )
            weight *= 1.0 + (
                (self.CONTINUE_TARGET_CHAIN_BONUS - 1.0)
                * target_chain_pressure
            )
            weight *= 1.0 + (
                (self.CONTINUE_FAILURE_RECOVERY_BONUS - 1.0)
                * failure_recovery_signal
            )
            weight *= 1.0 + (
                (self.CONTINUE_JOINT_COLLAPSE_BONUS - 1.0)
                * joint_collapse_signal
            )
            return weight

        weight = (
            self.STOP_LOW_ATTACKABILITY_BONUS
            if attackability < self.ATTACKABILITY_TIGHT_THRESHOLD
            else self.STOP_HIGH_ATTACKABILITY_PENALTY
        )
        weight *= 1.0 + (
            (self.STOP_SELF_EXPOSURE_BONUS - 1.0)
            * self_exposure_profile["total_exposure"]
        )
        weight *= 1.0 + (
            (self.STOP_NEW_DRAWN_EXPOSURE_BONUS - 1.0)
            * self_exposure_profile["newly_drawn_exposure"]
        )
        weight *= 1.0 + (
            (self.STOP_FINISH_FRAGILITY_BONUS - 1.0)
            * self_exposure_profile["finish_fragility"]
        )
        if target_followup_attackability >= self.ATTACKABILITY_TIGHT_THRESHOLD:
            weight *= self.STOP_TARGET_FOLLOWUP_PENALTY
        if self._remaining_hidden_on_target_after_hit(game_state, signal) <= 1:
            weight *= self.STOP_TARGET_FINISH_PENALTY
        weight *= 1.0 - (
            (1.0 - self.STOP_TARGET_FINISH_CHAIN_PENALTY)
            * target_finish_chain_signal
        )
        weight *= 1.0 - (
            (1.0 - self.STOP_TARGET_CHAIN_PENALTY)
            * target_chain_pressure
        )
        weight *= 1.0 - (
            (1.0 - self.STOP_FAILURE_RECOVERY_PENALTY)
            * failure_recovery_signal
        )
        weight *= 1.0 - (
            (1.0 - self.STOP_JOINT_COLLAPSE_PENALTY)
            * joint_collapse_signal
        )
        return weight

    def _continue_failure_recovery_signal(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
    ) -> float:
        best_recovery_signal = 0.0
        for player_id in game_state.players:
            if player_id == signal.guesser_id:
                continue
            exclude_slot = (
                slot_key(signal.target_player_id, signal.target_slot_index)
                if player_id == signal.target_player_id
                else None
            )
            attackability = self._player_attackability(
                game_state,
                hypothesis_by_player,
                player_id,
                exclude_slot=exclude_slot,
            )
            failed_guess_pressure = self._recent_failed_guess_pressure(
                game_state,
                player_id,
            )
            best_recovery_signal = max(
                best_recovery_signal,
                attackability * failed_guess_pressure,
            )
        return clamp(best_recovery_signal, 0.0, 1.0)

    def _continue_joint_collapse_signal(
        self,
        game_state: GameState,
        signal: GuessSignal,
    ) -> float:
        target_collapse = self._recent_player_collapse_streak_pressure(
            game_state,
            signal.target_player_id,
        )
        global_collapse = self._global_public_collapse_pressure(game_state)
        global_propagation = self._global_public_propagation_pressure(game_state)
        return clamp(
            (0.45 * target_collapse)
            + (0.25 * global_collapse)
            + (0.30 * global_propagation),
            0.0,
            1.0,
        )

    def _public_hand_exposure_profile(
        self,
        game_state: GameState,
        player_id: str,
    ) -> Dict[str, float]:
        try:
            public_slots = game_state.get_player(player_id).ordered_slots()
        except ValueError:
            return {
                "total_exposure": 0.0,
                "newly_drawn_exposure": 0.0,
                "finish_fragility": 0.0,
            }

        slot_exposures: List[float] = []
        newly_drawn_exposure = 0.0
        for slot in public_slots:
            if slot.is_revealed:
                continue
            slot_exposure = self._public_slot_exposure(public_slots, slot.slot_index)
            slot_exposures.append(slot_exposure)
            if slot.is_newly_drawn:
                newly_drawn_exposure = max(newly_drawn_exposure, slot_exposure)

        if not slot_exposures:
            return {
                "total_exposure": 0.0,
                "newly_drawn_exposure": 0.0,
                "finish_fragility": 0.0,
            }

        slot_exposures.sort(reverse=True)
        total_exposure = slot_exposures[0]
        if len(slot_exposures) >= 2:
            total_exposure += (
                self.PUBLIC_SELF_EXPOSURE_SECONDARY_BLEND * slot_exposures[1]
            )
        finish_fragility = clamp(
            (
                (self.PUBLIC_SELF_EXPOSURE_FINISH_REFERENCE - float(len(slot_exposures)))
                / self.PUBLIC_SELF_EXPOSURE_FINISH_REFERENCE
            ) * slot_exposures[0],
            0.0,
            1.0,
        )
        return {
            "total_exposure": clamp(total_exposure, 0.0, 1.0),
            "newly_drawn_exposure": clamp(newly_drawn_exposure, 0.0, 1.0),
            "finish_fragility": finish_fragility,
        }

    def _public_slot_exposure(
        self,
        public_slots: Sequence[CardSlot],
        slot_index: int,
    ) -> float:
        slot_by_index = {slot.slot_index: slot for slot in public_slots}
        slot = slot_by_index.get(slot_index)
        if slot is None or slot.is_revealed:
            return 0.0

        low, high, width = self._public_slot_interval(public_slots, slot_index)
        exposure = (1.12 if getattr(slot, "color", None) is not None else 1.0) / max(
            1.0,
            float(width + 1),
        )
        if low is not None and high is not None and width <= 2:
            exposure *= 1.06
        if (
            getattr(slot, "color", None) is not None
            and (
                (low is None and high is not None and high <= 4)
                or (high is None and low is not None and low >= (MAX_CARD_VALUE - 4))
            )
        ):
            exposure *= 1.08
        anchor_match_count = self._public_slot_same_color_anchor_count(
            public_slots,
            slot_index,
            getattr(slot, "color", None),
        )
        if anchor_match_count >= 2:
            exposure *= self.PUBLIC_SELF_EXPOSURE_DOUBLE_COLOR_ANCHOR_BONUS
        elif anchor_match_count == 1:
            exposure *= self.PUBLIC_SELF_EXPOSURE_SAME_COLOR_ANCHOR_BONUS
        if slot.is_newly_drawn:
            exposure *= 1.18
        return exposure

    def _public_slot_same_color_anchor_count(
        self,
        public_slots: Sequence[CardSlot],
        slot_index: int,
        slot_color: Optional[str],
    ) -> int:
        if slot_color not in CARD_COLORS:
            return 0

        ordered_slots = sorted(public_slots, key=lambda slot: slot.slot_index)
        index_by_slot = {slot.slot_index: idx for idx, slot in enumerate(ordered_slots)}
        order_index = index_by_slot.get(slot_index)
        if order_index is None:
            return 0

        same_color_anchor_count = 0
        cursor = order_index - 1
        while cursor >= 0:
            left_slot = ordered_slots[cursor]
            if left_slot.is_revealed and left_slot.color in CARD_COLORS:
                if left_slot.color == slot_color:
                    same_color_anchor_count += 1
                break
            cursor -= 1

        cursor = order_index + 1
        while cursor < len(ordered_slots):
            right_slot = ordered_slots[cursor]
            if right_slot.is_revealed and right_slot.color in CARD_COLORS:
                if right_slot.color == slot_color:
                    same_color_anchor_count += 1
                break
            cursor += 1

        return same_color_anchor_count

    def _public_slot_interval(
        self,
        public_slots: Sequence[CardSlot],
        slot_index: int,
    ) -> Tuple[Optional[int], Optional[int], int]:
        ordered_slots = sorted(public_slots, key=lambda slot: slot.slot_index)
        index_by_slot = {slot.slot_index: idx for idx, slot in enumerate(ordered_slots)}
        order_index = index_by_slot.get(slot_index)
        if order_index is None:
            return None, None, MAX_CARD_VALUE

        lower: Optional[int] = None
        upper: Optional[int] = None

        cursor = order_index - 1
        while cursor >= 0:
            left_slot = ordered_slots[cursor]
            if left_slot.is_revealed:
                left_value = numeric_card_value(left_slot.known_card())
                if left_value is not None:
                    lower = left_value
                    break
            cursor -= 1

        cursor = order_index + 1
        while cursor < len(ordered_slots):
            right_slot = ordered_slots[cursor]
            if right_slot.is_revealed:
                right_value = numeric_card_value(right_slot.known_card())
                if right_value is not None:
                    upper = right_value
                    break
            cursor += 1

        effective_low = 0 if lower is None else lower
        effective_high = MAX_CARD_VALUE if upper is None else upper
        if effective_high < effective_low:
            effective_high = effective_low
        width = max(0, effective_high - effective_low)
        return lower, upper, width

    def _remaining_hidden_on_target_after_hit(
        self,
        game_state: GameState,
        signal: GuessSignal,
    ) -> int:
        hidden_count = 0
        for slot in game_state.resolved_ordered_slots(signal.target_player_id):
            if slot.slot_index == signal.target_slot_index:
                continue
            if slot.known_card() is None:
                hidden_count += 1
        return hidden_count

    def _slot_finish_chain_pressure_after_hit(
        self,
        game_state: GameState,
        player_id: str,
        slot_index: int,
    ) -> float:
        return clamp(
            self._player_finish_pressure(
                game_state,
                player_id,
                exclude_slot=slot_key(player_id, slot_index),
            ),
            0.0,
            1.0,
        )

    def _player_attackability(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        player_id: str,
        *,
        exclude_slot: Optional[SlotKey] = None,
    ) -> float:
        slot_pressures: List[float] = []
        for slot in game_state.resolved_ordered_slots(player_id):
            if slot.known_card() is not None:
                continue
            key = slot_key(player_id, slot.slot_index)
            if exclude_slot is not None and key == exclude_slot:
                continue
            slot_pressures.append(
                self._slot_attackability(
                    game_state,
                    hypothesis_by_player,
                    player_id,
                    slot.slot_index,
                )
            )
        if not slot_pressures:
            return 0.0
        slot_pressures.sort(reverse=True)
        best = slot_pressures[0]
        if len(slot_pressures) >= 2:
            best += self.PLAYER_SECONDARY_ATTACKABILITY_BLEND * slot_pressures[1]
            best = clamp(best, 0.0, 1.0)
        finish_pressure = self._player_finish_pressure(
            game_state,
            player_id,
            exclude_slot=exclude_slot,
        )
        attack_window_pressure = self._player_attack_window_pressure(
            game_state,
            player_id,
            exclude_slot=exclude_slot,
        )
        global_collapse_pressure = self._global_public_collapse_pressure(game_state)
        global_propagation_pressure = self._global_public_propagation_pressure(game_state)
        recent_public_reveal = self._recent_public_reveal_pressure(game_state, player_id)
        recent_failed_guess = self._recent_failed_guess_pressure(game_state, player_id)
        return best * (
            1.0
            + (self.PLAYER_FINISH_PRESSURE_BONUS * finish_pressure)
            + (self.PLAYER_ATTACK_WINDOW_BONUS * attack_window_pressure)
            + (self.PLAYER_GLOBAL_COLLAPSE_BONUS * global_collapse_pressure)
            + (self.PLAYER_GLOBAL_PROPAGATION_BONUS * global_propagation_pressure)
            + (self.PLAYER_RECENT_PUBLIC_REVEAL_BONUS * recent_public_reveal)
            + (self.PLAYER_RECENT_FAILED_GUESS_BONUS * recent_failed_guess)
        )

    def _player_attack_window_pressure(
        self,
        game_state: GameState,
        player_id: str,
        *,
        exclude_slot: Optional[SlotKey] = None,
    ) -> float:
        slot_pressures: List[float] = []
        for slot in game_state.resolved_ordered_slots(player_id):
            if slot.known_card() is not None:
                continue
            key = slot_key(player_id, slot.slot_index)
            if exclude_slot is not None and key == exclude_slot:
                continue
            slot_pressures.append(
                self._slot_attack_window_pressure(
                    game_state,
                    player_id,
                    slot.slot_index,
                )
            )
        if not slot_pressures:
            return 0.0
        best = max(slot_pressures)
        finish_pressure = self._player_finish_pressure(
            game_state,
            player_id,
            exclude_slot=exclude_slot,
        )
        return clamp(
            (0.62 * best) + (0.38 * finish_pressure),
            0.0,
            1.0,
        )

    def _player_finish_pressure(
        self,
        game_state: GameState,
        player_id: str,
        *,
        exclude_slot: Optional[SlotKey] = None,
    ) -> float:
        hidden_count = sum(
            1
            for slot in game_state.resolved_ordered_slots(player_id)
            if slot.known_card() is None
            and (exclude_slot is None or slot_key(player_id, slot.slot_index) != exclude_slot)
        )
        return clamp(
            (self.PLAYER_FINISH_PRESSURE_REFERENCE - float(hidden_count))
            / self.PLAYER_FINISH_PRESSURE_REFERENCE,
            0.0,
            1.0,
        )

    def _recent_public_reveal_pressure(
        self,
        game_state: GameState,
        player_id: str,
    ) -> float:
        reveal_score = 0.0
        recency_weight = 1.0
        for action in reversed(getattr(game_state, "actions", ())):
            revealed_player_id = getattr(action, "revealed_player_id", None)
            if revealed_player_id == player_id and action.revealed_card() is not None:
                reveal_score += recency_weight
            recency_weight *= 0.6
            if recency_weight < 0.2:
                break
        return clamp(reveal_score, 0.0, 1.0)

    def _global_public_collapse_pressure(
        self,
        game_state: GameState,
    ) -> float:
        collapse_score = 0.0
        recency_weight = 1.0
        for action in reversed(getattr(game_state, "actions", ())):
            if action.revealed_card() is not None:
                collapse_score += recency_weight
            elif getattr(action, "action_type", None) == "guess" and not getattr(action, "result", False):
                if action.guessed_card() is not None:
                    collapse_score += 0.55 * recency_weight
            recency_weight *= 0.7
            if recency_weight < 0.2:
                break
        return clamp(collapse_score, 0.0, 1.0)

    def _global_public_propagation_pressure(
        self,
        game_state: GameState,
    ) -> float:
        inference_players = list(game_state.inference_player_ids())
        if not inference_players:
            return 0.0

        exposure_pressures: List[float] = []
        finish_pressures: List[float] = []
        for player_id in inference_players:
            exposure_profile = self._public_hand_exposure_profile(game_state, player_id)
            exposure_pressures.append(
                clamp(
                    exposure_profile["total_exposure"]
                    + (0.30 * exposure_profile["finish_fragility"]),
                    0.0,
                    1.0,
                )
            )
            finish_pressures.append(
                self._player_finish_pressure(game_state, player_id)
            )

        exposure_pressure = sum(exposure_pressures) / len(exposure_pressures)
        finish_pressure = sum(finish_pressures) / len(finish_pressures)
        collapse_players: Set[str] = set()
        player_recency_pressure: Dict[str, float] = {
            player_id: 0.0 for player_id in inference_players
        }

        recency_weight = 1.0
        for action in reversed(getattr(game_state, "actions", ())):
            revealed_player_id = getattr(action, "revealed_player_id", None)
            if revealed_player_id in player_recency_pressure and action.revealed_card() is not None:
                player_recency_pressure[revealed_player_id] = max(
                    player_recency_pressure[revealed_player_id],
                    recency_weight,
                )
                collapse_players.add(revealed_player_id)
            elif (
                getattr(action, "action_type", None) == "guess"
                and not getattr(action, "result", False)
            ):
                failed_target_player_id = getattr(action, "target_player_id", None)
                if (
                    failed_target_player_id in player_recency_pressure
                    and action.guessed_card() is not None
                ):
                    player_recency_pressure[failed_target_player_id] = max(
                        player_recency_pressure[failed_target_player_id],
                        0.72 * recency_weight,
                    )
                    collapse_players.add(failed_target_player_id)
            recency_weight *= 0.68
            if recency_weight < 0.18:
                break

        collapse_breadth = len(collapse_players) / len(inference_players)
        recency_pressure = sum(player_recency_pressure.values()) / len(inference_players)
        return clamp(
            (0.35 * self._global_public_collapse_pressure(game_state))
            + (0.20 * collapse_breadth)
            + (0.25 * exposure_pressure)
            + (0.10 * finish_pressure)
            + (0.10 * recency_pressure),
            0.0,
            1.0,
        )

    def _recent_player_collapse_streak_pressure(
        self,
        game_state: GameState,
        player_id: str,
    ) -> float:
        streak_score = 0.0
        recency_weight = 1.0
        for action in reversed(getattr(game_state, "actions", ())):
            is_player_collapse = False
            if getattr(action, "revealed_player_id", None) == player_id and action.revealed_card() is not None:
                is_player_collapse = True
            elif (
                getattr(action, "action_type", None) == "guess"
                and getattr(action, "target_player_id", None) == player_id
                and not getattr(action, "result", False)
                and action.guessed_card() is not None
            ):
                is_player_collapse = True

            if is_player_collapse:
                streak_score += recency_weight
                recency_weight *= 0.72
                if recency_weight < 0.18:
                    break
                continue

            if streak_score > 0.0:
                break
        return clamp(streak_score, 0.0, 1.0)

    def _recent_public_bridge_signal_for_guess(
        self,
        game_state: GameState,
        signal: GuessSignal,
    ) -> float:
        guessed_value = numeric_card_value(signal.guessed_card)
        if guessed_value is None:
            return 0.0

        bridge_score = 0.0
        recency_weight = 1.0
        for action in reversed(getattr(game_state, "actions", ())):
            revealed_card = action.revealed_card()
            revealed_player_id = getattr(action, "revealed_player_id", None)
            if (
                revealed_card is None
                or revealed_player_id is None
                or revealed_player_id == signal.target_player_id
                or revealed_card[0] != signal.guessed_card[0]
            ):
                recency_weight *= 0.7
                if recency_weight < 0.2:
                    break
                continue
            revealed_value = numeric_card_value(revealed_card)
            if revealed_value is None:
                recency_weight *= 0.7
                if recency_weight < 0.2:
                    break
                continue
            distance = abs(guessed_value - revealed_value)
            if distance <= 2:
                bridge_score += recency_weight * ((3.0 - distance) / 3.0)
            recency_weight *= 0.7
            if recency_weight < 0.2:
                break
        return clamp(bridge_score, 0.0, 1.0)

    def _recent_failed_guess_pressure(
        self,
        game_state: GameState,
        player_id: str,
        slot_index: Optional[int] = None,
    ) -> float:
        failure_score = 0.0
        recency_weight = 1.0
        for action in reversed(getattr(game_state, "actions", ())):
            if getattr(action, "action_type", None) != "guess":
                continue
            if getattr(action, "result", False):
                recency_weight *= 0.6
                if recency_weight < 0.2:
                    break
                continue
            if getattr(action, "target_player_id", None) != player_id:
                recency_weight *= 0.6
                if recency_weight < 0.2:
                    break
                continue
            if slot_index is not None and getattr(action, "target_slot_index", None) != slot_index:
                recency_weight *= 0.6
                if recency_weight < 0.2:
                    break
                continue
            if action.guessed_card() is not None:
                failure_score += recency_weight
            recency_weight *= 0.6
            if recency_weight < 0.2:
                break
        return clamp(failure_score, 0.0, 1.0)

    def _slot_attackability(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        player_id: str,
        slot_index: int,
    ) -> float:
        try:
            slot = game_state.get_slot(player_id, slot_index)
        except ValueError:
            return 0.0

        if slot.known_card() is not None:
            return 0.0

        low, high, width = self._slot_numeric_interval(
            game_state,
            hypothesis_by_player,
            player_id,
            slot_index,
        )
        color_bonus = 1.0
        if getattr(slot, "color", None) is not None:
            color_bonus = 1.12

        confidence = color_bonus / max(1.0, float(width + 1))
        if low is not None and high is not None and width <= 2:
            confidence *= 1.08
        if (
            getattr(slot, "color", None) is not None
            and (
                (low is None and high is not None and high <= 4)
                or (high is None and low is not None and low >= (MAX_CARD_VALUE - 4))
            )
        ):
            confidence *= self.SLOT_EDGE_PRESSURE_BONUS
        reveal_neighbor_pressure = self._recent_reveal_neighbor_pressure(
            game_state,
            player_id,
            slot_index,
        )
        if reveal_neighbor_pressure > 0.0:
            confidence *= 1.0 + (
                (self.SLOT_RECENT_REVEAL_NEIGHBOR_BONUS - 1.0)
                * reveal_neighbor_pressure
            )
        failed_guess_neighbor_pressure = self._recent_failed_guess_neighbor_pressure(
            game_state,
            player_id,
            slot_index,
        )
        if failed_guess_neighbor_pressure > 0.0:
            confidence *= 1.0 + (
                (self.SLOT_FAILURE_NEIGHBOR_ATTACK_BONUS - 1.0)
                * failed_guess_neighbor_pressure
            )
        attack_window_pressure = self._slot_attack_window_pressure(
            game_state,
            player_id,
            slot_index,
        )
        if attack_window_pressure > 0.0:
            confidence *= 1.0 + (
                (self.SLOT_ATTACK_WINDOW_BONUS - 1.0)
                * attack_window_pressure
            )
        return confidence

    def _slot_attack_window_pressure(
        self,
        game_state: GameState,
        player_id: str,
        slot_index: int,
    ) -> float:
        ordered_slots = list(game_state.resolved_ordered_slots(player_id))
        index_by_slot = {slot.slot_index: idx for idx, slot in enumerate(ordered_slots)}
        order_index = index_by_slot.get(slot_index)
        if order_index is None:
            return 0.0
        if ordered_slots[order_index].known_card() is not None:
            return 0.0

        span = 1
        left_anchor = 0.0
        cursor = order_index - 1
        while cursor >= 0:
            left_slot = ordered_slots[cursor]
            if left_slot.known_card() is not None:
                left_anchor = 1.0
                break
            span += 1
            cursor -= 1
        if cursor < 0:
            left_anchor = max(left_anchor, 0.45)

        right_anchor = 0.0
        cursor = order_index + 1
        while cursor < len(ordered_slots):
            right_slot = ordered_slots[cursor]
            if right_slot.known_card() is not None:
                right_anchor = 1.0
                break
            span += 1
            cursor += 1
        if cursor >= len(ordered_slots):
            right_anchor = max(right_anchor, 0.45)

        span_signal = clamp((4.0 - float(span)) / 3.0, 0.0, 1.0)
        anchor_signal = clamp((left_anchor + right_anchor) / 2.0, 0.0, 1.0)
        dual_anchor_bonus = 0.18 if left_anchor > 0.0 and right_anchor > 0.0 else 0.0
        return clamp(
            (0.58 * span_signal) + (0.32 * anchor_signal) + dual_anchor_bonus,
            0.0,
            1.0,
        )

    def _recent_reveal_neighbor_pressure(
        self,
        game_state: GameState,
        player_id: str,
        slot_index: int,
    ) -> float:
        reveal_score = 0.0
        recency_weight = 1.0
        for action in reversed(getattr(game_state, "actions", ())):
            revealed_player_id = getattr(action, "revealed_player_id", None)
            revealed_slot_index = getattr(action, "revealed_slot_index", None)
            if (
                revealed_player_id == player_id
                and isinstance(revealed_slot_index, int)
                and abs(revealed_slot_index - slot_index) == 1
                and action.revealed_card() is not None
            ):
                reveal_score += recency_weight
            recency_weight *= 0.6
            if recency_weight < 0.2:
                break
        return clamp(reveal_score, 0.0, 1.0)

    def _recent_failed_guess_neighbor_pressure(
        self,
        game_state: GameState,
        player_id: str,
        slot_index: int,
    ) -> float:
        failure_score = 0.0
        recency_weight = 1.0
        for action in reversed(getattr(game_state, "actions", ())):
            if getattr(action, "action_type", None) != "guess":
                continue
            failed_slot_index = getattr(action, "target_slot_index", None)
            if (
                not getattr(action, "result", False)
                and getattr(action, "target_player_id", None) == player_id
                and isinstance(failed_slot_index, int)
                and abs(failed_slot_index - slot_index) == 1
                and action.guessed_card() is not None
            ):
                failure_score += recency_weight
            recency_weight *= 0.6
            if recency_weight < 0.2:
                break
        return clamp(failure_score, 0.0, 1.0)

    def _previous_guess_signal(
        self,
        game_state: GameState,
        signal: GuessSignal,
    ) -> Optional[GuessSignal]:
        previous: Optional[GuessSignal] = None
        guess_action_index = 0

        for action in getattr(game_state, "actions", ()):
            if getattr(action, "action_type", None) != "guess":
                continue
            if guess_action_index >= signal.action_index:
                break

            guessed_card = action.guessed_card()
            guesser_id = getattr(action, "guesser_id", None)
            target_player_id = getattr(action, "target_player_id", None)
            target_slot_index = getattr(action, "target_slot_index", None)
            if (
                guessed_card is not None
                and guesser_id == signal.guesser_id
                and target_player_id is not None
                and target_slot_index is not None
            ):
                previous = GuessSignal(
                    action_index=guess_action_index,
                    guesser_id=guesser_id,
                    target_player_id=target_player_id,
                    target_slot_index=target_slot_index,
                    guessed_card=guessed_card,
                    result=bool(getattr(action, "result", False)),
                    continued_turn=getattr(action, "continued_turn", None),
                )

            guess_action_index += 1

        return previous

    def _has_prior_failed_guess_on_slot(
        self,
        game_state: GameState,
        signal: GuessSignal,
    ) -> bool:
        return bool(self._prior_failed_numeric_guesses_on_slot(game_state, signal))

    def _prior_failed_numeric_guesses_on_slot(
        self,
        game_state: GameState,
        signal: GuessSignal,
    ) -> List[int]:
        failed_values: List[int] = []
        guess_action_index = 0
        for action in getattr(game_state, "actions", ()):
            if getattr(action, "action_type", None) != "guess":
                continue
            if guess_action_index >= signal.action_index:
                break

            if (
                getattr(action, "guesser_id", None) == signal.guesser_id
                and getattr(action, "target_player_id", None) == signal.target_player_id
                and getattr(action, "target_slot_index", None) == signal.target_slot_index
                and not bool(getattr(action, "result", False))
            ):
                guessed_card = action.guessed_card()
                guessed_numeric = numeric_card_value(guessed_card)
                if guessed_numeric is not None:
                    failed_values.append(guessed_numeric)

            guess_action_index += 1
        return failed_values

    def _latest_failed_numeric_guess_on_slot(
        self,
        game_state: GameState,
        *,
        guesser_id: str,
        target_player_id: str,
        target_slot_index: int,
    ) -> Optional[int]:
        latest_failed: Optional[int] = None
        for action in getattr(game_state, "actions", ()):
            if getattr(action, "action_type", None) != "guess":
                continue
            if (
                getattr(action, "guesser_id", None) == guesser_id
                and getattr(action, "target_player_id", None) == target_player_id
                and getattr(action, "target_slot_index", None) == target_slot_index
                and not bool(getattr(action, "result", False))
            ):
                guessed_numeric = numeric_card_value(action.guessed_card())
                if guessed_numeric is not None:
                    latest_failed = guessed_numeric
        return latest_failed

    def _slot_numeric_interval(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        player_id: str,
        slot_index: int,
    ) -> Tuple[Optional[int], Optional[int], int]:
        slots = game_state.resolved_ordered_slots(player_id)
        index_by_slot = {slot.slot_index: idx for idx, slot in enumerate(slots)}
        order_index = index_by_slot.get(slot_index)
        if order_index is None:
            return None, None, MAX_CARD_VALUE

        lower: Optional[int] = None
        upper: Optional[int] = None

        cursor = order_index - 1
        while cursor >= 0:
            left_card = self._resolve_slot_card(game_state, hypothesis_by_player, player_id, slots[cursor].slot_index)
            left_value = numeric_card_value(left_card)
            if left_value is not None:
                lower = left_value
                break
            cursor -= 1

        cursor = order_index + 1
        while cursor < len(slots):
            right_card = self._resolve_slot_card(game_state, hypothesis_by_player, player_id, slots[cursor].slot_index)
            right_value = numeric_card_value(right_card)
            if right_value is not None:
                upper = right_value
                break
            cursor += 1

        effective_low = 0 if lower is None else lower
        effective_high = MAX_CARD_VALUE if upper is None else upper
        if effective_high < effective_low:
            effective_high = effective_low
        width = max(0, effective_high - effective_low)
        return lower, upper, width

    def _resolve_slot_card(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        player_id: str,
        slot_index: int,
    ) -> Optional[Card]:
        try:
            slot = game_state.get_slot(player_id, slot_index)
        except ValueError:
            return None

        known_card = slot.known_card()
        if known_card is not None:
            return known_card
        return hypothesis_by_player.get(player_id, {}).get(slot_index)

class DaVinciInferenceEngine:
    """Infer hard posterior, soft posterior, and blended posterior for all hidden slots."""

    def __init__(self, game_state: GameState):
        self.game_state = game_state
        self.inference_player_ids = [
            player_id
            for player_id in game_state.inference_player_ids()
            if game_state.resolved_ordered_slots(player_id)
        ]
        self.player_slots = {
            player_id: game_state.resolved_ordered_slots(player_id)
            for player_id in self.inference_player_ids
        }
        self.hard_constraints = HardConstraintCompiler(game_state).compile()
        self.preassigned_hidden: Dict[str, Dict[int, Card]] = {
            player_id: {} for player_id in self.inference_player_ids
        }
        self.search_positions: List[HiddenPosition] = []
        self.position_by_key: Dict[SlotKey, HiddenPosition] = {}
        self.player_slot_order: Dict[str, Dict[int, int]] = {}
        self.publicly_collapsed_slots: Set[SlotKey] = set()

        self._build_slot_views()
        self.available_cards = tuple(self._get_available_cards())
        self.base_domains = self._build_base_domains()
        self._validate_initial_domains()

    def _build_all_possible_cards(self) -> Tuple[Card, ...]:
        ordered_cards = [
            (color, value)
            for value in range(MAX_CARD_VALUE + 1)
            for color in CARD_COLORS
        ]
        ordered_cards.extend((color, JOKER) for color in CARD_COLORS)
        return tuple(ordered_cards)

    def _build_slot_views(self) -> None:
        for player_id, slots in self.player_slots.items():
            order_map: Dict[int, int] = {}
            for order_index, slot in enumerate(slots):
                order_map[slot.slot_index] = order_index
                key = slot_key(player_id, slot.slot_index)
                forced = self.hard_constraints.fixed_by_slot.get(key)
                known = slot.known_card()

                if known is not None:
                    if forced is not None and forced != known:
                        raise ValueError(
                            f"Slot {key} conflicts between state {known} and action evidence {forced}"
                        )
                    continue

                if forced is not None:
                    self.preassigned_hidden[player_id][slot.slot_index] = forced
                    self.publicly_collapsed_slots.add(key)
                    continue

                if slot.value is None:
                    hidden = HiddenPosition(
                        player_id=player_id,
                        slot_index=slot.slot_index,
                        order_index=order_index,
                        color=getattr(slot, "color", None),
                    )
                    self.search_positions.append(hidden)
                    self.position_by_key[key] = hidden
            self.player_slot_order[player_id] = order_map

    def _get_available_cards(self) -> Tuple[Card, ...]:
        known_cards = set(self.game_state.known_cards())
        fixed_hidden_cards = {
            card
            for cards_by_slot in self.preassigned_hidden.values()
            for card in cards_by_slot.values()
        }
        used_cards = known_cards | fixed_hidden_cards
        return tuple(
            card
            for card in self._build_all_possible_cards()
            if card not in used_cards
        )

    def _build_base_domains(self) -> Dict[SlotKey, Tuple[Card, ...]]:
        base_domains: Dict[SlotKey, Tuple[Card, ...]] = {}
        for position in self.search_positions:
            key = slot_key(position.player_id, position.slot_index)
            forbidden = self.hard_constraints.forbidden_by_slot.get(key, set())
            domain = []
            for card in self.available_cards:
                if position.color is not None and card[0] != position.color:
                    continue
                if card in forbidden:
                    continue
                domain.append(card)
            base_domains[key] = tuple(domain)
        return base_domains

    def _validate_initial_domains(self) -> None:
        if len(set(self.available_cards)) != len(self.available_cards):
            raise ValueError("Available card pool contains duplicates.")
        for key, domain in self.base_domains.items():
            if not domain:
                raise ValueError(f"Hidden slot {key} has empty hard domain.")

    def infer_hidden_probabilities(
        self,
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        behavior_model: BehavioralLikelihoodModel,
    ) -> Tuple[
        FullProbabilityMatrix,
        FullProbabilityMatrix,
        FullProbabilityMatrix,
        int,
        float,
    ]:
        if not self.search_positions and not any(self.preassigned_hidden.values()):
            return {}, {}, {}, 0, 0.0

        hard_position_weights: Dict[str, DefaultDict[int, DefaultDict[Card, float]]] = {
            player_id: defaultdict(lambda: defaultdict(float))
            for player_id in self.inference_player_ids
        }
        soft_position_weights: Dict[str, DefaultDict[int, DefaultDict[Card, float]]] = {
            player_id: defaultdict(lambda: defaultdict(float))
            for player_id in self.inference_player_ids
        }

        legal_world_count = 0
        total_soft_weight = 0.0
        current_assignment: Dict[str, Dict[int, Card]] = {
            player_id: dict(cards_by_slot)
            for player_id, cards_by_slot in self.preassigned_hidden.items()
        }
        dead_state_cache: Set[Tuple[Tuple[Card, ...], Tuple[Tuple[str, Tuple[Optional[Card], ...]], ...]]] = set()

        def dfs(remaining_positions: Tuple[HiddenPosition, ...], remaining_cards: Tuple[Card, ...]) -> bool:
            nonlocal legal_world_count, total_soft_weight

            state_signature = self._state_signature(current_assignment, remaining_cards)
            if state_signature in dead_state_cache:
                return False

            if not remaining_positions:
                legal_world_count += 1
                hypothesis_by_player = {
                    player_id: dict(cards_by_slot)
                    for player_id, cards_by_slot in current_assignment.items()
                    if cards_by_slot
                }
                soft_weight = behavior_model.score_hypothesis(
                    hypothesis_by_player,
                    guess_signals_by_player,
                    self.game_state,
                )
                total_soft_weight += soft_weight
                self._accumulate_world(
                    hard_position_weights,
                    soft_position_weights,
                    current_assignment,
                    soft_weight,
                )
                return True

            feasible_domains = self._feasible_domains(remaining_positions, remaining_cards, current_assignment)
            if any(len(domain) == 0 for _, domain in feasible_domains):
                dead_state_cache.add(state_signature)
                return False

            next_position, next_domain = min(feasible_domains, key=lambda item: (len(item[1]), item[0].player_id, item[0].slot_index))
            next_remaining_positions = tuple(
                position for position in remaining_positions if position != next_position
            )
            found_any_solution = False

            for candidate in next_domain:
                current_assignment[next_position.player_id][next_position.slot_index] = candidate
                next_remaining_cards = tuple(card for card in remaining_cards if card != candidate)
                found_any_solution = dfs(next_remaining_positions, next_remaining_cards) or found_any_solution
                del current_assignment[next_position.player_id][next_position.slot_index]

            if not found_any_solution:
                dead_state_cache.add(state_signature)
            return found_any_solution

        dfs(tuple(self.search_positions), self.available_cards)

        if legal_world_count == 0:
            return {}, {}, {}, 0, 0.0

        hard_matrix = self._normalize_matrix(hard_position_weights, legal_world_count)
        soft_matrix = (
            self._normalize_matrix(soft_position_weights, total_soft_weight)
            if total_soft_weight > 0
            else hard_matrix
        )
        blended_matrix = self._blend_probability_matrices(hard_matrix, soft_matrix)
        return hard_matrix, soft_matrix, blended_matrix, legal_world_count, total_soft_weight

    def _state_signature(
        self,
        current_assignment: Dict[str, Dict[int, Card]],
        remaining_cards: Tuple[Card, ...],
    ) -> Tuple[Tuple[Card, ...], Tuple[Tuple[str, Tuple[Optional[Card], ...]], ...]]:
        player_signatures: List[Tuple[str, Tuple[Optional[Card], ...]]] = []
        for player_id in sorted(self.player_slots):
            signature: List[Optional[Card]] = []
            for slot in self.player_slots[player_id]:
                known = slot.known_card()
                if known is not None:
                    signature.append(known)
                    continue
                signature.append(current_assignment[player_id].get(slot.slot_index))
            player_signatures.append((player_id, tuple(signature)))
        return tuple(sorted(remaining_cards, key=card_sort_key)), tuple(player_signatures)

    def _feasible_domains(
        self,
        remaining_positions: Tuple[HiddenPosition, ...],
        remaining_cards: Tuple[Card, ...],
        current_assignment: Dict[str, Dict[int, Card]],
    ) -> List[Tuple[HiddenPosition, Tuple[Card, ...]]]:
        remaining_card_set = set(remaining_cards)
        feasible: List[Tuple[HiddenPosition, Tuple[Card, ...]]] = []
        for position in remaining_positions:
            key = slot_key(position.player_id, position.slot_index)
            candidates = []
            for candidate in self.base_domains[key]:
                if candidate not in remaining_card_set:
                    continue
                if not self._fits_local_order(position, candidate, current_assignment):
                    continue
                candidates.append(candidate)
            candidates.sort(key=card_sort_key)
            feasible.append((position, tuple(candidates)))
        return feasible

    def _fits_local_order(
        self,
        position: HiddenPosition,
        candidate: Card,
        current_assignment: Dict[str, Dict[int, Card]],
    ) -> bool:
        if candidate[1] == JOKER:
            return True

        left_bound = self._nearest_numeric_bound(position.player_id, position.order_index, -1, current_assignment)
        if left_bound is not None and self._card_precedes(candidate, left_bound):
            return False

        right_bound = self._nearest_numeric_bound(position.player_id, position.order_index, 1, current_assignment)
        if right_bound is not None and self._card_precedes(right_bound, candidate):
            return False

        return True

    def _nearest_numeric_bound(
        self,
        player_id: str,
        order_index: int,
        direction: int,
        current_assignment: Dict[str, Dict[int, Card]],
    ) -> Optional[Card]:
        slots = self.player_slots[player_id]
        cursor = order_index + direction
        while 0 <= cursor < len(slots):
            slot = slots[cursor]
            card = slot.known_card()
            if card is None:
                card = current_assignment[player_id].get(slot.slot_index)
            if card is not None and card[1] != JOKER:
                return card
            cursor += direction
        return None

    def _card_precedes(self, left: Card, right: Card) -> bool:
        if left[1] == JOKER or right[1] == JOKER:
            return False
        if left[1] != right[1]:
            return int(left[1]) < int(right[1])
        return CARD_COLOR_RANK[left[0]] < CARD_COLOR_RANK[right[0]]

    def _accumulate_world(
        self,
        hard_position_weights: Dict[str, DefaultDict[int, DefaultDict[Card, float]]],
        soft_position_weights: Dict[str, DefaultDict[int, DefaultDict[Card, float]]],
        current_assignment: Dict[str, Dict[int, Card]],
        soft_weight: float,
    ) -> None:
        for player_id, cards_by_slot in current_assignment.items():
            for slot_index, card in cards_by_slot.items():
                hard_position_weights[player_id][slot_index][card] += 1.0
                soft_position_weights[player_id][slot_index][card] += soft_weight

    def _normalize_matrix(
        self,
        position_weights: Dict[str, DefaultDict[int, DefaultDict[Card, float]]],
        total_weight: float,
    ) -> FullProbabilityMatrix:
        if total_weight <= 0:
            return {}
        result: FullProbabilityMatrix = {}
        for player_id, weighted_slots in position_weights.items():
            if not weighted_slots:
                continue
            result[player_id] = {
                slot_index: {
                    card: weight / total_weight
                    for card, weight in weighted_cards.items()
                }
                for slot_index, weighted_cards in weighted_slots.items()
            }
        return result

    def _blend_probability_matrices(
        self,
        hard_matrix: FullProbabilityMatrix,
        soft_matrix: FullProbabilityMatrix,
    ) -> FullProbabilityMatrix:
        blended: FullProbabilityMatrix = {}
        for player_id in sorted(set(hard_matrix) | set(soft_matrix)):
            player_positions: ProbabilityMatrix = {}
            all_slot_indices = set(hard_matrix.get(player_id, {})) | set(soft_matrix.get(player_id, {}))
            for slot_index in sorted(all_slot_indices):
                hard_dist = hard_matrix.get(player_id, {}).get(slot_index, {})
                soft_dist = soft_matrix.get(player_id, {}).get(slot_index, hard_dist)
                cards = set(hard_dist) | set(soft_dist)
                blended_weights = {
                    card: ((1.0 - SOFT_BEHAVIOR_BLEND) * hard_dist.get(card, 0.0))
                    + (SOFT_BEHAVIOR_BLEND * soft_dist.get(card, 0.0))
                    for card in cards
                }
                player_positions[slot_index] = normalize_card_distribution(blended_weights)
            if player_positions:
                blended[player_id] = player_positions
        return blended


class DaVinciDecisionEngine:
    """Score moves by immediate EV, continuation likelihood, and stop-aware thresholding."""

    HIT_REWARD = 10.0
    MISS_BASE = 12.0
    MISS_ENDGAME_MULTIPLIER = 12.0
    INFORMATION_GAIN_WEIGHT = 0.45
    CONTINUATION_DISCOUNT = 0.72
    MAX_CANDIDATES_PER_SLOT = 8
    MIN_CANDIDATE_PROBABILITY = 1e-9
    EPSILON = 1e-9

    STOP_MARGIN_BASE = 0.18
    STOP_MARGIN_RISK_SCALE = 0.035
    STOP_MARGIN_SHORT_HAND = 0.22
    STOP_MARGIN_LOW_CONFIDENCE = 0.28
    STOP_MARGIN_WEAK_CONTINUATION = 0.16
    STOP_MARGIN_WEAK_EDGE = 0.22
    STOP_MARGIN_WEAK_ROLLOUT = 0.24
    STOP_MARGIN_FRAGILE_POST_HIT = 0.18
    STOP_MARGIN_TOP_K_SUPPORT = 0.18
    STOP_MARGIN_LOW_ATTACKABILITY = 0.18
    STOP_MARGIN_BEHAVIOR_ROLLOUT = 0.16
    STOP_MARGIN_SELF_EXPOSURE = 0.18
    STOP_MARGIN_NEW_DRAWN_EXPOSURE = 0.22
    STOP_MARGIN_FINISH_FRAGILITY = 0.18
    STOP_MARGIN_EDGE_SELF_EXPOSURE_BOOST = 0.45
    STOP_MARGIN_FAILURE_RECOVERY = 0.12
    STOP_MARGIN_ATTACK_WINDOW = 0.10
    STOP_MARGIN_JOINT_COLLAPSE = 0.08
    STOP_MARGIN_GLOBAL_PROPAGATION = 0.08
    STOP_MARGIN_PUBLIC_BRIDGE = 0.06
    STOP_MARGIN_TARGET_CHAIN = 0.07
    STOP_MARGIN_FINISH_CHAIN = 0.08
    STOP_MARGIN_BRANCH_SEARCH = 0.10
    STOP_MARGIN_EXPECTIMAX = 0.12
    STOP_MARGIN_MCTS = 0.10
    STOP_EDGE_REFERENCE = 0.18
    ROLLOUT_MARGIN_REFERENCE = 0.40
    FAILURE_RECOVERY_REFERENCE = 0.30
    POST_HIT_GAP_REFERENCE = 0.22
    POST_HIT_TOP_K_COUNT = 3
    DRAW_ROLLOUT_CONTINUE_MARGIN_REFERENCE = 1.0
    CONTINUATION_TOP_K_BLEND = 0.38
    CONTINUATION_BRANCH_SEARCH_BLEND = 0.24
    CONTINUATION_EXPECTIMAX_BLEND = 0.18
    LOW_CONFIDENCE_GUARD_MARGIN = 0.22
    WEAK_EDGE_GUARD_MARGIN = 0.18
    SELF_EXPOSURE_GUARD_REFERENCE = 0.40
    SELF_EXPOSURE_GUARD_MARGIN = 0.20
    LOW_CONFIDENCE_SELF_EXPOSURE_BOOST = 0.10
    WEAK_EDGE_SELF_EXPOSURE_BOOST = 0.08
    ATTACKABILITY_REFERENCE = BehavioralLikelihoodModel.ATTACKABILITY_TIGHT_THRESHOLD
    DEFAULT_BEHAVIOR_GUIDANCE_MULTIPLIER = 1.0
    DEFAULT_BEHAVIOR_MATCH_MULTIPLIER = 1.0
    BEHAVIOR_MATCH_CONTEXT_TOP_K = 3
    BEHAVIOR_MATCH_DECISION_SCALE = 0.45
    BEHAVIOR_MATCH_SUPPORT_REFERENCE = 0.10
    BEHAVIOR_MATCH_CONTEXT_COUNT_REFERENCE = 4.0
    BEHAVIOR_MATCH_COMPONENT_WEIGHT_REFERENCE = 0.15
    BEHAVIOR_MATCH_COMPONENT_PENALTY_WEIGHT = 0.40
    BEHAVIOR_MATCH_NET_STRUCTURE_SCALE = 0.10
    BEHAVIOR_MATCH_DECISION_WINDOW = 0.12
    BEHAVIOR_MATCH_DECISION_NET_STRUCTURE_SCALE = 0.50
    POST_HIT_BEHAVIOR_SUPPORT_SCALE = 0.08
    POST_HIT_BEHAVIOR_SUPPORT_REFERENCE = 0.20
    POST_HIT_FAILURE_RECOVERY_SCALE = 0.65
    CONTINUATION_SELF_EXPOSURE_DRAG = 0.26
    CONTINUATION_FINISH_FRAGILITY_DRAG = 0.22
    SELF_EXPOSURE_SECONDARY_BLEND = 0.35
    SELF_EXPOSURE_RISK_SCALE = 0.28
    SELF_NEW_DRAWN_RISK_SCALE = 0.34
    SELF_EXPOSURE_COLOR_BONUS = 1.12
    SELF_EXPOSURE_NARROW_BONUS = 1.06
    SELF_EXPOSURE_EDGE_BONUS = 1.08
    SELF_EXPOSURE_NEW_DRAWN_BONUS = 1.18
    SELF_EXPOSURE_SAME_COLOR_ANCHOR_BONUS = 1.08
    SELF_EXPOSURE_DOUBLE_COLOR_ANCHOR_BONUS = 1.12
    FAILED_GUESS_SLOT_COLLAPSE_VALUE_BONUS = 0.28
    FAILED_GUESS_NEIGHBOR_COLLAPSE_VALUE_BONUS = 0.12
    FAILED_GUESS_PLAYER_COLLAPSE_VALUE_BONUS = 0.10
    FAILED_GUESS_SWITCH_CONTINUITY_VALUE_BONUS = 0.18
    TARGET_ATTACK_WINDOW_VALUE_BONUS = 0.24
    TARGET_ATTACK_WINDOW_CONTINUATION_SCALE = 0.12
    JOINT_COLLAPSE_VALUE_BONUS = 0.22
    JOINT_COLLAPSE_CONTINUATION_SCALE = 0.08
    GLOBAL_PROPAGATION_VALUE_BONUS = 0.20
    GLOBAL_PROPAGATION_CONTINUATION_SCALE = 0.08
    PUBLIC_REVEAL_BRIDGE_VALUE_BONUS = 0.18
    PUBLIC_REVEAL_BRIDGE_CONTINUATION_SCALE = 0.06
    TARGET_CHAIN_VALUE_BONUS = 0.22
    TARGET_CHAIN_CONTINUATION_SCALE = 0.08
    TARGET_FINISH_CHAIN_VALUE_BONUS = 0.26
    TARGET_FINISH_CHAIN_CONTINUATION_SCALE = 0.10
    BRANCH_SEARCH_FUTURE_MARGIN_BLEND = 0.55
    TREE_SEARCH_TOP_K = 2
    TREE_SEARCH_FUTURE_SCALE = 0.72
    MCTS_TOP_K = 2
    MCTS_DEEP_CHILD_TOP_K = 2
    MCTS_SIMULATION_COUNT = 6
    MCTS_SIMULATION_COUNT_PER_HORIZON = 3
    MCTS_SIMULATION_COUNT_PER_ROOT_MOVE = 2
    MCTS_SIMULATION_COUNT_CAP = 48
    MCTS_EXPLORATION_SCALE = 0.32
    MCTS_FUTURE_SCALE = 0.74
    GLOBAL_MCTS_TOP_K = 3
    GLOBAL_MCTS_OBJECTIVE_SCALE = 0.14
    GLOBAL_MCTS_RANKING_SCALE = 0.08
    EXACT_TREE_SEARCH_HORIZON_THRESHOLD = 4
    EXACT_TREE_SEARCH_NODE_BUDGET = 96
    EXPECTIMAX_TOP_K = 3
    EXPECTIMAX_FUTURE_SCALE = 0.60
    DEEP_ROLLOUT_DEPTH = 3
    DEEP_ROLLOUT_TARGET_HIDDEN_THRESHOLD = 1
    DEEP_ROLLOUT_SELF_HIDDEN_THRESHOLD = 3
    DEEP_ROLLOUT_SEARCH_SPACE_THRESHOLD = 80.0
    SELF_PLAY_BENCHMARK_HAND_SIZE = 3
    SELF_PLAY_BENCHMARK_WORLD_COUNT = 24
    SELF_PLAY_BENCHMARK_MAX_STEPS = 3
    LONG_SELF_PLAY_HAND_SIZE = 2
    LONG_SELF_PLAY_GAME_COUNT = 4
    LONG_SELF_PLAY_MAX_TURNS = 10
    LONG_SELF_PLAY_PRE_DRAW_ROLLOUT_DEPTH = 1
    LONG_SELF_PLAY_DRAW_ROLLOUT_SAMPLE_LIMIT = 2
    LONG_SELF_PLAY_PAIRED_WORLD_REPEATS = 2
    LONG_SELF_PLAY_LEAGUE_MATCH_COUNT = 6
    LONG_SELF_PLAY_LEAGUE_GAMES_PER_MATCH = 4
    LONG_SELF_PLAY_STABILITY_MIN_TOTAL_GAMES = 24
    LONG_SELF_PLAY_EVALUATION_MIN_TOTAL_GAMES = 96
    LONG_SELF_PLAY_STABILITY_SEED_STRIDE = 1009
    STRATEGY_OBJECTIVE_WIN_SCALE = 0.18
    STRATEGY_OBJECTIVE_CONTINUATION_SCALE = 0.20
    STRATEGY_OBJECTIVE_ATTACKABILITY_SCALE = 0.14
    STRATEGY_OBJECTIVE_INFORMATION_GAIN_SCALE = 0.12
    STRATEGY_OBJECTIVE_EXPECTIMAX_SCALE = 0.18
    STRATEGY_OBJECTIVE_SUPPORT_SCALE = 0.08
    STRATEGY_OBJECTIVE_BEHAVIOR_SCALE = 0.10
    STRATEGY_OBJECTIVE_SEARCH_SIGNAL_SCALE = 0.14
    STRATEGY_OBJECTIVE_MCTS_SCALE = 0.08
    STRATEGY_OBJECTIVE_RECOVERY_SCALE = 0.08
    STRATEGY_OBJECTIVE_OPENING_PRECISION_SCALE = 0.14
    STRATEGY_OBJECTIVE_INITIATIVE_RECOVERY_SCALE = 0.10
    STRATEGY_OBJECTIVE_CONFIDENCE_DRAG = 0.12
    STRATEGY_OBJECTIVE_NEW_DRAWN_DRAG = 0.10
    STRATEGY_OBJECTIVE_SELF_EXPOSURE_DRAG = 0.12
    STRATEGY_OBJECTIVE_FINISH_FRAGILITY_DRAG = 0.10
    STOP_MARGIN_OBJECTIVE_CREDIT = 0.16
    STOP_MARGIN_SEARCH_CREDIT = 0.08
    STOP_MARGIN_SEARCH_DEPTH_CREDIT = 0.05
    STOP_MARGIN_GUARD_RELIEF_SCALE = 0.30
    STOP_MARGIN_GUARD_POLICY_SCALE = 0.22
    GUARD_POLICY_BIAS = -1.70
    GUARD_POLICY_LOW_CONFIDENCE_WEIGHT = 2.10
    GUARD_POLICY_WEAK_EDGE_WEIGHT = 1.55
    GUARD_POLICY_SELF_EXPOSURE_WEIGHT = 1.85
    GUARD_POLICY_MARGIN_RELIEF_WEIGHT = 1.20
    GUARD_POLICY_SEARCH_RELIEF_WEIGHT = 0.95
    OPENING_PRECISION_MARGIN_REFERENCE = 0.28
    OPENING_JOKER_SUPPRESSION_MARGIN = 0.03
    OPENING_JOKER_PENALTY_SCALE = 0.78
    OPENING_JOKER_POSTERIOR_DAMPING = 0.42
    INITIATIVE_RECOVERY_REFERENCE = 3.0

    def calculate_risk_factor(self, my_hidden_count: int) -> float:
        exposure = 1.0 / max(1, my_hidden_count)
        return self.MISS_BASE + self.MISS_ENDGAME_MULTIPLIER * exposure

    def _recent_failed_guess_switch_continuity_signal(
        self,
        game_state: Optional[GameState],
        acting_player_id: Optional[str],
        target_player_id: str,
        guessed_card: Card,
    ) -> float:
        if game_state is None or acting_player_id is None:
            return 0.0

        previous_failed_action: Optional[GuessAction] = None
        for action in reversed(getattr(game_state, "actions", ())):
            if getattr(action, "action_type", None) != "guess":
                continue
            if getattr(action, "guesser_id", None) != acting_player_id:
                continue
            if getattr(action, "result", False):
                continue
            if action.guessed_card() is None:
                continue
            previous_failed_action = action
            break

        if previous_failed_action is None:
            return 0.0
        if getattr(previous_failed_action, "target_player_id", None) == target_player_id:
            return 0.0

        previous_card = previous_failed_action.guessed_card()
        if previous_card is None or previous_card[0] != guessed_card[0]:
            return 0.0

        previous_value = numeric_card_value(previous_card)
        current_value = numeric_card_value(guessed_card)
        if previous_value is None or current_value is None:
            return 0.0

        delta = abs(previous_value - current_value)
        if delta > 2:
            return 0.0
        return clamp(1.0 - (0.35 * delta), 0.0, 1.0)

    def _recent_public_reveal_bridge_signal(
        self,
        game_state: Optional[GameState],
        target_player_id: str,
        guessed_card: Card,
    ) -> float:
        if game_state is None:
            return 0.0
        guessed_value = numeric_card_value(guessed_card)
        if guessed_value is None:
            return 0.0

        bridge_score = 0.0
        recency_weight = 1.0
        for action in reversed(getattr(game_state, "actions", ())):
            revealed_card = action.revealed_card()
            revealed_player_id = getattr(action, "revealed_player_id", None)
            if (
                revealed_card is None
                or revealed_player_id is None
                or revealed_player_id == target_player_id
                or revealed_card[0] != guessed_card[0]
            ):
                recency_weight *= 0.7
                if recency_weight < 0.2:
                    break
                continue
            revealed_value = numeric_card_value(revealed_card)
            if revealed_value is None:
                recency_weight *= 0.7
                if recency_weight < 0.2:
                    break
                continue
            distance = abs(guessed_value - revealed_value)
            if distance <= 2:
                bridge_score += recency_weight * ((3.0 - distance) / 3.0)
            recency_weight *= 0.7
            if recency_weight < 0.2:
                break
        return clamp(bridge_score, 0.0, 1.0)

    def _recent_target_chain_signal(
        self,
        game_state: Optional[GameState],
        target_player_id: str,
    ) -> float:
        if game_state is None:
            return 0.0
        chain_score = 0.0
        recency_weight = 1.0
        for action in reversed(getattr(game_state, "actions", ())):
            if getattr(action, "action_type", None) != "guess":
                continue
            if getattr(action, "target_player_id", None) != target_player_id:
                continue
            if getattr(action, "result", False):
                chain_score += 0.65 * recency_weight
                if getattr(action, "continued_turn", None) is True:
                    chain_score += 0.20 * recency_weight
            if getattr(action, "revealed_player_id", None) == target_player_id:
                chain_score += 0.35 * recency_weight
            recency_weight *= 0.62
            if recency_weight < 0.15:
                break
        return clamp(chain_score, 0.0, 1.0)

    def evaluate_all_moves(
        self,
        *,
        full_probability_matrix: FullProbabilityMatrix,
        my_hidden_count: int,
        hidden_index_by_player: Dict[str, Dict[int, int]],
        behavior_model: BehavioralLikelihoodModel,
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        acting_player_id: Optional[str],
        behavior_guidance_profile: Optional[Dict[str, float]] = None,
        game_state: Optional[GameState] = None,
        behavior_map_hypothesis: Optional[Dict[str, Dict[int, Card]]] = None,
        blocked_slots: Optional[Set[SlotKey]] = None,
        rollout_depth: int = 1,
    ) -> Tuple[List[Dict[str, Any]], float]:
        risk_factor = self.calculate_risk_factor(my_hidden_count)
        self_exposure_profile = self._public_self_exposure_profile(game_state)
        blocked_slots = blocked_slots or set()
        moves: List[Dict[str, Any]] = []

        for player_id in sorted(full_probability_matrix):
            probability_matrix = full_probability_matrix[player_id]
            hidden_index_by_slot = hidden_index_by_player.get(player_id, {})
            for target_slot_index, slot_distribution in probability_matrix.items():
                if slot_key(player_id, target_slot_index) in blocked_slots:
                    continue
                ranked_candidates = sorted(
                    slot_distribution.items(),
                    key=lambda item: (-item[1], card_sort_key(item[0])),
                )
                for card, probability in ranked_candidates[: self.MAX_CANDIDATES_PER_SLOT]:
                    if probability <= self.MIN_CANDIDATE_PROBABILITY:
                        continue
                    move = self._score_single_move(
                        full_probability_matrix=full_probability_matrix,
                        my_hidden_count=my_hidden_count,
                        risk_factor=risk_factor,
                        hidden_index_by_slot=hidden_index_by_slot,
                        player_id=player_id,
                        slot_index=target_slot_index,
                        card=card,
                        probability=probability,
                        slot_distribution=slot_distribution,
                        behavior_model=behavior_model,
                        guess_signals_by_player=guess_signals_by_player,
                        acting_player_id=acting_player_id,
                        behavior_guidance_profile=behavior_guidance_profile,
                        game_state=game_state,
                        behavior_map_hypothesis=behavior_map_hypothesis,
                        self_exposure_profile=self_exposure_profile,
                        rollout_depth=rollout_depth,
                    )
                    moves.append(move)

        opening_phase = self._strategy_phase_for_state(game_state) == "post_draw_opening"
        if opening_phase:
            moves.sort(
                key=lambda move: (
                    -move.get("opening_precision_support", 0.0),
                    -move.get("opening_behavior_posterior_support", 0.0),
                    move.get("opening_joker_penalty", 0.0),
                    -move["win_probability"],
                    -move.get("opening_margin_signal", 0.0),
                    -move.get("strategy_objective", move.get("ranking_score", move["expected_value"])),
                    -move.get("ranking_score", move["expected_value"]),
                    -move["continuation_value"],
                    move["target_player_id"],
                    move["target_slot_index"],
                    card_sort_key((move["guess_card"][0], move["guess_card"][1])),
                )
            )
        else:
            moves.sort(
                key=lambda move: (
                    -move.get("strategy_objective", move.get("ranking_score", move["expected_value"])),
                    -move.get("ranking_score", move["expected_value"]),
                    -move["win_probability"],
                    -move["continuation_value"],
                    -move["continuation_likelihood"],
                    move["target_player_id"],
                    move["target_slot_index"],
                    card_sort_key((move["guess_card"][0], move["guess_card"][1])),
                )
            )
        return moves, risk_factor

    def _sort_scored_moves(
        self,
        moves: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        sorted_moves = list(moves)
        opening_phase = (
            bool(sorted_moves)
            and str(sorted_moves[0].get("strategy_phase", "")) == "post_draw_opening"
        )
        if opening_phase:
            sorted_moves.sort(
                key=lambda move: (
                    -move.get("opening_precision_support", 0.0),
                    -move.get("opening_behavior_posterior_support", 0.0),
                    move.get("opening_joker_penalty", 0.0),
                    -move["win_probability"],
                    -move.get("opening_margin_signal", 0.0),
                    -move.get(
                        "strategy_objective",
                        move.get("ranking_score", move["expected_value"]),
                    ),
                    -move.get("ranking_score", move["expected_value"]),
                    -move["continuation_value"],
                    move.get("target_player_id", ""),
                    int(move.get("target_slot_index", -1)),
                    card_sort_key(
                        tuple(move.get("guess_card", ("B", 0)))  # type: ignore[arg-type]
                    ),
                )
            )
        else:
            sorted_moves.sort(
                key=lambda move: (
                    -move.get(
                        "strategy_objective",
                        move.get("ranking_score", move["expected_value"]),
                    ),
                    -move.get("ranking_score", move["expected_value"]),
                    -move["win_probability"],
                    -move["continuation_value"],
                    -move["continuation_likelihood"],
                    move.get("target_player_id", ""),
                    int(move.get("target_slot_index", -1)),
                    card_sort_key(
                        tuple(move.get("guess_card", ("B", 0)))  # type: ignore[arg-type]
                    ),
                )
            )
        return sorted_moves

    def _apply_global_mcts_rerank(
        self,
        *,
        all_moves: Sequence[Dict[str, Any]],
        risk_factor: float,
        my_hidden_count: int,
    ) -> List[Dict[str, Any]]:
        ranked_moves = list(all_moves)
        if len(ranked_moves) <= 1:
            return ranked_moves
        root_moves = ranked_moves
        if any(
            "target_player_id" not in move
            or "target_slot_index" not in move
            or "guess_card" not in move
            for move in root_moves
        ):
            return ranked_moves
        baseline_stop_threshold = self._stop_threshold_breakdown(
            risk_factor=risk_factor,
            my_hidden_count=my_hidden_count,
            best_move=root_moves[0],
        )["threshold"]
        global_mcts = self._run_hybrid_full_game_search(
            tree_moves=root_moves,
            stop_score=baseline_stop_threshold,
        )
        root_stats_by_key = {
            tuple(root_stat.get("move_key", ())): root_stat
            for root_stat in global_mcts.get("root_children", ())
        }
        for move in ranked_moves:
            root_stat = root_stats_by_key.get(self._move_identity_key(move), {})
            global_mcts_margin = float(root_stat.get("margin", 0.0))
            global_mcts_signal = float(root_stat.get("signal", 0.0))
            global_mcts_bonus = (
                self.GLOBAL_MCTS_OBJECTIVE_SCALE * global_mcts_margin
                + self.GLOBAL_MCTS_RANKING_SCALE * global_mcts_signal
            )
            move["global_mcts_value"] = float(
                root_stat.get(
                    "value",
                    move.get(
                        "strategy_objective",
                        move.get("ranking_score", move.get("expected_value", 0.0)),
                    ),
                )
            )
            move["global_mcts_margin"] = global_mcts_margin
            move["global_mcts_support_ratio"] = float(
                root_stat.get("support_ratio", 0.0)
            )
            move["global_mcts_signal"] = global_mcts_signal
            move["global_mcts_visit_share"] = float(
                root_stat.get("visit_share", 0.0)
            )
            move["global_mcts_peak_value"] = float(
                root_stat.get("peak_value", move["global_mcts_value"])
            )
            move["global_mcts_visits"] = float(root_stat.get("visits", 0.0))
            move["global_mcts_node_count"] = float(
                global_mcts.get("node_count", 0.0)
            )
            move["global_mcts_max_depth"] = float(
                global_mcts.get("max_depth", 0.0)
            )
            move["global_mcts_search_mode"] = str(
                global_mcts.get("search_mode", "mcts")
            )
            move["global_mcts_simulation_budget"] = float(
                global_mcts.get("simulation_budget", 0.0)
            )
            move["global_mcts_bonus"] = global_mcts_bonus
            move["strategy_objective"] = float(
                move.get(
                    "strategy_objective",
                    move.get("ranking_score", move.get("expected_value", 0.0)),
                )
            ) + global_mcts_bonus
            move["ranking_score"] = float(
                move.get("ranking_score", move.get("expected_value", 0.0))
            ) + (0.45 * global_mcts_bonus)
            score_breakdown = move.get("score_breakdown")
            if isinstance(score_breakdown, dict):
                score_breakdown["global_mcts_value"] = move["global_mcts_value"]
                score_breakdown["global_mcts_margin"] = global_mcts_margin
                score_breakdown["global_mcts_support_ratio"] = move[
                    "global_mcts_support_ratio"
                ]
                score_breakdown["global_mcts_signal"] = global_mcts_signal
                score_breakdown["global_mcts_bonus"] = global_mcts_bonus
                score_breakdown["global_mcts_visit_share"] = move[
                    "global_mcts_visit_share"
                ]
                score_breakdown["global_mcts_node_count"] = move[
                    "global_mcts_node_count"
                ]
                score_breakdown["global_mcts_max_depth"] = move[
                    "global_mcts_max_depth"
                ]
                score_breakdown["global_mcts_search_mode"] = move[
                    "global_mcts_search_mode"
                ]
                score_breakdown["global_mcts_simulation_budget"] = move[
                    "global_mcts_simulation_budget"
                ]
        return self._sort_scored_moves(ranked_moves)

    def choose_best_move(
        self,
        all_moves: List[Dict[str, Any]],
        *,
        risk_factor: float,
        my_hidden_count: int,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        if not all_moves:
            stop_threshold_breakdown = self._stop_threshold_breakdown(
                risk_factor=risk_factor,
                my_hidden_count=my_hidden_count,
                best_move=None,
            )
            stop_threshold = stop_threshold_breakdown["threshold"]
            return None, {
                "evaluated_move_count": 0,
                "best_immediate_value": 0.0,
                "best_expected_value": 0.0,
                "best_win_probability": 0.0,
                "best_continuation_value": 0.0,
                "best_continuation_likelihood": 0.0,
                "best_behavior_guidance_multiplier": self.DEFAULT_BEHAVIOR_GUIDANCE_MULTIPLIER,
                "best_behavior_guidance_signal_count": 0.0,
                "best_behavior_guidance_support": 0.0,
                "best_behavior_guidance_stable_ratio": 0.0,
                "best_information_gain": 0.0,
                "best_ranking_score": 0.0,
                "best_behavior_match_multiplier": self.DEFAULT_BEHAVIOR_MATCH_MULTIPLIER,
                "best_behavior_match_bonus": 0.0,
                "best_behavior_match_support": 0.0,
                "best_behavior_match_decision_bonus": 0.0,
                "best_behavior_match_decision_structure_adjustment": 0.0,
                "best_behavior_match_ranking_bonus": 0.0,
                "best_behavior_match_net_structure": 0.0,
                "best_behavior_match_structure_adjustment": 0.0,
                "best_behavior_match_candidate_confidence": 0.0,
                "best_behavior_match_component_support": 0.0,
                "best_behavior_match_component_strength": 0.0,
                "best_behavior_match_component_penalty": 0.0,
                "best_behavior_match_context_focus": 0.0,
                "best_behavior_rollout_pressure": 0.0,
                "best_post_hit_behavior_support_adjustment": 0.0,
                "best_post_hit_behavior_support_gain": 0.0,
                "best_post_hit_behavior_fragility_drag": 0.0,
                "best_post_hit_continue_score": 0.0,
                "best_post_hit_stop_score": 0.0,
                "best_post_hit_continue_margin": 0.0,
                "best_post_hit_best_gap": 0.0,
                "best_post_hit_guidance_multiplier": self.DEFAULT_BEHAVIOR_GUIDANCE_MULTIPLIER,
                "best_post_hit_guidance_support": 0.0,
                "best_post_hit_guidance_stable_ratio": 0.0,
                "best_post_hit_guidance_signal_count": 0.0,
                "best_post_hit_guidance_rebuild_applied": False,
                "best_post_hit_guidance_rebuild_signal_count": 0.0,
                "best_post_hit_guidance_augmented_slot_count": 0.0,
                "best_post_hit_guidance_multiplier_delta": 0.0,
                "best_post_hit_guidance_source_shift": "neutral",
                "best_post_hit_guidance_source_shift_strength": 0.0,
                "best_post_hit_guidance_rebuild_reason": "post-hit rebuild unavailable",
                "best_post_hit_top_k_expected_continue_margin": 0.0,
                "best_post_hit_top_k_continue_margin": 0.0,
                "best_post_hit_top_k_expected_support_ratio": 0.0,
                "best_post_hit_top_k_support_ratio": 0.0,
                "best_post_hit_failure_recovery_bonus": 0.0,
                "best_post_hit_failed_switch_bonus": 0.0,
                "best_post_hit_failed_switch_signal": 0.0,
                "best_self_public_exposure": 0.0,
                "best_self_newly_drawn_exposure": 0.0,
                "best_self_finish_fragility": 0.0,
                "best_failure_collapse_bonus": 0.0,
                "best_failed_guess_switch_bonus": 0.0,
                "best_failed_guess_switch_continuity_signal": 0.0,
                "best_target_attack_window_signal": 0.0,
                "best_target_attack_window_bonus": 0.0,
                "best_target_attack_window_continuation_bonus": 0.0,
                "best_joint_collapse_signal": 0.0,
                "best_joint_collapse_bonus": 0.0,
                "best_joint_collapse_continuation_bonus": 0.0,
                "best_global_propagation_signal": 0.0,
                "best_global_propagation_bonus": 0.0,
                "best_global_propagation_continuation_bonus": 0.0,
                "best_target_chain_signal": 0.0,
                "best_target_chain_bonus": 0.0,
                "best_target_chain_continuation_bonus": 0.0,
                "best_target_finish_chain_signal": 0.0,
                "best_target_finish_chain_bonus": 0.0,
                "best_target_finish_chain_continuation_bonus": 0.0,
                "best_post_hit_branch_search_value": 0.0,
                "best_post_hit_branch_search_margin": 0.0,
                "best_post_hit_branch_search_support_ratio": 0.0,
                "best_post_hit_branch_search_signal": 0.0,
                "best_post_hit_mcts_value": 0.0,
                "best_post_hit_mcts_margin": 0.0,
                "best_post_hit_mcts_support_ratio": 0.0,
                "best_post_hit_mcts_signal": 0.0,
                "best_post_hit_expectimax_value": 0.0,
                "best_post_hit_expectimax_margin": 0.0,
                "best_post_hit_expectimax_support_ratio": 0.0,
                "best_post_hit_expectimax_signal": 0.0,
                "best_global_mcts_value": 0.0,
                "best_global_mcts_margin": 0.0,
                "best_global_mcts_support_ratio": 0.0,
                "best_global_mcts_signal": 0.0,
                "best_global_mcts_visit_share": 0.0,
                "best_global_mcts_node_count": 0.0,
                "best_global_mcts_max_depth": 0.0,
                "best_strategy_objective": 0.0,
                "best_strategy_objective_core": 0.0,
                "strategy_objective_guess": 0.0,
                "strategy_objective_stop": stop_threshold,
                "strategy_action_margin": -stop_threshold,
                "stop_threshold": stop_threshold,
                "stop_score": stop_threshold,
                "continue_score": 0.0,
                "continue_margin": -stop_threshold,
                "recommend_stop": True,
                "decision_score_breakdown": {
                    "base_stop_threshold": stop_threshold_breakdown["base_stop_threshold"],
                    "short_hand_threshold": stop_threshold_breakdown["short_hand_threshold"],
                    "self_exposure_threshold": stop_threshold_breakdown["self_exposure_threshold"],
                    "newly_drawn_threshold": stop_threshold_breakdown["newly_drawn_threshold"],
                    "finish_fragility_threshold": stop_threshold_breakdown["finish_fragility_threshold"],
                    "low_confidence_threshold": stop_threshold_breakdown["low_confidence_threshold"],
                    "weak_continuation_threshold": stop_threshold_breakdown["weak_continuation_threshold"],
                    "strategy_objective_credit": stop_threshold_breakdown["strategy_objective_credit"],
                    "search_credit": stop_threshold_breakdown["search_credit"],
                    "total_stop_threshold": stop_threshold,
                    "edge_pressure": 0.0,
                    "rollout_pressure": 0.0,
                    "fragile_rollout_pressure": 0.0,
                    "top_k_rollout_pressure": 0.0,
                    "attackability_pressure": 0.0,
                    "behavior_rollout_pressure": 0.0,
                    "post_hit_behavior_support_adjustment": 0.0,
                    "post_hit_behavior_support_gain": 0.0,
                    "post_hit_behavior_fragility_drag": 0.0,
                    "post_hit_behavior_support_signal": 0.0,
                    "post_hit_behavior_fragility_signal": 0.0,
                    "post_hit_behavior_support_strength": 0.0,
                    "post_hit_behavior_fragility_strength": 0.0,
                    "behavior_match_decision_bonus": 0.0,
                    "behavior_match_decision_structure_adjustment": 0.0,
                    "behavior_match_net_structure": 0.0,
                    "behavior_match_ranking_bonus": 0.0,
                    "behavior_match_structure_adjustment": 0.0,
                    "behavior_match_candidate_confidence": 0.0,
                    "behavior_match_component_support": 0.0,
                    "behavior_match_component_strength": 0.0,
                    "behavior_match_component_penalty": 0.0,
                    "behavior_match_context_focus": 0.0,
                    "failure_recovery_pressure": 0.0,
                    "attack_window_support": 0.0,
                    "joint_collapse_support": 0.0,
                    "global_propagation_support": 0.0,
                    "public_reveal_bridge_support": 0.0,
                    "target_chain_support": 0.0,
                    "target_finish_chain_support": 0.0,
                    "branch_search_support": 0.0,
                    "expectimax_support": 0.0,
                    "mcts_support": 0.0,
                    "global_mcts_support": 0.0,
                    "strategy_objective_support": 0.0,
                    "strategy_objective_continue": 0.0,
                    "strategy_objective_stop": stop_threshold,
                    "strategy_action_margin": -stop_threshold,
                    "low_confidence_guard_boost": 0.0,
                    "weak_edge_guard_boost": 0.0,
                    "self_exposure_guard_boost": 0.0,
                },
                "stop_reason": "没有可评估的候选动作。",
            }

        all_moves = self._apply_global_mcts_rerank(
            all_moves=all_moves,
            risk_factor=risk_factor,
            my_hidden_count=my_hidden_count,
        )
        best_move = all_moves[0]
        second_move = all_moves[1] if len(all_moves) > 1 else None
        stop_threshold_breakdown = self._stop_threshold_breakdown(
            risk_factor=risk_factor,
            my_hidden_count=my_hidden_count,
            best_move=best_move,
        )
        stop_threshold = stop_threshold_breakdown["threshold"]
        decision_snapshot = self._evaluate_continue_decision(
            best_move=best_move,
            second_move=second_move,
            stop_threshold=stop_threshold,
            stop_threshold_breakdown=stop_threshold_breakdown,
            my_hidden_count=my_hidden_count,
        )
        stop_reason = self._build_stop_reason(
            should_continue=decision_snapshot["should_continue"],
            best_move=best_move,
            stop_threshold=stop_threshold,
            stop_score=decision_snapshot["stop_score"],
            continue_score=decision_snapshot["continue_score"],
            continue_margin=decision_snapshot["continue_margin"],
            low_confidence_guard=decision_snapshot["low_confidence_guard"],
            weak_edge_guard=decision_snapshot["weak_edge_guard"],
            self_exposure_guard=decision_snapshot["self_exposure_guard"],
        )

        decision_summary = {
            "evaluated_move_count": len(all_moves),
            "best_immediate_value": best_move.get("immediate_expected_value", best_move["expected_value"]),
            "best_expected_value": best_move["expected_value"],
            "best_win_probability": best_move["win_probability"],
            "best_continuation_value": best_move.get("continuation_value", 0.0),
            "best_continuation_likelihood": best_move.get("continuation_likelihood", 0.0),
            "best_behavior_guidance_multiplier": best_move.get("behavior_guidance_multiplier", self.DEFAULT_BEHAVIOR_GUIDANCE_MULTIPLIER),
            "best_behavior_guidance_signal_count": best_move.get("behavior_guidance_signal_count", 0.0),
            "best_behavior_guidance_support": best_move.get("behavior_guidance_support", 0.0),
            "best_behavior_guidance_stable_ratio": best_move.get("behavior_guidance_stable_ratio", 0.0),
            "best_information_gain": best_move.get("information_gain", 0.0),
            "best_ranking_score": best_move.get("ranking_score", best_move["expected_value"]),
            "best_behavior_match_multiplier": best_move.get("behavior_match_multiplier", self.DEFAULT_BEHAVIOR_MATCH_MULTIPLIER),
            "best_behavior_match_bonus": best_move.get("behavior_match_bonus", 0.0),
            "best_behavior_match_support": best_move.get("behavior_match_support", 0.0),
            "best_behavior_match_decision_bonus": decision_snapshot["behavior_match_decision_bonus"],
            "best_behavior_match_decision_structure_adjustment": decision_snapshot["behavior_match_decision_structure_adjustment"],
            "best_behavior_match_ranking_bonus": best_move.get("behavior_match_ranking_bonus", 0.0),
            "best_behavior_match_net_structure": decision_snapshot["behavior_match_net_structure"],
            "best_behavior_match_structure_adjustment": best_move.get("behavior_match_structure_adjustment", 0.0),
            "best_behavior_match_candidate_confidence": decision_snapshot["behavior_match_candidate_confidence"],
            "best_behavior_match_component_support": decision_snapshot["behavior_match_component_support"],
            "best_behavior_match_component_strength": decision_snapshot["behavior_match_component_strength"],
            "best_behavior_match_component_penalty": decision_snapshot["behavior_match_component_penalty"],
            "best_behavior_match_context_focus": decision_snapshot["behavior_match_context_focus"],
            "best_behavior_rollout_pressure": decision_snapshot["decision_score_breakdown"]["behavior_rollout_pressure"],
            "best_post_hit_behavior_support_adjustment": decision_snapshot["decision_score_breakdown"]["post_hit_behavior_support_adjustment"],
            "best_post_hit_behavior_support_gain": decision_snapshot["decision_score_breakdown"]["post_hit_behavior_support_gain"],
            "best_post_hit_behavior_fragility_drag": decision_snapshot["decision_score_breakdown"]["post_hit_behavior_fragility_drag"],
            "best_attackability_after_hit": best_move.get("attackability_after_hit", 0.0),
            "best_post_hit_continue_score": best_move.get("post_hit_continue_score", 0.0),
            "best_post_hit_stop_score": best_move.get("post_hit_stop_score", 0.0),
            "best_post_hit_continue_margin": best_move.get("post_hit_continue_margin", 0.0),
            "best_post_hit_best_gap": best_move.get("post_hit_best_gap", 0.0),
            "best_post_hit_guidance_multiplier": best_move.get("post_hit_guidance_multiplier", self.DEFAULT_BEHAVIOR_GUIDANCE_MULTIPLIER),
            "best_post_hit_guidance_support": best_move.get("post_hit_guidance_support", 0.0),
            "best_post_hit_guidance_stable_ratio": best_move.get("post_hit_guidance_stable_ratio", 0.0),
            "best_post_hit_guidance_signal_count": best_move.get("post_hit_guidance_signal_count", 0.0),
            "best_post_hit_guidance_rebuild_applied": bool(
                best_move.get("post_hit_guidance_debug", {}).get("rebuild_applied", False)
            ),
            "best_post_hit_guidance_rebuild_signal_count": best_move.get(
                "post_hit_guidance_debug",
                {},
            ).get("rebuilt_signal_count", 0.0),
            "best_post_hit_guidance_augmented_slot_count": best_move.get(
                "post_hit_guidance_debug",
                {},
            ).get("augmented_known_slot_count", 0.0),
            "best_post_hit_guidance_multiplier_delta": best_move.get(
                "post_hit_guidance_debug",
                {},
            ).get("blended_delta_from_base", {}).get("guidance_multiplier", 0.0),
            "best_post_hit_guidance_source_shift": best_move.get(
                "post_hit_guidance_debug",
                {},
            ).get("blended_dominant_source_shift", "neutral"),
            "best_post_hit_guidance_source_shift_strength": best_move.get(
                "post_hit_guidance_debug",
                {},
            ).get("blended_dominant_source_shift_strength", 0.0),
            "best_post_hit_guidance_rebuild_reason": self._build_post_hit_guidance_rebuild_reason(
                best_move.get("post_hit_guidance_debug", {})
            ),
            "best_post_hit_top_k_expected_continue_margin": best_move.get("post_hit_top_k_expected_continue_margin", 0.0),
            "best_post_hit_top_k_continue_margin": best_move.get("post_hit_top_k_continue_margin", 0.0),
            "best_post_hit_top_k_expected_support_ratio": best_move.get("post_hit_top_k_expected_support_ratio", 0.0),
            "best_post_hit_top_k_support_ratio": best_move.get("post_hit_top_k_support_ratio", 0.0),
            "best_post_hit_failure_recovery_bonus": best_move.get("post_hit_failure_recovery_bonus", 0.0),
            "best_post_hit_failed_switch_bonus": best_move.get("post_hit_failed_switch_bonus", 0.0),
            "best_post_hit_failed_switch_signal": best_move.get("post_hit_failed_switch_signal", 0.0),
            "best_self_public_exposure": best_move.get("self_public_exposure", 0.0),
            "best_self_newly_drawn_exposure": best_move.get("self_newly_drawn_exposure", 0.0),
            "best_self_finish_fragility": best_move.get("self_finish_fragility", 0.0),
            "best_failure_collapse_bonus": best_move.get("failure_collapse_bonus", 0.0),
            "best_failed_guess_switch_bonus": best_move.get("failed_guess_switch_bonus", 0.0),
            "best_failed_guess_switch_continuity_signal": best_move.get(
                "failed_guess_switch_continuity_signal",
                0.0,
            ),
            "best_target_attack_window_signal": best_move.get("target_attack_window_signal", 0.0),
            "best_target_attack_window_bonus": best_move.get("target_attack_window_bonus", 0.0),
            "best_target_attack_window_continuation_bonus": best_move.get(
                "target_attack_window_continuation_bonus",
                0.0,
            ),
            "best_joint_collapse_signal": best_move.get("joint_collapse_signal", 0.0),
            "best_joint_collapse_bonus": best_move.get("joint_collapse_bonus", 0.0),
            "best_joint_collapse_continuation_bonus": best_move.get(
                "joint_collapse_continuation_bonus",
                0.0,
            ),
            "best_global_propagation_signal": best_move.get("global_propagation_signal", 0.0),
            "best_global_propagation_bonus": best_move.get("global_propagation_bonus", 0.0),
            "best_global_propagation_continuation_bonus": best_move.get(
                "global_propagation_continuation_bonus",
                0.0,
            ),
            "best_target_chain_signal": best_move.get("target_chain_signal", 0.0),
            "best_target_chain_bonus": best_move.get("target_chain_bonus", 0.0),
            "best_target_chain_continuation_bonus": best_move.get(
                "target_chain_continuation_bonus",
                0.0,
            ),
            "best_target_finish_chain_signal": best_move.get("target_finish_chain_signal", 0.0),
            "best_target_finish_chain_bonus": best_move.get("target_finish_chain_bonus", 0.0),
            "best_target_finish_chain_continuation_bonus": best_move.get(
                "target_finish_chain_continuation_bonus",
                0.0,
            ),
            "best_post_hit_branch_search_value": best_move.get("post_hit_branch_search_value", 0.0),
            "best_post_hit_branch_search_margin": best_move.get("post_hit_branch_search_margin", 0.0),
            "best_post_hit_branch_search_support_ratio": best_move.get(
                "post_hit_branch_search_support_ratio",
                0.0,
            ),
            "best_post_hit_branch_search_signal": best_move.get("post_hit_branch_search_signal", 0.0),
            "best_post_hit_mcts_value": best_move.get("post_hit_mcts_value", 0.0),
            "best_post_hit_mcts_margin": best_move.get("post_hit_mcts_margin", 0.0),
            "best_post_hit_mcts_support_ratio": best_move.get(
                "post_hit_mcts_support_ratio",
                0.0,
            ),
            "best_post_hit_mcts_signal": best_move.get("post_hit_mcts_signal", 0.0),
            "best_post_hit_expectimax_value": best_move.get("post_hit_expectimax_value", 0.0),
            "best_post_hit_expectimax_margin": best_move.get("post_hit_expectimax_margin", 0.0),
            "best_post_hit_expectimax_support_ratio": best_move.get(
                "post_hit_expectimax_support_ratio",
                0.0,
            ),
            "best_post_hit_expectimax_signal": best_move.get("post_hit_expectimax_signal", 0.0),
            "best_global_mcts_value": best_move.get("global_mcts_value", 0.0),
            "best_global_mcts_margin": best_move.get("global_mcts_margin", 0.0),
            "best_global_mcts_support_ratio": best_move.get(
                "global_mcts_support_ratio",
                0.0,
            ),
            "best_global_mcts_signal": best_move.get("global_mcts_signal", 0.0),
            "best_global_mcts_visit_share": best_move.get(
                "global_mcts_visit_share",
                0.0,
            ),
            "best_global_mcts_node_count": best_move.get(
                "global_mcts_node_count",
                0.0,
            ),
            "best_global_mcts_max_depth": best_move.get(
                "global_mcts_max_depth",
                0.0,
            ),
            "best_strategy_objective": best_move.get("strategy_objective", 0.0),
            "best_strategy_objective_core": best_move.get("strategy_objective_core", 0.0),
            "strategy_objective_guess": decision_snapshot["strategy_objective_continue"],
            "strategy_objective_stop": decision_snapshot["strategy_objective_stop"],
            "strategy_action_margin": decision_snapshot["strategy_action_margin"],
            "best_gap": decision_snapshot["best_gap"],
            "stop_threshold": stop_threshold,
            "stop_score": decision_snapshot["stop_score"],
            "continue_score": decision_snapshot["continue_score"],
            "continue_margin": decision_snapshot["continue_margin"],
            "recommend_stop": not decision_snapshot["should_continue"],
            "decision_score_breakdown": decision_snapshot["decision_score_breakdown"],
            "stop_reason": stop_reason,
        }
        return (best_move if decision_snapshot["should_continue"] else None), decision_summary

    def benchmark_decision_cases(
        self,
        cases: Sequence[Any],
    ) -> Dict[str, float]:
        expanded_cases = self._expand_decision_benchmark_cases(cases)
        case_count = 0
        correct_count = 0
        continue_margins: List[float] = []
        stop_margins: List[float] = []
        correct_signed_margins: List[float] = []

        for case in expanded_cases:
            my_hidden_count = int(
                case.get("my_hidden_count", 0)
                if isinstance(case, dict)
                else getattr(case, "my_hidden_count", 0)
            )
            moves = list(
                case.get("moves", ())
                if isinstance(case, dict)
                else getattr(case, "moves", ())
            )
            expect_continue = bool(
                case.get("expect_continue", False)
                if isinstance(case, dict)
                else getattr(case, "expect_continue", False)
            )
            best_move, summary = self.choose_best_move(
                moves,
                risk_factor=self.calculate_risk_factor(my_hidden_count),
                my_hidden_count=my_hidden_count,
            )
            predicted_continue = best_move is not None
            continue_margin = float(summary.get("continue_margin", 0.0))

            case_count += 1
            if predicted_continue == expect_continue:
                correct_count += 1
                correct_signed_margins.append(
                    continue_margin if expect_continue else -continue_margin
                )
            if expect_continue:
                continue_margins.append(continue_margin)
            else:
                stop_margins.append(continue_margin)

        average_continue_margin = (
            sum(continue_margins) / len(continue_margins)
            if continue_margins
            else 0.0
        )
        average_stop_margin = (
            sum(stop_margins) / len(stop_margins)
            if stop_margins
            else 0.0
        )
        return {
            "base_case_count": float(len(cases)),
            "case_count": float(case_count),
            "accuracy": (
                float(correct_count) / float(case_count)
                if case_count > 0
                else 0.0
            ),
            "average_continue_margin": average_continue_margin,
            "average_stop_margin": average_stop_margin,
            "margin_separation": average_continue_margin - average_stop_margin,
            "min_correct_margin": (
                min(correct_signed_margins)
                if correct_signed_margins
                else 0.0
            ),
        }

    def benchmark_behavior_cases(
        self,
        cases: Sequence[Any],
        *,
        model: Optional[BehavioralLikelihoodModel] = None,
    ) -> Dict[str, float]:
        behavior_model = model or BehavioralLikelihoodModel()
        expanded_cases = self._expand_behavior_benchmark_cases(cases)
        case_count = 0
        correct_count = 0
        log_margins: List[float] = []
        score_ratios: List[float] = []

        for case in expanded_cases:
            preferred_state = (
                case["preferred_state"]
                if isinstance(case, dict)
                else getattr(case, "preferred_state")
            )
            alternative_state = (
                case["alternative_state"]
                if isinstance(case, dict)
                else getattr(case, "alternative_state")
            )
            preferred_hypothesis = (
                case["preferred_hypothesis"]
                if isinstance(case, dict)
                else getattr(case, "preferred_hypothesis")
            )
            alternative_hypothesis = (
                case["alternative_hypothesis"]
                if isinstance(case, dict)
                else getattr(case, "alternative_hypothesis")
            )

            preferred_score = behavior_model.score_hypothesis(
                preferred_hypothesis,
                behavior_model.build_guess_signals(preferred_state),
                preferred_state,
            )
            alternative_score = behavior_model.score_hypothesis(
                alternative_hypothesis,
                behavior_model.build_guess_signals(alternative_state),
                alternative_state,
            )
            log_margin = log2(
                max(behavior_model.EPSILON, preferred_score)
                / max(behavior_model.EPSILON, alternative_score)
            )

            case_count += 1
            if preferred_score > alternative_score:
                correct_count += 1
            log_margins.append(log_margin)
            score_ratios.append(
                preferred_score / max(behavior_model.EPSILON, alternative_score)
            )

        return {
            "base_case_count": float(len(cases)),
            "case_count": float(case_count),
            "accuracy": (
                float(correct_count) / float(case_count)
                if case_count > 0
                else 0.0
            ),
            "average_log_margin": (
                sum(log_margins) / len(log_margins)
                if log_margins
                else 0.0
            ),
            "min_log_margin": min(log_margins) if log_margins else 0.0,
            "average_score_ratio": (
                sum(score_ratios) / len(score_ratios)
                if score_ratios
                else 0.0
            ),
        }

    def _self_play_benchmark_deck(self) -> List[Card]:
        return [
            (color, value)
            for color in CARD_COLORS
            for value in range(MAX_CARD_VALUE + 1)
        ]

    def _build_self_play_benchmark_world(
        self,
        rng: Random,
        *,
        world_index: int,
    ) -> Dict[str, Any]:
        deck = self._self_play_benchmark_deck()

        def draw_sorted_hand() -> List[Card]:
            sampled = [
                deck.pop(rng.randrange(len(deck)))
                for _ in range(self.SELF_PLAY_BENCHMARK_HAND_SIZE)
            ]
            return sorted(sampled, key=card_sort_key)

        me_hand = draw_sorted_hand()
        opp_hand = draw_sorted_hand()
        side_hand = draw_sorted_hand()
        actual_remaining_cards = sorted(deck, key=card_sort_key)

        opp_hidden_index = rng.randrange(len(opp_hand))
        side_hidden_index = rng.randrange(len(side_hand))
        opp_reveal_indices = {
            index for index in range(len(opp_hand)) if index != opp_hidden_index
        }
        side_reveal_indices = {
            index for index in range(len(side_hand)) if index != side_hidden_index
        }

        def build_slots(
            hand: Sequence[Card],
            *,
            reveal_indices: Set[int],
            is_self: bool = False,
        ) -> List[CardSlot]:
            slots: List[CardSlot] = []
            for slot_index, card in enumerate(hand):
                if is_self:
                    slots.append(
                        CardSlot(
                            slot_index=slot_index,
                            color=card[0],
                            value=card[1],
                            is_revealed=False,
                        )
                    )
                elif slot_index in reveal_indices:
                    slots.append(
                        CardSlot(
                            slot_index=slot_index,
                            color=card[0],
                            value=card[1],
                            is_revealed=True,
                        )
                    )
                else:
                    slots.append(
                        CardSlot(
                            slot_index=slot_index,
                            color=card[0],
                            value=None,
                            is_revealed=False,
                        )
                    )
            return slots

        def off_by_one(card: Card, *, favor_higher: bool) -> int:
            if favor_higher and card[1] < MAX_CARD_VALUE:
                return int(card[1]) + 1
            if int(card[1]) > 0:
                return int(card[1]) - 1
            return int(card[1]) + 1

        actions: List[GuessAction] = []
        opp_hidden_indices = [
            index for index in range(len(opp_hand)) if index not in opp_reveal_indices
        ]
        side_hidden_indices = [
            index for index in range(len(side_hand)) if index not in side_reveal_indices
        ]
        if opp_hidden_indices:
            failed_slot_index = opp_hidden_indices[world_index % len(opp_hidden_indices)]
            failed_card = opp_hand[failed_slot_index]
            actions.append(
                GuessAction(
                    guesser_id="side",
                    target_player_id="opp",
                    target_slot_index=failed_slot_index,
                    guessed_color=failed_card[0],
                    guessed_value=off_by_one(
                        failed_card,
                        favor_higher=(world_index % 2 == 0),
                    ),
                    result=False,
                )
            )
        if side_hidden_indices:
            revealed_slot_index = side_hidden_indices[0]
            revealed_card = side_hand[revealed_slot_index]
            actions.append(
                GuessAction(
                    guesser_id="opp",
                    target_player_id="side",
                    target_slot_index=revealed_slot_index,
                    guessed_color=revealed_card[0],
                    guessed_value=revealed_card[1],
                    result=True,
                    continued_turn=False,
                    revealed_player_id="side",
                    revealed_slot_index=revealed_slot_index,
                    revealed_color=revealed_card[0],
                    revealed_value=revealed_card[1],
                )
            )

        public_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=build_slots(me_hand, reveal_indices=set(), is_self=True),
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=build_slots(opp_hand, reveal_indices=opp_reveal_indices),
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=build_slots(side_hand, reveal_indices=side_reveal_indices),
                ),
            },
            actions=actions,
        )
        truth_by_slot = {
            ("opp", slot_index): card for slot_index, card in enumerate(opp_hand)
        }
        truth_by_slot.update(
            {
                ("side", slot_index): card
                for slot_index, card in enumerate(side_hand)
            }
        )
        return {
            "public_state": public_state,
            "truth_by_slot": truth_by_slot,
            "actual_remaining_cards": actual_remaining_cards,
        }

    def _apply_benchmark_reveal(
        self,
        game_state: GameState,
        *,
        guesser_id: str,
        target_player_id: str,
        target_slot_index: int,
        revealed_card: Card,
        continued_turn: bool,
    ) -> GameState:
        players = {
            player_id: PlayerState(
                player_id=player_id,
                slots=[replace(slot) for slot in player_state.ordered_slots()],
            )
            for player_id, player_state in game_state.players.items()
        }
        actions = list(game_state.actions)
        actions.append(
            GuessAction(
                guesser_id=guesser_id,
                target_player_id=target_player_id,
                target_slot_index=target_slot_index,
                guessed_color=revealed_card[0],
                guessed_value=revealed_card[1],
                result=True,
                continued_turn=continued_turn,
                revealed_player_id=target_player_id,
                revealed_slot_index=target_slot_index,
                revealed_color=revealed_card[0],
                revealed_value=revealed_card[1],
            )
        )
        return GameState(
            self_player_id=game_state.self_player_id,
            target_player_id=game_state.target_player_id,
            players=players,
            actions=actions,
        )

    def _simulate_self_play_turn(
        self,
        game_state: GameState,
        truth_by_slot: Dict[Tuple[str, int], Card],
        *,
        max_steps: int,
    ) -> Dict[str, float]:
        current_state = GameState(
            self_player_id=game_state.self_player_id,
            target_player_id=game_state.target_player_id,
            players={
                player_id: PlayerState(
                    player_id=player_id,
                    slots=[replace(slot) for slot in player_state.ordered_slots()],
                )
                for player_id, player_state in game_state.players.items()
            },
            actions=list(game_state.actions),
        )
        hits = 0.0
        miss = 0.0
        realized_strategy_objective = 0.0
        executed_steps = 0.0
        guessed_any = 0.0
        stopped_cleanly = 0.0

        for step in range(max_steps):
            result = GameController(current_state).run_turn(
                include_draw_color_summary=False,
            )
            decision_summary = result.get("decision_summary", {})
            realized_strategy_objective += float(
                decision_summary.get("best_strategy_objective", 0.0)
            )
            best_move = result.get("best_move")
            if best_move is None and result.get("strategy_phase") == "post_draw_opening":
                top_moves = list(result.get("top_moves", ()))
                if top_moves:
                    best_move = top_moves[0]
            if best_move is None:
                if executed_steps > 0.0 and miss == 0.0:
                    stopped_cleanly = 1.0
                break

            guess_card = best_move.get("guess_card")
            if not isinstance(guess_card, (list, tuple)) or len(guess_card) != 2:
                miss = 1.0
                guessed_any = 1.0
                break

            target_key = (
                str(best_move.get("target_player_id")),
                int(best_move.get("target_slot_index", -1)),
            )
            actual_card = truth_by_slot.get(target_key)
            if actual_card is None or (guess_card[0], guess_card[1]) != actual_card:
                miss = 1.0
                guessed_any = 1.0
                break

            hits += 1.0
            executed_steps += 1.0
            guessed_any = 1.0
            current_state = self._apply_benchmark_reveal(
                current_state,
                guesser_id=current_state.self_player_id,
                target_player_id=target_key[0],
                target_slot_index=target_key[1],
                revealed_card=actual_card,
                continued_turn=(step + 1) < max_steps,
            )

        return {
            "hits": hits,
            "miss": miss,
            "executed_steps": executed_steps,
            "guessed_any": guessed_any,
            "stopped_cleanly": stopped_cleanly,
            "realized_strategy_objective": realized_strategy_objective,
        }

    def _representative_actual_draw_card(
        self,
        actual_remaining_cards: Sequence[Card],
        *,
        color: str,
    ) -> Optional[Card]:
        color_cards = sorted(
            (card for card in actual_remaining_cards if card[0] == color),
            key=card_sort_key,
        )
        if not color_cards:
            return None
        return color_cards[len(color_cards) // 2]

    def _sampled_actual_draw_cards(
        self,
        actual_remaining_cards: Sequence[Card],
        *,
        color: str,
    ) -> List[Card]:
        color_cards = sorted(
            (card for card in actual_remaining_cards if card[0] == color),
            key=card_sort_key,
        )
        if len(color_cards) <= 3:
            return color_cards
        sampled_indices = {0, len(color_cards) // 2, len(color_cards) - 1}
        return [color_cards[index] for index in sorted(sampled_indices)]

    def _draw_color_realized_strategy_objective(
        self,
        game_state: GameState,
        actual_remaining_cards: Sequence[Card],
        color: str,
    ) -> float:
        color_cards = sorted(
            (card for card in actual_remaining_cards if card[0] == color),
            key=card_sort_key,
        )
        if not color_cards:
            return 0.0
        if len(color_cards) <= 3:
            sampled_cards = color_cards
        else:
            sampled_cards = [
                color_cards[0],
                color_cards[len(color_cards) // 2],
                color_cards[-1],
            ]
        base_controller = GameController(game_state)
        realized_objectives: List[float] = []
        for drawn_card in sampled_cards:
            simulated_state = base_controller._simulated_draw_game_state(drawn_card)
            simulated_result = GameController(simulated_state).run_turn(
                include_draw_color_summary=False,
            )
            realized_objectives.append(
                float(
                    simulated_result.get("decision_summary", {}).get(
                        "best_strategy_objective",
                        0.0,
                    )
                )
            )
        return (
            sum(realized_objectives) / len(realized_objectives)
            if realized_objectives
            else 0.0
        )

    def benchmark_self_play_worlds(
        self,
        *,
        world_count: Optional[int] = None,
        seed: int = 7,
    ) -> Dict[str, float]:
        total_worlds = int(world_count or self.SELF_PLAY_BENCHMARK_WORLD_COUNT)
        if total_worlds <= 0:
            return {
                "world_count": 0.0,
                "guess_rate": 0.0,
                "top1_guess_accuracy": 0.0,
                "top3_guess_accuracy": 0.0,
                "recommended_guess_rate": 0.0,
                "recommended_draw_rate": 0.0,
                "recommended_stop_rate": 0.0,
                "draw_color_alignment": 0.0,
                "average_draw_strategy_margin": 0.0,
                "average_expected_probability": 0.0,
                "average_realized_hits_per_turn": 0.0,
                "turn_miss_rate": 0.0,
                "average_realized_strategy_objective": 0.0,
                "average_executed_steps": 0.0,
                "full_turn_guess_execution_rate": 0.0,
                "full_turn_clean_stop_rate": 0.0,
                "deep_rollout_usage": 0.0,
            }

        rng = Random(seed)
        guess_count = 0.0
        top1_guess_hits = 0.0
        top3_guess_hits = 0.0
        expected_probability_sum = 0.0
        recommended_guess_count = 0.0
        recommended_draw_count = 0.0
        recommended_stop_count = 0.0
        draw_alignment_count = 0.0
        comparable_draw_worlds = 0.0
        draw_strategy_margin_sum = 0.0
        realized_hits_sum = 0.0
        turn_miss_sum = 0.0
        realized_strategy_sum = 0.0
        executed_steps_sum = 0.0
        full_turn_guess_execution_sum = 0.0
        full_turn_clean_stop_sum = 0.0
        deep_rollout_count = 0.0

        for world_index in range(total_worlds):
            world = self._build_self_play_benchmark_world(
                rng,
                world_index=world_index,
            )
            public_state = world["public_state"]
            truth_by_slot = world["truth_by_slot"]
            actual_remaining_cards = world["actual_remaining_cards"]
            result = GameController(public_state).run_turn()
            recommended_action = str(result.get("recommended_action", "stop"))
            best_move = result.get("best_move")
            top_moves = list(result.get("top_moves", ()))
            rollout_depth = float(result.get("strategy_rollout_depth", 1.0))
            if rollout_depth >= float(self.DEEP_ROLLOUT_DEPTH):
                deep_rollout_count += 1.0
            if recommended_action == "guess":
                recommended_guess_count += 1.0
            elif recommended_action in {"draw_black", "draw_white"}:
                recommended_draw_count += 1.0
            else:
                recommended_stop_count += 1.0

            draw_summary = result.get("draw_color_summary", {})
            draw_black = self._draw_color_realized_strategy_objective(
                public_state,
                actual_remaining_cards,
                "B",
            )
            draw_white = self._draw_color_realized_strategy_objective(
                public_state,
                actual_remaining_cards,
                "W",
            )
            if draw_summary.get("recommended_color") in CARD_COLORS:
                comparable_draw_worlds += 1.0
                actual_draw_color = "B" if draw_black >= draw_white else "W"
                if draw_summary.get("recommended_color") == actual_draw_color:
                    draw_alignment_count += 1.0
                draw_strategy_margin_sum += abs(draw_black - draw_white)

            recommended_color = str(
                draw_summary.get("recommended_color", result.get("recommended_draw_color", ""))
            )
            if recommended_color not in CARD_COLORS:
                recommended_color = "B" if draw_black >= draw_white else "W"
            sampled_draw_cards = self._sampled_actual_draw_cards(
                actual_remaining_cards,
                color=recommended_color,
            )
            if not sampled_draw_cards:
                fallback_card = self._representative_actual_draw_card(
                    actual_remaining_cards,
                    color="W" if recommended_color == "B" else "B",
                )
                sampled_draw_cards = [fallback_card] if fallback_card is not None else []

            if not sampled_draw_cards:
                turn_metrics = self._simulate_self_play_turn(
                    public_state,
                    truth_by_slot,
                    max_steps=self.SELF_PLAY_BENCHMARK_MAX_STEPS,
                )
                realized_hits_sum += turn_metrics["hits"]
                turn_miss_sum += turn_metrics["miss"]
                realized_strategy_sum += turn_metrics["realized_strategy_objective"]
                executed_steps_sum += turn_metrics["executed_steps"]
                full_turn_guess_execution_sum += turn_metrics["guessed_any"]
                full_turn_clean_stop_sum += turn_metrics["stopped_cleanly"]
                continue

            sample_guess_rate = 0.0
            sample_top1_hits = 0.0
            sample_top3_hits = 0.0
            sample_expected_probability_sum = 0.0
            sample_realized_hits = 0.0
            sample_miss = 0.0
            sample_realized_strategy = 0.0
            sample_executed_steps = 0.0
            sample_guessed_any = 0.0
            sample_stopped_cleanly = 0.0
            for drawn_card in sampled_draw_cards:
                benchmark_state = GameController(public_state)._simulated_draw_game_state(
                    drawn_card
                )
                post_draw_result = GameController(benchmark_state).run_turn(
                    include_draw_color_summary=False,
                )
                post_draw_best_move = post_draw_result.get("best_move")
                post_draw_top_moves = list(post_draw_result.get("top_moves", ()))
                if (
                    post_draw_best_move is None
                    and post_draw_result.get("strategy_phase") == "post_draw_opening"
                    and post_draw_top_moves
                ):
                    post_draw_best_move = post_draw_top_moves[0]
                if post_draw_best_move is not None:
                    sample_guess_rate += 1.0
                    sample_expected_probability_sum += float(
                        post_draw_best_move.get("win_probability", 0.0)
                    )
                    guess_card = post_draw_best_move.get("guess_card")
                    target_key = (
                        str(post_draw_best_move.get("target_player_id")),
                        int(post_draw_best_move.get("target_slot_index", -1)),
                    )
                    actual_card = truth_by_slot.get(target_key)
                    if (
                        actual_card is not None
                        and isinstance(guess_card, (list, tuple))
                        and len(guess_card) == 2
                        and (guess_card[0], guess_card[1]) == actual_card
                    ):
                        sample_top1_hits += 1.0

                post_draw_top3_hit = False
                for move in post_draw_top_moves[:3]:
                    move_card = move.get("guess_card")
                    target_key = (
                        str(move.get("target_player_id")),
                        int(move.get("target_slot_index", -1)),
                    )
                    actual_card = truth_by_slot.get(target_key)
                    if (
                        actual_card is not None
                        and isinstance(move_card, (list, tuple))
                        and len(move_card) == 2
                        and (move_card[0], move_card[1]) == actual_card
                    ):
                        post_draw_top3_hit = True
                        break
                if post_draw_top3_hit:
                    sample_top3_hits += 1.0

                turn_metrics = self._simulate_self_play_turn(
                    benchmark_state,
                    truth_by_slot,
                    max_steps=self.SELF_PLAY_BENCHMARK_MAX_STEPS,
                )
                sample_realized_hits += turn_metrics["hits"]
                sample_miss += turn_metrics["miss"]
                sample_realized_strategy += turn_metrics["realized_strategy_objective"]
                sample_executed_steps += turn_metrics["executed_steps"]
                sample_guessed_any += turn_metrics["guessed_any"]
                sample_stopped_cleanly += turn_metrics["stopped_cleanly"]

            sample_count = float(len(sampled_draw_cards))
            guess_count += sample_guess_rate / sample_count
            top1_guess_hits += sample_top1_hits / sample_count
            top3_guess_hits += sample_top3_hits / sample_count
            expected_probability_sum += sample_expected_probability_sum / sample_count
            realized_hits_sum += sample_realized_hits / sample_count
            turn_miss_sum += sample_miss / sample_count
            realized_strategy_sum += sample_realized_strategy / sample_count
            executed_steps_sum += sample_executed_steps / sample_count
            full_turn_guess_execution_sum += sample_guessed_any / sample_count
            full_turn_clean_stop_sum += sample_stopped_cleanly / sample_count

        return {
            "world_count": float(total_worlds),
            "guess_rate": guess_count / float(total_worlds),
            "top1_guess_accuracy": (
                top1_guess_hits / guess_count if guess_count > 0.0 else 0.0
            ),
            "top3_guess_accuracy": top3_guess_hits / float(total_worlds),
            "recommended_guess_rate": (
                recommended_guess_count / float(total_worlds)
            ),
            "recommended_draw_rate": (
                recommended_draw_count / float(total_worlds)
            ),
            "recommended_stop_rate": (
                recommended_stop_count / float(total_worlds)
            ),
            "draw_color_alignment": (
                draw_alignment_count / comparable_draw_worlds
                if comparable_draw_worlds > 0.0
                else 0.0
            ),
            "average_draw_strategy_margin": (
                draw_strategy_margin_sum / comparable_draw_worlds
                if comparable_draw_worlds > 0.0
                else 0.0
            ),
            "average_expected_probability": (
                expected_probability_sum / guess_count if guess_count > 0.0 else 0.0
            ),
            "average_realized_hits_per_turn": (
                realized_hits_sum / float(total_worlds)
            ),
            "turn_miss_rate": turn_miss_sum / float(total_worlds),
            "average_realized_strategy_objective": (
                realized_strategy_sum / float(total_worlds)
            ),
            "average_executed_steps": executed_steps_sum / float(total_worlds),
            "full_turn_guess_execution_rate": (
                full_turn_guess_execution_sum / float(total_worlds)
            ),
            "full_turn_clean_stop_rate": (
                full_turn_clean_stop_sum / float(total_worlds)
            ),
            "deep_rollout_usage": deep_rollout_count / float(total_worlds),
        }

    def _clone_truth_slots(
        self,
        truth_slots: Sequence[CardSlot],
    ) -> List[CardSlot]:
        return [replace(slot) for slot in truth_slots]

    def _sorted_truth_slots(
        self,
        truth_slots: Sequence[CardSlot],
    ) -> List[CardSlot]:
        return [
            replace(slot, slot_index=index)
            for index, slot in enumerate(
                sorted(
                    self._clone_truth_slots(truth_slots),
                    key=lambda slot: card_sort_key(
                        (
                            slot.color if slot.color in CARD_COLORS else "B",
                            slot.value if slot.value is not None else JOKER,
                        )
                    ),
                )
            )
        ]

    def _remap_actions_for_reindexed_player(
        self,
        actions: Sequence[GuessAction],
        *,
        player_id: str,
        previous_slots: Sequence[CardSlot],
        next_slots: Sequence[CardSlot],
    ) -> List[GuessAction]:
        old_card_by_index = {
            slot.slot_index: slot.known_card()
            for slot in previous_slots
            if slot.known_card() is not None
        }
        new_index_by_card = {
            slot.known_card(): slot.slot_index
            for slot in next_slots
            if slot.known_card() is not None
        }
        remapped_actions: List[GuessAction] = []
        for action in actions:
            updated_target_slot_index = action.target_slot_index
            if (
                action.target_player_id == player_id
                and action.target_slot_index is not None
            ):
                old_card = old_card_by_index.get(action.target_slot_index)
                if old_card in new_index_by_card:
                    updated_target_slot_index = new_index_by_card[old_card]
            updated_revealed_slot_index = action.revealed_slot_index
            if (
                action.revealed_player_id == player_id
                and action.revealed_slot_index is not None
            ):
                old_card = old_card_by_index.get(action.revealed_slot_index)
                if old_card in new_index_by_card:
                    updated_revealed_slot_index = new_index_by_card[old_card]
            remapped_actions.append(
                replace(
                    action,
                    target_slot_index=updated_target_slot_index,
                    revealed_slot_index=updated_revealed_slot_index,
                )
            )
        return remapped_actions

    def _build_long_self_play_world(
        self,
        rng: Random,
    ) -> Dict[str, Any]:
        deck = self._self_play_benchmark_deck()
        rng.shuffle(deck)

        def draw_truth_hand() -> List[CardSlot]:
            hand_cards = sorted(
                [
                    deck.pop()
                    for _ in range(self.LONG_SELF_PLAY_HAND_SIZE)
                ],
                key=card_sort_key,
            )
            return [
                CardSlot(
                    slot_index=index,
                    color=card[0],
                    value=card[1],
                    is_revealed=(index == 0),
                    is_newly_drawn=False,
                )
                for index, card in enumerate(hand_cards)
            ]

        return {
            "truth_slots_by_player": {
                "p0": draw_truth_hand(),
                "p1": draw_truth_hand(),
            },
            "remaining_deck": list(deck),
            "actions": [],
        }

    def _swap_long_self_play_world(
        self,
        world: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "truth_slots_by_player": {
                "p0": self._clone_truth_slots(world["truth_slots_by_player"]["p1"]),
                "p1": self._clone_truth_slots(world["truth_slots_by_player"]["p0"]),
            },
            "remaining_deck": list(world["remaining_deck"]),
            "actions": list(world.get("actions", ())),
        }

    def _build_perspective_state_from_truth_slots(
        self,
        *,
        truth_slots_by_player: Dict[str, Sequence[CardSlot]],
        actions: Sequence[GuessAction],
        self_player_id: str,
        target_player_id: str,
    ) -> GameState:
        players: Dict[str, PlayerState] = {}
        for player_id, truth_slots in truth_slots_by_player.items():
            public_slots: List[CardSlot] = []
            for truth_slot in truth_slots:
                if player_id == self_player_id:
                    public_slots.append(replace(truth_slot))
                else:
                    public_slots.append(
                        CardSlot(
                            slot_index=truth_slot.slot_index,
                            color=truth_slot.color,
                            value=truth_slot.value if truth_slot.is_revealed else None,
                            is_revealed=truth_slot.is_revealed,
                            is_newly_drawn=False,
                        )
                    )
            players[player_id] = PlayerState(player_id=player_id, slots=public_slots)
        return GameState(
            self_player_id=self_player_id,
            target_player_id=target_player_id,
            players=players,
            actions=list(actions),
        )

    def _draw_remaining_card_by_color(
        self,
        remaining_deck: List[Card],
        *,
        color: str,
        rng: Random,
    ) -> Optional[Card]:
        color_indices = [
            index for index, card in enumerate(remaining_deck) if card[0] == color
        ]
        if not color_indices:
            return None
        selected_index = color_indices[rng.randrange(len(color_indices))]
        return remaining_deck.pop(selected_index)

    def _benchmark_draw_color_choice(
        self,
        *,
        truth_slots: Sequence[CardSlot],
        remaining_deck: Sequence[Card],
    ) -> str:
        self_color_counts = {
            color: sum(1 for slot in truth_slots if slot.color == color)
            for color in CARD_COLORS
        }
        remaining_color_counts = {
            color: sum(1 for card in remaining_deck if card[0] == color)
            for color in CARD_COLORS
        }
        color_scores = {
            color: (
                0.65 * remaining_color_counts[color]
                - (0.35 * self_color_counts[color])
            )
            for color in CARD_COLORS
        }
        return max(
            CARD_COLORS,
            key=lambda color: (color_scores[color], remaining_color_counts[color]),
        )

    def _benchmark_recommended_draw_color(
        self,
        *,
        truth_slots_by_player: Dict[str, Sequence[CardSlot]],
        actions: Sequence[GuessAction],
        acting_player_id: str,
        target_player_id: str,
        remaining_deck: Sequence[Card],
    ) -> str:
        pre_draw_state = self._build_perspective_state_from_truth_slots(
            truth_slots_by_player=truth_slots_by_player,
            actions=actions,
            self_player_id=acting_player_id,
            target_player_id=target_player_id,
        )
        controller = GameController(pre_draw_state)
        controller.decision_engine.DEEP_ROLLOUT_DEPTH = (
            self.LONG_SELF_PLAY_PRE_DRAW_ROLLOUT_DEPTH
        )
        controller.decision_engine.DRAW_ROLLOUT_SAMPLE_LIMIT = (
            self.LONG_SELF_PLAY_DRAW_ROLLOUT_SAMPLE_LIMIT
        )
        result = controller.run_turn(include_draw_color_summary=True)
        recommended_color = result.get("recommended_draw_color")
        if recommended_color in CARD_COLORS:
            return str(recommended_color)
        return self._benchmark_draw_color_choice(
            truth_slots=truth_slots_by_player[acting_player_id],
            remaining_deck=remaining_deck,
        )

    def _all_revealed(
        self,
        truth_slots: Sequence[CardSlot],
    ) -> bool:
        return all(slot.is_revealed for slot in truth_slots)

    def _simulate_long_self_play_game_from_world(
        self,
        *,
        world: Dict[str, Any],
        rng: Random,
        starting_player_id: str = "p0",
    ) -> Dict[str, float]:
        truth_slots_by_player = {
            player_id: self._clone_truth_slots(slots)
            for player_id, slots in world["truth_slots_by_player"].items()
        }
        remaining_deck = list(world["remaining_deck"])
        actions: List[GuessAction] = list(world["actions"])
        acting_player_id = starting_player_id
        non_starting_player_id = "p1" if starting_player_id == "p0" else "p0"
        winner: Optional[str] = None
        total_turns = 0.0
        total_guesses = 0.0
        successful_guesses = 0.0
        draw_turns = 0.0
        post_draw_stop_turns = 0.0

        for _ in range(self.LONG_SELF_PLAY_MAX_TURNS):
            total_turns += 1.0
            target_player_id = "p1" if acting_player_id == "p0" else "p0"
            requested_color = self._benchmark_recommended_draw_color(
                truth_slots_by_player=truth_slots_by_player,
                actions=actions,
                acting_player_id=acting_player_id,
                target_player_id=target_player_id,
                remaining_deck=remaining_deck,
            )
            drawn_card = self._draw_remaining_card_by_color(
                remaining_deck,
                color=requested_color,
                rng=rng,
            )
            if drawn_card is None:
                fallback_color = "W" if requested_color == "B" else "B"
                drawn_card = self._draw_remaining_card_by_color(
                    remaining_deck,
                    color=fallback_color,
                    rng=rng,
                )
            if drawn_card is None:
                break

            draw_turns += 1.0
            previous_slots = self._clone_truth_slots(truth_slots_by_player[acting_player_id])
            updated_slots = [
                replace(slot, is_newly_drawn=False)
                for slot in truth_slots_by_player[acting_player_id]
            ]
            updated_slots.append(
                CardSlot(
                    slot_index=-1,
                    color=drawn_card[0],
                    value=drawn_card[1],
                    is_revealed=False,
                    is_newly_drawn=True,
                )
            )
            updated_slots = self._sorted_truth_slots(updated_slots)
            actions = self._remap_actions_for_reindexed_player(
                actions,
                player_id=acting_player_id,
                previous_slots=previous_slots,
                next_slots=updated_slots,
            )
            truth_slots_by_player[acting_player_id] = updated_slots
            drawn_slot_index = next(
                slot.slot_index for slot in updated_slots if slot.is_newly_drawn
            )
            turn_had_guess = False
            last_success_action_index: Optional[int] = None

            while True:
                post_draw_state = self._build_perspective_state_from_truth_slots(
                    truth_slots_by_player=truth_slots_by_player,
                    actions=actions,
                    self_player_id=acting_player_id,
                    target_player_id=target_player_id,
                )
                post_draw_controller = GameController(post_draw_state)
                post_draw_controller.decision_engine.DEEP_ROLLOUT_DEPTH = 1
                post_draw_result = post_draw_controller.run_turn(
                    include_draw_color_summary=False,
                )
                best_move = post_draw_result.get("best_move")
                if (
                    best_move is None
                    and post_draw_result.get("strategy_phase") == "post_draw_opening"
                ):
                    top_moves = list(post_draw_result.get("top_moves", ()))
                    if top_moves:
                        best_move = top_moves[0]
                if best_move is None:
                    if not turn_had_guess:
                        post_draw_stop_turns += 1.0
                    if last_success_action_index is not None:
                        actions[last_success_action_index] = replace(
                            actions[last_success_action_index],
                            continued_turn=False,
                        )
                    break

                total_guesses += 1.0
                turn_had_guess = True
                guess_card = best_move.get("guess_card")
                target_slot_index = int(best_move.get("target_slot_index", -1))
                target_truth_slot = next(
                    (
                        slot
                        for slot in truth_slots_by_player[target_player_id]
                        if slot.slot_index == target_slot_index
                    ),
                    None,
                )
                if (
                    target_truth_slot is None
                    or not isinstance(guess_card, (list, tuple))
                    or len(guess_card) != 2
                ):
                    break
                guessed_card = (guess_card[0], guess_card[1])
                actual_card = target_truth_slot.known_card()
                if actual_card == guessed_card:
                    successful_guesses += 1.0
                    updated_target_slots: List[CardSlot] = []
                    for slot in truth_slots_by_player[target_player_id]:
                        if slot.slot_index == target_slot_index:
                            updated_target_slots.append(replace(slot, is_revealed=True))
                        else:
                            updated_target_slots.append(replace(slot))
                    truth_slots_by_player[target_player_id] = updated_target_slots
                    actions.append(
                        GuessAction(
                            guesser_id=acting_player_id,
                            target_player_id=target_player_id,
                            target_slot_index=target_slot_index,
                            guessed_color=guessed_card[0],
                            guessed_value=guessed_card[1],
                            result=True,
                            continued_turn=True,
                            revealed_player_id=target_player_id,
                            revealed_slot_index=target_slot_index,
                            revealed_color=actual_card[0],
                            revealed_value=actual_card[1],
                        )
                    )
                    last_success_action_index = len(actions) - 1
                    if self._all_revealed(truth_slots_by_player[target_player_id]):
                        winner = acting_player_id
                        break
                    continue

                updated_self_slots = []
                revealed_drawn_card = None
                for slot in truth_slots_by_player[acting_player_id]:
                    if slot.slot_index == drawn_slot_index:
                        revealed_drawn_card = slot.known_card()
                        updated_self_slots.append(replace(slot, is_revealed=True))
                    else:
                        updated_self_slots.append(replace(slot))
                truth_slots_by_player[acting_player_id] = updated_self_slots
                if revealed_drawn_card is not None:
                    actions.append(
                        GuessAction(
                            guesser_id=acting_player_id,
                            target_player_id=target_player_id,
                            target_slot_index=target_slot_index,
                            guessed_color=guessed_card[0],
                            guessed_value=guessed_card[1],
                            result=False,
                            continued_turn=False,
                            revealed_player_id=acting_player_id,
                            revealed_slot_index=drawn_slot_index,
                            revealed_color=revealed_drawn_card[0],
                            revealed_value=revealed_drawn_card[1],
                        )
                    )
                if self._all_revealed(truth_slots_by_player[acting_player_id]):
                    winner = target_player_id
                break

            for player_id, slots in truth_slots_by_player.items():
                truth_slots_by_player[player_id] = [
                    replace(slot, is_newly_drawn=False)
                    for slot in slots
                ]
            if winner is not None:
                break
            acting_player_id = target_player_id

        return {
            "p0_win": 1.0 if winner == "p0" else 0.0,
            "p1_win": 1.0 if winner == "p1" else 0.0,
            "starting_player_win": 1.0 if winner == starting_player_id else 0.0,
            "non_starting_player_win": 1.0
            if winner == non_starting_player_id
            else 0.0,
            "draw_game": 1.0 if winner is None else 0.0,
            "turn_count": total_turns,
            "guess_count": total_guesses,
            "successful_guesses": successful_guesses,
            "post_draw_stop_turns": post_draw_stop_turns,
            "draw_turns": draw_turns,
        }

    def _simulate_long_self_play_game(
        self,
        *,
        rng: Random,
        starting_player_id: str = "p0",
    ) -> Dict[str, float]:
        world = self._build_long_self_play_world(rng)
        return self._simulate_long_self_play_game_from_world(
            world=world,
            rng=rng,
            starting_player_id=starting_player_id,
        )

    def benchmark_long_horizon_self_play(
        self,
        *,
        game_count: Optional[int] = None,
        seed: int = 19,
    ) -> Dict[str, float]:
        total_games = int(game_count or self.LONG_SELF_PLAY_GAME_COUNT)
        if total_games <= 0:
            return {
                "game_count": 0.0,
                "simulated_game_count": 0.0,
                "p0_win_rate": 0.0,
                "p1_win_rate": 0.0,
                "draw_rate": 0.0,
                "average_turn_count": 0.0,
                "average_guess_count": 0.0,
                "average_successful_guesses": 0.0,
                "post_draw_stop_rate": 0.0,
                "starting_player_win_rate": 0.0,
                "non_starting_player_win_rate": 0.0,
                "starting_player_advantage": 0.0,
                "seat_bias": 0.0,
            }

        rng = Random(seed)
        p0_wins = 0.0
        p1_wins = 0.0
        starting_player_wins = 0.0
        non_starting_player_wins = 0.0
        draw_games = 0.0
        turn_sum = 0.0
        guess_sum = 0.0
        success_sum = 0.0
        post_draw_stop_sum = 0.0
        draw_turn_sum = 0.0
        simulated_game_count = 0.0

        for game_index in range(total_games):
            starting_player_id = "p0" if game_index % 2 == 0 else "p1"
            paired_results: List[Dict[str, float]] = []
            for _ in range(self.LONG_SELF_PLAY_PAIRED_WORLD_REPEATS):
                world = self._build_long_self_play_world(rng)
                mirrored_world = self._swap_long_self_play_world(world)
                mirrored_starting_player_id = (
                    "p1" if starting_player_id == "p0" else "p0"
                )
                pair_seed = rng.randrange(1_000_000_000)
                primary_result = self._simulate_long_self_play_game_from_world(
                    world=world,
                    rng=Random(pair_seed),
                    starting_player_id=starting_player_id,
                )
                mirrored_result = self._simulate_long_self_play_game_from_world(
                    world=mirrored_world,
                    rng=Random(pair_seed),
                    starting_player_id=mirrored_starting_player_id,
                )
                paired_results.append(primary_result)
                paired_results.append(mirrored_result)
                simulated_game_count += 2.0
            result = {
                key: sum(
                    float(paired_result.get(key, 0.0))
                    for paired_result in paired_results
                )
                / float(len(paired_results))
                for key in {
                    "p0_win",
                    "p1_win",
                    "starting_player_win",
                    "non_starting_player_win",
                    "draw_game",
                    "turn_count",
                    "guess_count",
                    "successful_guesses",
                    "post_draw_stop_turns",
                    "draw_turns",
                }
            }
            p0_wins += result["p0_win"]
            p1_wins += result["p1_win"]
            starting_player_wins += result["starting_player_win"]
            non_starting_player_wins += result["non_starting_player_win"]
            draw_games += result["draw_game"]
            turn_sum += result["turn_count"]
            guess_sum += result["guess_count"]
            success_sum += result["successful_guesses"]
            post_draw_stop_sum += result["post_draw_stop_turns"]
            draw_turn_sum += result["draw_turns"]

        return {
            "game_count": float(total_games),
            "simulated_game_count": simulated_game_count,
            "p0_win_rate": p0_wins / float(total_games),
            "p1_win_rate": p1_wins / float(total_games),
            "draw_rate": draw_games / float(total_games),
            "average_turn_count": turn_sum / float(total_games),
            "average_guess_count": guess_sum / float(total_games),
            "average_successful_guesses": success_sum / float(total_games),
            "post_draw_stop_rate": (
                post_draw_stop_sum / draw_turn_sum if draw_turn_sum > 0.0 else 0.0
            ),
            "starting_player_win_rate": starting_player_wins / float(total_games),
            "non_starting_player_win_rate": (
                non_starting_player_wins / float(total_games)
            ),
            "starting_player_advantage": (
                (starting_player_wins - non_starting_player_wins)
                / float(total_games)
            ),
            "seat_bias": abs(p0_wins - p1_wins) / float(total_games),
        }

    def benchmark_long_horizon_league(
        self,
        *,
        match_count: Optional[int] = None,
        games_per_match: Optional[int] = None,
        seed: int = 29,
    ) -> Dict[str, float]:
        total_matches = int(match_count or self.LONG_SELF_PLAY_LEAGUE_MATCH_COUNT)
        match_games = int(
            games_per_match or self.LONG_SELF_PLAY_LEAGUE_GAMES_PER_MATCH
        )
        if total_matches <= 0 or match_games <= 0:
            return {
                "match_count": 0.0,
                "games_per_match": 0.0,
                "total_game_count": 0.0,
                "simulated_game_count": 0.0,
                "p0_win_rate": 0.0,
                "p1_win_rate": 0.0,
                "draw_rate": 0.0,
                "average_turn_count": 0.0,
                "average_guess_count": 0.0,
                "average_successful_guesses": 0.0,
                "average_post_draw_stop_rate": 0.0,
                "starting_player_win_rate": 0.0,
                "non_starting_player_win_rate": 0.0,
                "average_starting_player_advantage": 0.0,
                "stop_rate_stddev": 0.0,
                "seat_bias": 0.0,
                "average_match_seat_bias": 0.0,
            }

        rng = Random(seed)
        total_game_count = float(total_matches * match_games)
        p0_wins = 0.0
        p1_wins = 0.0
        starting_player_win_sum = 0.0
        non_starting_player_win_sum = 0.0
        draw_games = 0.0
        turn_sum = 0.0
        guess_sum = 0.0
        success_sum = 0.0
        stop_rate_sum = 0.0
        seat_bias_sum = 0.0
        starting_advantage_sum = 0.0
        stop_rates: List[float] = []
        simulated_game_count = 0.0

        for _ in range(total_matches):
            match_seed_primary = rng.randrange(1_000_000_000)
            match_seed_secondary = rng.randrange(1_000_000_000)
            primary_benchmark = self.benchmark_long_horizon_self_play(
                game_count=match_games,
                seed=match_seed_primary,
            )
            secondary_benchmark = self.benchmark_long_horizon_self_play(
                game_count=match_games,
                seed=match_seed_secondary,
            )
            match_benchmark = {
                key: (
                    float(primary_benchmark.get(key, 0.0))
                    + float(secondary_benchmark.get(key, 0.0))
                )
                / 2.0
                for key in primary_benchmark
            }
            match_benchmark["simulated_game_count"] = (
                float(primary_benchmark.get("simulated_game_count", 0.0))
                + float(secondary_benchmark.get("simulated_game_count", 0.0))
            )
            p0_wins += match_benchmark["p0_win_rate"] * match_games
            p1_wins += match_benchmark["p1_win_rate"] * match_games
            starting_player_win_sum += (
                match_benchmark["starting_player_win_rate"] * match_games
            )
            non_starting_player_win_sum += (
                match_benchmark["non_starting_player_win_rate"] * match_games
            )
            draw_games += match_benchmark["draw_rate"] * match_games
            turn_sum += match_benchmark["average_turn_count"] * match_games
            guess_sum += match_benchmark["average_guess_count"] * match_games
            success_sum += (
                match_benchmark["average_successful_guesses"] * match_games
            )
            stop_rate_sum += match_benchmark["post_draw_stop_rate"]
            seat_bias_sum += match_benchmark["seat_bias"]
            starting_advantage_sum += match_benchmark["starting_player_advantage"]
            stop_rates.append(match_benchmark["post_draw_stop_rate"])
            simulated_game_count += float(
                match_benchmark.get("simulated_game_count", 0.0)
            )

        average_post_draw_stop_rate = (
            stop_rate_sum / float(total_matches) if total_matches > 0 else 0.0
        )
        stop_rate_variance = (
            sum(
                (stop_rate - average_post_draw_stop_rate) ** 2
                for stop_rate in stop_rates
            )
            / float(len(stop_rates))
            if stop_rates
            else 0.0
        )
        return {
            "match_count": float(total_matches),
            "games_per_match": float(match_games),
            "total_game_count": total_game_count,
            "simulated_game_count": simulated_game_count,
            "p0_win_rate": p0_wins / total_game_count,
            "p1_win_rate": p1_wins / total_game_count,
            "draw_rate": draw_games / total_game_count,
            "average_turn_count": turn_sum / total_game_count,
            "average_guess_count": guess_sum / total_game_count,
            "average_successful_guesses": success_sum / total_game_count,
            "average_post_draw_stop_rate": average_post_draw_stop_rate,
            "starting_player_win_rate": starting_player_win_sum / total_game_count,
            "non_starting_player_win_rate": (
                non_starting_player_win_sum / total_game_count
            ),
            "average_starting_player_advantage": (
                starting_advantage_sum / float(total_matches)
            ),
            "stop_rate_stddev": sqrt(stop_rate_variance),
            "seat_bias": abs(p0_wins - p1_wins) / total_game_count,
            "average_match_seat_bias": seat_bias_sum / float(total_matches),
        }

    def benchmark_long_horizon_matrix(
        self,
        *,
        seeds: Optional[Sequence[int]] = None,
        match_count: Optional[int] = None,
        games_per_match: Optional[int] = None,
    ) -> Dict[str, float]:
        benchmark_seeds = tuple(seeds or (11, 19, 29, 37, 53))
        if not benchmark_seeds:
            return {
                "seed_count": 0.0,
                "match_count": 0.0,
                "games_per_match": 0.0,
                "total_game_count": 0.0,
                "simulated_game_count": 0.0,
                "p0_win_rate": 0.0,
                "p1_win_rate": 0.0,
                "draw_rate": 0.0,
                "average_turn_count": 0.0,
                "average_guess_count": 0.0,
                "average_successful_guesses": 0.0,
                "average_post_draw_stop_rate": 0.0,
                "starting_player_win_rate": 0.0,
                "non_starting_player_win_rate": 0.0,
                "average_starting_player_advantage": 0.0,
                "seat_bias": 0.0,
                "average_match_seat_bias": 0.0,
                "seed_seat_bias_stddev": 0.0,
                "seed_stop_rate_stddev": 0.0,
                "seed_starting_advantage_stddev": 0.0,
            }

        seed_benchmarks = [
            self.benchmark_long_horizon_league(
                match_count=match_count,
                games_per_match=games_per_match,
                seed=int(seed),
            )
            for seed in benchmark_seeds
        ]
        seed_count = float(len(seed_benchmarks))
        total_game_count = sum(
            float(benchmark["total_game_count"]) for benchmark in seed_benchmarks
        )
        if total_game_count <= 0.0:
            total_game_count = 1.0
        simulated_game_count = sum(
            float(benchmark.get("simulated_game_count", 0.0))
            for benchmark in seed_benchmarks
        )
        average_post_draw_stop_rate = (
            sum(
                float(benchmark["average_post_draw_stop_rate"])
                for benchmark in seed_benchmarks
            )
            / seed_count
        )
        seat_bias_values = [
            float(benchmark["seat_bias"]) for benchmark in seed_benchmarks
        ]
        stop_rate_values = [
            float(benchmark["average_post_draw_stop_rate"])
            for benchmark in seed_benchmarks
        ]
        starting_advantage_values = [
            float(benchmark["average_starting_player_advantage"])
            for benchmark in seed_benchmarks
        ]
        return {
            "seed_count": seed_count,
            "match_count": float(seed_benchmarks[0]["match_count"]),
            "games_per_match": float(seed_benchmarks[0]["games_per_match"]),
            "total_game_count": total_game_count,
            "simulated_game_count": simulated_game_count,
            "p0_win_rate": sum(
                float(benchmark["p0_win_rate"]) * float(benchmark["total_game_count"])
                for benchmark in seed_benchmarks
            )
            / total_game_count,
            "p1_win_rate": sum(
                float(benchmark["p1_win_rate"]) * float(benchmark["total_game_count"])
                for benchmark in seed_benchmarks
            )
            / total_game_count,
            "draw_rate": sum(
                float(benchmark["draw_rate"]) * float(benchmark["total_game_count"])
                for benchmark in seed_benchmarks
            )
            / total_game_count,
            "average_turn_count": sum(
                float(benchmark["average_turn_count"])
                * float(benchmark["total_game_count"])
                for benchmark in seed_benchmarks
            )
            / total_game_count,
            "average_guess_count": sum(
                float(benchmark["average_guess_count"])
                * float(benchmark["total_game_count"])
                for benchmark in seed_benchmarks
            )
            / total_game_count,
            "average_successful_guesses": sum(
                float(benchmark["average_successful_guesses"])
                * float(benchmark["total_game_count"])
                for benchmark in seed_benchmarks
            )
            / total_game_count,
            "average_post_draw_stop_rate": average_post_draw_stop_rate,
            "starting_player_win_rate": sum(
                float(benchmark["starting_player_win_rate"])
                * float(benchmark["total_game_count"])
                for benchmark in seed_benchmarks
            )
            / total_game_count,
            "non_starting_player_win_rate": sum(
                float(benchmark["non_starting_player_win_rate"])
                * float(benchmark["total_game_count"])
                for benchmark in seed_benchmarks
            )
            / total_game_count,
            "average_starting_player_advantage": (
                sum(starting_advantage_values) / seed_count
            ),
            "seat_bias": sum(seat_bias_values) / seed_count,
            "average_match_seat_bias": sum(
                float(benchmark["average_match_seat_bias"])
                for benchmark in seed_benchmarks
            )
            / seed_count,
            "seed_seat_bias_stddev": sqrt(
                sum(
                    (value - (sum(seat_bias_values) / seed_count)) ** 2
                    for value in seat_bias_values
                )
                / seed_count
            ),
            "seed_stop_rate_stddev": sqrt(
                sum(
                    (value - average_post_draw_stop_rate) ** 2
                    for value in stop_rate_values
                )
                / seed_count
            ),
            "seed_starting_advantage_stddev": sqrt(
                sum(
                    (
                        value
                        - (sum(starting_advantage_values) / seed_count)
                    )
                    ** 2
                    for value in starting_advantage_values
                )
                / seed_count
            ),
        }

    def _aggregate_long_horizon_matrix_runs(
        self,
        benchmarks: Sequence[Dict[str, float]],
    ) -> Dict[str, float]:
        if not benchmarks:
            return {
                "seed_count": 0.0,
                "match_count": 0.0,
                "games_per_match": 0.0,
                "total_game_count": 0.0,
                "simulated_game_count": 0.0,
                "p0_win_rate": 0.0,
                "p1_win_rate": 0.0,
                "draw_rate": 0.0,
                "average_turn_count": 0.0,
                "average_guess_count": 0.0,
                "average_successful_guesses": 0.0,
                "average_post_draw_stop_rate": 0.0,
                "starting_player_win_rate": 0.0,
                "non_starting_player_win_rate": 0.0,
                "average_starting_player_advantage": 0.0,
                "seat_bias": 0.0,
                "average_match_seat_bias": 0.0,
                "seed_seat_bias_stddev": 0.0,
                "seed_stop_rate_stddev": 0.0,
                "seed_starting_advantage_stddev": 0.0,
            }

        total_game_count = sum(
            float(benchmark.get("total_game_count", 0.0))
            for benchmark in benchmarks
        )
        if total_game_count <= 0.0:
            total_game_count = 1.0
        run_count = float(len(benchmarks))
        seat_bias_values = [
            float(benchmark.get("seat_bias", 0.0))
            for benchmark in benchmarks
        ]
        stop_rate_values = [
            float(benchmark.get("average_post_draw_stop_rate", 0.0))
            for benchmark in benchmarks
        ]
        starting_advantage_values = [
            float(benchmark.get("average_starting_player_advantage", 0.0))
            for benchmark in benchmarks
        ]
        weighted_seat_bias = (
            sum(
                float(benchmark.get("seat_bias", 0.0))
                * float(benchmark.get("total_game_count", 0.0))
                for benchmark in benchmarks
            )
            / total_game_count
        )
        weighted_starting_advantage = (
            sum(
                float(benchmark.get("average_starting_player_advantage", 0.0))
                * float(benchmark.get("total_game_count", 0.0))
                for benchmark in benchmarks
            )
            / total_game_count
        )
        weighted_stop_rate = (
            sum(
                float(benchmark.get("average_post_draw_stop_rate", 0.0))
                * float(benchmark.get("total_game_count", 0.0))
                for benchmark in benchmarks
            )
            / total_game_count
        )
        return {
            "seed_count": sum(
                float(benchmark.get("seed_count", 0.0))
                for benchmark in benchmarks
            ),
            "match_count": sum(
                float(benchmark.get("match_count", 0.0))
                for benchmark in benchmarks
            )
            / run_count,
            "games_per_match": sum(
                float(benchmark.get("games_per_match", 0.0))
                for benchmark in benchmarks
            )
            / run_count,
            "total_game_count": total_game_count,
            "simulated_game_count": sum(
                float(benchmark.get("simulated_game_count", 0.0))
                for benchmark in benchmarks
            ),
            "p0_win_rate": sum(
                float(benchmark.get("p0_win_rate", 0.0))
                * float(benchmark.get("total_game_count", 0.0))
                for benchmark in benchmarks
            )
            / total_game_count,
            "p1_win_rate": sum(
                float(benchmark.get("p1_win_rate", 0.0))
                * float(benchmark.get("total_game_count", 0.0))
                for benchmark in benchmarks
            )
            / total_game_count,
            "draw_rate": sum(
                float(benchmark.get("draw_rate", 0.0))
                * float(benchmark.get("total_game_count", 0.0))
                for benchmark in benchmarks
            )
            / total_game_count,
            "average_turn_count": sum(
                float(benchmark.get("average_turn_count", 0.0))
                * float(benchmark.get("total_game_count", 0.0))
                for benchmark in benchmarks
            )
            / total_game_count,
            "average_guess_count": sum(
                float(benchmark.get("average_guess_count", 0.0))
                * float(benchmark.get("total_game_count", 0.0))
                for benchmark in benchmarks
            )
            / total_game_count,
            "average_successful_guesses": sum(
                float(benchmark.get("average_successful_guesses", 0.0))
                * float(benchmark.get("total_game_count", 0.0))
                for benchmark in benchmarks
            )
            / total_game_count,
            "average_post_draw_stop_rate": weighted_stop_rate,
            "starting_player_win_rate": sum(
                float(benchmark.get("starting_player_win_rate", 0.0))
                * float(benchmark.get("total_game_count", 0.0))
                for benchmark in benchmarks
            )
            / total_game_count,
            "non_starting_player_win_rate": sum(
                float(benchmark.get("non_starting_player_win_rate", 0.0))
                * float(benchmark.get("total_game_count", 0.0))
                for benchmark in benchmarks
            )
            / total_game_count,
            "average_starting_player_advantage": weighted_starting_advantage,
            "seat_bias": weighted_seat_bias,
            "average_match_seat_bias": sum(
                float(benchmark.get("average_match_seat_bias", 0.0))
                * float(benchmark.get("total_game_count", 0.0))
                for benchmark in benchmarks
            )
            / total_game_count,
            "seed_seat_bias_stddev": sqrt(
                sum((value - weighted_seat_bias) ** 2 for value in seat_bias_values)
                / run_count
            ),
            "seed_stop_rate_stddev": sqrt(
                sum((value - weighted_stop_rate) ** 2 for value in stop_rate_values)
                / run_count
            ),
            "seed_starting_advantage_stddev": sqrt(
                sum(
                    (value - weighted_starting_advantage) ** 2
                    for value in starting_advantage_values
                )
                / run_count
            ),
        }

    def benchmark_long_horizon_stability_matrix(
        self,
        *,
        seeds: Optional[Sequence[int]] = None,
        match_counts: Optional[Sequence[int]] = None,
        games_per_match_options: Optional[Sequence[int]] = None,
        minimum_total_game_count: Optional[int] = None,
    ) -> Dict[str, float]:
        benchmark_seeds = tuple(seeds or (11, 19, 29, 37, 53))
        benchmark_match_counts = tuple(match_counts or (4, 6))
        benchmark_games_per_match = tuple(games_per_match_options or (4, 6))
        min_total_games = int(
            minimum_total_game_count or self.LONG_SELF_PLAY_STABILITY_MIN_TOTAL_GAMES
        )
        configurations = [
            (int(match_count), int(games_per_match))
            for match_count in benchmark_match_counts
            for games_per_match in benchmark_games_per_match
            if int(match_count) > 0 and int(games_per_match) > 0
        ]
        if not benchmark_seeds or not configurations:
            return {
                "config_count": 0.0,
                "seed_count": 0.0,
                "matrix_run_count": 0.0,
                "minimum_total_game_count_per_configuration": float(min_total_games),
                "total_game_count": 0.0,
                "simulated_game_count": 0.0,
                "average_turn_count": 0.0,
                "average_guess_count": 0.0,
                "average_successful_guesses": 0.0,
                "average_post_draw_stop_rate": 0.0,
                "starting_player_win_rate": 0.0,
                "non_starting_player_win_rate": 0.0,
                "average_starting_player_advantage": 0.0,
                "seat_bias": 0.0,
                "average_match_seat_bias": 0.0,
                "configuration_seat_bias_stddev": 0.0,
                "configuration_starting_advantage_stddev": 0.0,
                "average_repeats_per_configuration": 0.0,
            }

        aggregated_configurations: List[Dict[str, float]] = []
        repeats_per_configuration: List[float] = []
        for match_count, games_per_match in configurations:
            estimated_total = (
                len(benchmark_seeds) * int(match_count) * int(games_per_match)
            )
            repeat_count = max(
                1,
                ceil(float(min_total_games) / max(1.0, float(estimated_total))),
            )
            repeats_per_configuration.append(float(repeat_count))
            repeated_benchmarks = [
                self.benchmark_long_horizon_matrix(
                    seeds=tuple(
                        int(seed)
                        + (
                            repeat_index
                            * self.LONG_SELF_PLAY_STABILITY_SEED_STRIDE
                        )
                        for seed in benchmark_seeds
                    ),
                    match_count=match_count,
                    games_per_match=games_per_match,
                )
                for repeat_index in range(repeat_count)
            ]
            aggregated_configuration = self._aggregate_long_horizon_matrix_runs(
                repeated_benchmarks
            )
            aggregated_configuration["match_count"] = float(match_count)
            aggregated_configuration["games_per_match"] = float(games_per_match)
            aggregated_configuration["repeat_count"] = float(repeat_count)
            aggregated_configurations.append(aggregated_configuration)

        aggregate = self._aggregate_long_horizon_matrix_runs(aggregated_configurations)
        seat_bias_values = [
            float(benchmark.get("seat_bias", 0.0))
            for benchmark in aggregated_configurations
        ]
        starting_advantage_values = [
            float(benchmark.get("average_starting_player_advantage", 0.0))
            for benchmark in aggregated_configurations
        ]
        config_count = float(len(aggregated_configurations))
        return {
            "config_count": config_count,
            "seed_count": float(len(benchmark_seeds)),
            "matrix_run_count": float(sum(repeats_per_configuration)),
            "minimum_total_game_count_per_configuration": float(min_total_games),
            "total_game_count": float(aggregate["total_game_count"]),
            "simulated_game_count": float(aggregate.get("simulated_game_count", 0.0)),
            "average_turn_count": float(aggregate["average_turn_count"]),
            "average_guess_count": float(aggregate["average_guess_count"]),
            "average_successful_guesses": float(
                aggregate["average_successful_guesses"]
            ),
            "average_post_draw_stop_rate": float(
                aggregate["average_post_draw_stop_rate"]
            ),
            "starting_player_win_rate": float(aggregate["starting_player_win_rate"]),
            "non_starting_player_win_rate": float(
                aggregate["non_starting_player_win_rate"]
            ),
            "average_starting_player_advantage": float(
                aggregate["average_starting_player_advantage"]
            ),
            "seat_bias": float(aggregate["seat_bias"]),
            "average_match_seat_bias": float(aggregate["average_match_seat_bias"]),
            "configuration_seat_bias_stddev": sqrt(
                sum(
                    (value - float(aggregate["seat_bias"])) ** 2
                    for value in seat_bias_values
                )
                / config_count
            ),
            "configuration_starting_advantage_stddev": sqrt(
                sum(
                    (
                        value
                        - float(aggregate["average_starting_player_advantage"])
                    )
                    ** 2
                    for value in starting_advantage_values
                )
                / config_count
            ),
            "average_repeats_per_configuration": (
                sum(repeats_per_configuration) / config_count
            ),
        }

    def _summarize_long_horizon_balance(
        self,
        benchmark: Dict[str, float],
        *,
        benchmark_mode: str,
        seat_bias_stddev_keys: Sequence[str],
    ) -> Dict[str, float]:
        fairness_gap = abs(
            float(benchmark.get("starting_player_win_rate", 0.0))
            - float(benchmark.get("non_starting_player_win_rate", 0.0))
        )
        seat_bias = float(benchmark.get("seat_bias", 0.0))
        seat_bias_stddev = 0.0
        for seat_bias_stddev_key in seat_bias_stddev_keys:
            if seat_bias_stddev_key in benchmark:
                seat_bias_stddev = float(benchmark.get(seat_bias_stddev_key, 0.0))
                break
        balance_score = clamp(
            1.0
            - (
                0.50 * seat_bias
                + 0.35 * fairness_gap
                + 0.15 * seat_bias_stddev
            ),
            0.0,
            1.0,
        )
        return {
            **benchmark,
            "benchmark_mode": benchmark_mode,
            "effective_simulated_game_count": float(
                benchmark.get("simulated_game_count", 0.0)
            ),
            "fairness_gap": fairness_gap,
            "seat_bias_stddev": seat_bias_stddev,
            "balance_score": balance_score,
        }

    def benchmark_long_horizon_nightly_suite(
        self,
        *,
        seeds: Optional[Sequence[int]] = None,
        match_counts: Optional[Sequence[int]] = None,
        games_per_match_options: Optional[Sequence[int]] = None,
        benchmark_mode: str = "configuration",
        minimum_total_game_count: Optional[int] = None,
    ) -> Dict[str, float]:
        normalized_mode = str(benchmark_mode or "configuration").lower()
        if normalized_mode == "stability":
            benchmark = self.benchmark_long_horizon_stability_matrix(
                seeds=seeds or (11, 19, 29),
                match_counts=match_counts or (1, 2),
                games_per_match_options=games_per_match_options or (1, 2),
                minimum_total_game_count=(
                    minimum_total_game_count
                    or self.LONG_SELF_PLAY_EVALUATION_MIN_TOTAL_GAMES
                ),
            )
            return self._summarize_long_horizon_balance(
                benchmark,
                benchmark_mode="nightly_stability",
                seat_bias_stddev_keys=("configuration_seat_bias_stddev",),
            )
        benchmark = self.benchmark_long_horizon_configuration_matrix(
            seeds=seeds or (11, 19, 29),
            match_counts=match_counts or (1, 2),
            games_per_match_options=games_per_match_options or (1, 2),
        )
        return self._summarize_long_horizon_balance(
            benchmark,
            benchmark_mode="nightly_configuration",
            seat_bias_stddev_keys=("config_seat_bias_stddev",),
        )

    def benchmark_long_horizon_evaluation_suite(
        self,
        *,
        seeds: Optional[Sequence[int]] = None,
        match_counts: Optional[Sequence[int]] = None,
        games_per_match_options: Optional[Sequence[int]] = None,
        minimum_total_game_count: Optional[int] = None,
        benchmark_mode: str = "fast",
    ) -> Dict[str, float]:
        normalized_mode = str(benchmark_mode or "fast").lower()
        if normalized_mode in {"nightly", "nightly_configuration", "nightly_stability"}:
            return self.benchmark_long_horizon_nightly_suite(
                seeds=seeds,
                match_counts=match_counts,
                games_per_match_options=games_per_match_options,
                benchmark_mode=(
                    "stability"
                    if normalized_mode == "nightly_stability"
                    else "configuration"
                ),
                minimum_total_game_count=minimum_total_game_count,
            )
        if normalized_mode in {"stability", "full"}:
            benchmark = self.benchmark_long_horizon_stability_matrix(
                seeds=seeds or (11, 19, 29, 37, 53, 71),
                match_counts=match_counts or (4, 6, 8),
                games_per_match_options=games_per_match_options or (4, 6),
                minimum_total_game_count=(
                    minimum_total_game_count
                    or self.LONG_SELF_PLAY_EVALUATION_MIN_TOTAL_GAMES
                ),
            )
            return self._summarize_long_horizon_balance(
                benchmark,
                benchmark_mode="stability",
                seat_bias_stddev_keys=("configuration_seat_bias_stddev",),
            )
        if minimum_total_game_count is not None:
            benchmark = self.benchmark_long_horizon_stability_matrix(
                seeds=seeds or (11, 19, 29),
                match_counts=match_counts or (1,),
                games_per_match_options=games_per_match_options or (1, 2),
                minimum_total_game_count=minimum_total_game_count,
            )
            return self._summarize_long_horizon_balance(
                benchmark,
                benchmark_mode="fast_stability",
                seat_bias_stddev_keys=("configuration_seat_bias_stddev",),
            )
        benchmark = self.benchmark_long_horizon_configuration_matrix(
            seeds=seeds or (11, 19, 29),
            match_counts=match_counts or (1,),
            games_per_match_options=games_per_match_options or (1, 2),
        )
        summary = self._summarize_long_horizon_balance(
            benchmark,
            benchmark_mode="fast_configuration",
            seat_bias_stddev_keys=("config_seat_bias_stddev",),
        )
        summary["minimum_total_game_count_per_configuration"] = float(
            benchmark.get("total_game_count", 0.0)
        )
        return summary

    def benchmark_long_horizon_configuration_matrix(
        self,
        *,
        seeds: Optional[Sequence[int]] = None,
        match_counts: Optional[Sequence[int]] = None,
        games_per_match_options: Optional[Sequence[int]] = None,
    ) -> Dict[str, float]:
        benchmark_seeds = tuple(seeds or (11, 19, 29, 37, 53))
        benchmark_match_counts = tuple(match_counts or (4, 6))
        benchmark_games_per_match = tuple(games_per_match_options or (4, 6))
        configurations = [
            (int(match_count), int(games_per_match))
            for match_count in benchmark_match_counts
            for games_per_match in benchmark_games_per_match
            if int(match_count) > 0 and int(games_per_match) > 0
        ]
        if not benchmark_seeds or not configurations:
            return {
                "config_count": 0.0,
                "seed_count": 0.0,
                "total_game_count": 0.0,
                "simulated_game_count": 0.0,
                "average_turn_count": 0.0,
                "average_guess_count": 0.0,
                "average_successful_guesses": 0.0,
                "average_post_draw_stop_rate": 0.0,
                "starting_player_win_rate": 0.0,
                "non_starting_player_win_rate": 0.0,
                "average_starting_player_advantage": 0.0,
                "seat_bias": 0.0,
                "average_match_seat_bias": 0.0,
                "config_seat_bias_stddev": 0.0,
                "config_starting_advantage_stddev": 0.0,
            }
        configuration_benchmarks = [
            self.benchmark_long_horizon_matrix(
                seeds=benchmark_seeds,
                match_count=match_count,
                games_per_match=games_per_match,
            )
            for match_count, games_per_match in configurations
        ]
        total_game_count = sum(
            float(benchmark["total_game_count"])
            for benchmark in configuration_benchmarks
        )
        if total_game_count <= 0.0:
            total_game_count = 1.0
        simulated_game_count = sum(
            float(benchmark.get("simulated_game_count", 0.0))
            for benchmark in configuration_benchmarks
        )
        seat_bias_values = [
            float(benchmark["seat_bias"])
            for benchmark in configuration_benchmarks
        ]
        starting_advantage_values = [
            float(benchmark["average_starting_player_advantage"])
            for benchmark in configuration_benchmarks
        ]
        configuration_weights = [
            float(benchmark["total_game_count"])
            for benchmark in configuration_benchmarks
        ]
        weighted_total = sum(configuration_weights)
        if weighted_total <= 0.0:
            weighted_total = float(len(configuration_benchmarks))
        config_count = float(len(configuration_benchmarks))
        return {
            "config_count": config_count,
            "seed_count": float(len(benchmark_seeds)),
            "total_game_count": total_game_count,
            "simulated_game_count": simulated_game_count,
            "average_turn_count": sum(
                float(benchmark["average_turn_count"])
                * float(benchmark["total_game_count"])
                for benchmark in configuration_benchmarks
            )
            / total_game_count,
            "average_guess_count": sum(
                float(benchmark["average_guess_count"])
                * float(benchmark["total_game_count"])
                for benchmark in configuration_benchmarks
            )
            / total_game_count,
            "average_successful_guesses": sum(
                float(benchmark["average_successful_guesses"])
                * float(benchmark["total_game_count"])
                for benchmark in configuration_benchmarks
            )
            / total_game_count,
            "average_post_draw_stop_rate": sum(
                float(benchmark["average_post_draw_stop_rate"])
                * float(benchmark["total_game_count"])
                for benchmark in configuration_benchmarks
            )
            / total_game_count,
            "starting_player_win_rate": sum(
                float(benchmark["starting_player_win_rate"])
                * float(benchmark["total_game_count"])
                for benchmark in configuration_benchmarks
            )
            / total_game_count,
            "non_starting_player_win_rate": sum(
                float(benchmark["non_starting_player_win_rate"])
                * float(benchmark["total_game_count"])
                for benchmark in configuration_benchmarks
            )
            / total_game_count,
            "average_starting_player_advantage": sum(
                value * weight
                for value, weight in zip(starting_advantage_values, configuration_weights)
            )
            / weighted_total,
            "seat_bias": sum(
                value * weight
                for value, weight in zip(seat_bias_values, configuration_weights)
            )
            / weighted_total,
            "average_match_seat_bias": sum(
                float(benchmark["average_match_seat_bias"])
                * float(benchmark["total_game_count"])
                for benchmark in configuration_benchmarks
            )
            / weighted_total,
            "config_seat_bias_stddev": sqrt(
                sum(
                    (
                        value
                        - (
                            sum(
                                seat_bias * weight
                                for seat_bias, weight in zip(
                                    seat_bias_values,
                                    configuration_weights,
                                )
                            )
                            / weighted_total
                        )
                    )
                    ** 2
                    for value in seat_bias_values
                )
                / config_count
            ),
            "config_starting_advantage_stddev": sqrt(
                sum(
                    (
                        value
                        - (
                            sum(
                                advantage * weight
                                for advantage, weight in zip(
                                    starting_advantage_values,
                                    configuration_weights,
                                )
                            )
                            / weighted_total
                        )
                    )
                    ** 2
                    for value in starting_advantage_values
                )
                / config_count
            ),
        }

    def _strategy_objective_breakdown(
        self,
        *,
        expected_value: float,
        win_probability: float,
        continuation_likelihood: float,
        attackability_after_hit: float,
        information_gain: float,
        expectimax_margin: float,
        tree_search_signal: float,
        mcts_signal: float,
        support_ratio: float,
        behavior_match_bonus: float,
        post_hit_behavior_support_adjustment: float,
        failure_recovery_signal: float,
        opening_precision_signal: float,
        initiative_recovery_signal: float,
        newly_drawn_exposure: float,
        self_public_exposure: float,
        self_finish_fragility: float,
    ) -> Dict[str, float]:
        confidence_shortfall = clamp((0.5 - win_probability) / 0.5, 0.0, 1.0)
        breakdown = {
            "expected_value": expected_value,
            "win_support": self.STRATEGY_OBJECTIVE_WIN_SCALE * win_probability,
            "continuation_support": (
                self.STRATEGY_OBJECTIVE_CONTINUATION_SCALE
                * continuation_likelihood
            ),
            "attackability_support": (
                self.STRATEGY_OBJECTIVE_ATTACKABILITY_SCALE
                * attackability_after_hit
            ),
            "information_gain_support": (
                self.STRATEGY_OBJECTIVE_INFORMATION_GAIN_SCALE
                * information_gain
            ),
            "expectimax_support": (
                self.STRATEGY_OBJECTIVE_EXPECTIMAX_SCALE
                * clamp(
                    expectimax_margin
                    / self.DRAW_ROLLOUT_CONTINUE_MARGIN_REFERENCE,
                    0.0,
                    1.0,
                )
            ),
            "support_ratio_support": (
                self.STRATEGY_OBJECTIVE_SUPPORT_SCALE * support_ratio
            ),
            "behavior_support": (
                self.STRATEGY_OBJECTIVE_BEHAVIOR_SCALE
                * max(0.0, behavior_match_bonus + post_hit_behavior_support_adjustment)
            ),
            "tree_search_support": (
                self.STRATEGY_OBJECTIVE_SEARCH_SIGNAL_SCALE * tree_search_signal
            ),
            "mcts_support": (
                self.STRATEGY_OBJECTIVE_MCTS_SCALE * mcts_signal
            ),
            "recovery_support": (
                self.STRATEGY_OBJECTIVE_RECOVERY_SCALE * failure_recovery_signal
            ),
            "opening_precision_support": (
                self.STRATEGY_OBJECTIVE_OPENING_PRECISION_SCALE
                * opening_precision_signal
            ),
            "initiative_recovery_support": (
                self.STRATEGY_OBJECTIVE_INITIATIVE_RECOVERY_SCALE
                * initiative_recovery_signal
            ),
            "confidence_drag": (
                self.STRATEGY_OBJECTIVE_CONFIDENCE_DRAG
                * confidence_shortfall
            ),
            "newly_drawn_drag": (
                self.STRATEGY_OBJECTIVE_NEW_DRAWN_DRAG
                * newly_drawn_exposure
            ),
            "self_exposure_drag": (
                self.STRATEGY_OBJECTIVE_SELF_EXPOSURE_DRAG
                * self_public_exposure
            ),
            "finish_fragility_drag": (
                self.STRATEGY_OBJECTIVE_FINISH_FRAGILITY_DRAG
                * self_finish_fragility
            ),
        }
        breakdown["total"] = (
            breakdown["expected_value"]
            + breakdown["win_support"]
            + breakdown["continuation_support"]
            + breakdown["attackability_support"]
            + breakdown["information_gain_support"]
            + breakdown["expectimax_support"]
            + breakdown["support_ratio_support"]
            + breakdown["behavior_support"]
            + breakdown["tree_search_support"]
            + breakdown["mcts_support"]
            + breakdown["recovery_support"]
            + breakdown["opening_precision_support"]
            + breakdown["initiative_recovery_support"]
            - breakdown["confidence_drag"]
            - breakdown["newly_drawn_drag"]
            - breakdown["self_exposure_drag"]
            - breakdown["finish_fragility_drag"]
        )
        return breakdown

    def _expand_decision_benchmark_cases(
        self,
        cases: Sequence[Any],
    ) -> List[Any]:
        expanded_cases: List[Any] = []
        nudges = (0.0, 0.0001, 0.0002, 0.00035, 0.0005)
        for case in cases:
            expect_continue = bool(getattr(case, "expect_continue", False))
            direction = 1.0 if expect_continue else -1.0
            for variant_index, nudge in enumerate(nudges):
                adjusted_moves: List[Dict[str, Any]] = []
                for move_index, move in enumerate(getattr(case, "moves", ())):
                    adjusted_move = dict(move)
                    rank_scale = 1.0 if move_index == 0 else 0.65
                    adjusted_move["expected_value"] = float(
                        adjusted_move.get("expected_value", 0.0)
                    ) + (direction * nudge * rank_scale)
                    adjusted_move["continuation_value"] = float(
                        adjusted_move.get("continuation_value", 0.0)
                    ) + (direction * nudge * rank_scale)
                    adjusted_move["continuation_likelihood"] = clamp(
                        float(adjusted_move.get("continuation_likelihood", 0.0))
                        + (direction * nudge * rank_scale),
                        0.0,
                        1.0,
                    )
                    adjusted_move["win_probability"] = clamp(
                        float(adjusted_move.get("win_probability", 0.0))
                        + (direction * 0.5 * nudge * rank_scale),
                        0.0,
                        1.0,
                    )
                    adjusted_move["post_hit_continue_margin"] = float(
                        adjusted_move.get("post_hit_continue_margin", 0.0)
                    ) + (direction * nudge * rank_scale)
                    adjusted_move["post_hit_top_k_continue_margin"] = float(
                        adjusted_move.get("post_hit_top_k_continue_margin", 0.0)
                    ) + (direction * 0.75 * nudge * rank_scale)
                    adjusted_moves.append(adjusted_move)
                expanded_cases.append(
                    {
                        "name": f"{getattr(case, 'name', 'case')}_variant_{variant_index}",
                        "moves": adjusted_moves,
                        "my_hidden_count": int(getattr(case, "my_hidden_count", 0)),
                        "expect_continue": expect_continue,
                    }
                )
        return expanded_cases

    def _copy_game_state_with_noise(
        self,
        game_state: GameState,
        *,
        variant_index: int,
    ) -> GameState:
        players = {
            player_id: PlayerState(
                player_id=player_id,
                slots=[
                    replace(slot)
                    for slot in player_state.ordered_slots()
                ],
            )
            for player_id, player_state in game_state.players.items()
        }
        actions = list(game_state.actions)
        player_ids = list(game_state.players)
        if len(player_ids) >= 2:
            guesser_id = player_ids[variant_index % len(player_ids)]
            target_player_id = player_ids[(variant_index + 1) % len(player_ids)]
        elif player_ids:
            guesser_id = target_player_id = player_ids[0]
        else:
            guesser_id = target_player_id = None
        if guesser_id is not None and target_player_id is not None:
            target_slots = [
                slot
                for slot in players[target_player_id].ordered_slots()
                if not slot.is_revealed
            ]
            if target_slots:
                target_slot = target_slots[0]
                noise_color = (
                    target_slot.color
                    if target_slot.color in CARD_COLORS
                    else "B"
                )
                noise_value = 2 + variant_index
                actions.append(
                    GuessAction(
                        guesser_id=guesser_id,
                        target_player_id=target_player_id,
                        target_slot_index=target_slot.slot_index,
                        guessed_color=noise_color,
                        guessed_value=min(MAX_CARD_VALUE, noise_value),
                        result=False,
                    )
                )
        return GameState(
            self_player_id=game_state.self_player_id,
            target_player_id=game_state.target_player_id,
            players=players,
            actions=actions,
        )

    def _expand_behavior_benchmark_cases(
        self,
        cases: Sequence[Any],
    ) -> List[Any]:
        expanded_cases: List[Any] = []
        for case in cases:
            expanded_cases.append(case)
            for variant_index in (1, 2, 3):
                expanded_cases.append(
                    {
                        "name": f"{getattr(case, 'name', 'case')}_noise_{variant_index}",
                        "preferred_state": self._copy_game_state_with_noise(
                            getattr(case, "preferred_state"),
                            variant_index=variant_index,
                        ),
                        "alternative_state": self._copy_game_state_with_noise(
                            getattr(case, "alternative_state"),
                            variant_index=variant_index,
                        ),
                        "preferred_hypothesis": dict(
                            getattr(case, "preferred_hypothesis")
                        ),
                        "alternative_hypothesis": dict(
                            getattr(case, "alternative_hypothesis")
                        ),
                    }
                )
        return expanded_cases

    def _score_single_move(
        self,
        *,
        full_probability_matrix: FullProbabilityMatrix,
        my_hidden_count: int,
        risk_factor: float,
        hidden_index_by_slot: Dict[int, int],
        player_id: str,
        slot_index: int,
        card: Card,
        probability: float,
        slot_distribution: Dict[Card, float],
        behavior_model: BehavioralLikelihoodModel,
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        acting_player_id: Optional[str],
        behavior_guidance_profile: Optional[Dict[str, float]],
        game_state: Optional[GameState],
        behavior_map_hypothesis: Optional[Dict[str, Dict[int, Card]]],
        self_exposure_profile: Optional[Dict[str, float]],
        rollout_depth: int,
    ) -> Dict[str, Any]:
        info_gain = self._expected_slot_entropy_reduction(slot_distribution, card)
        continuation_value = 0.0
        continuation_likelihood = 0.0
        attackability_after_hit = 0.0
        history_continue_rate = behavior_model.CONTINUATION_PRIOR_BASE
        next_best_immediate_ev = 0.0
        post_hit_continuation_value = 0.0
        post_hit_continue_score = 0.0
        post_hit_stop_score = 0.0
        post_hit_continue_margin = 0.0
        post_hit_should_continue = False
        post_hit_best_gap = 0.0
        post_hit_gap_adjustment = 1.0
        post_hit_guidance_multiplier = self.DEFAULT_BEHAVIOR_GUIDANCE_MULTIPLIER
        post_hit_guidance_support = 0.0
        post_hit_guidance_stable_ratio = 0.0
        post_hit_guidance_signal_count = 0.0
        post_hit_guidance_debug: Dict[str, Any] = self._default_post_hit_guidance_rebuild_debug(
            base_profile=behavior_guidance_profile,
            acting_player_id=acting_player_id,
        )
        post_hit_top_k_expected_continue_margin = 0.0
        post_hit_top_k_continue_margin = 0.0
        post_hit_top_k_expected_support_ratio = 0.0
        post_hit_top_k_support_ratio = 0.0
        post_hit_top_k_positive_count = 0.0
        post_hit_branch_search_value = 0.0
        post_hit_branch_search_margin = 0.0
        post_hit_branch_search_support_ratio = 0.0
        post_hit_branch_search_signal = 0.0
        post_hit_tree_search_value = 0.0
        post_hit_tree_search_margin = 0.0
        post_hit_tree_search_support_ratio = 0.0
        post_hit_tree_search_signal = 0.0
        post_hit_mcts_value = 0.0
        post_hit_mcts_margin = 0.0
        post_hit_mcts_support_ratio = 0.0
        post_hit_mcts_signal = 0.0
        post_hit_mcts_node_count = 0.0
        post_hit_mcts_max_depth = 0.0
        post_hit_mcts_search_mode = "mcts"
        post_hit_mcts_simulation_budget = 0.0
        post_hit_expectimax_value = 0.0
        post_hit_expectimax_margin = 0.0
        post_hit_expectimax_support_ratio = 0.0
        post_hit_expectimax_signal = 0.0
        post_hit_game_state_for_search: Optional[GameState] = None
        post_hit_guess_signals_for_search: Optional[Dict[str, Sequence[GuessSignal]]] = None
        post_hit_behavior_guidance_profile_for_search: Optional[Dict[str, float]] = None
        post_hit_behavior_map_hypothesis_for_search: Optional[Dict[str, Dict[int, Card]]] = None
        post_hit_behavior_support_adjustment = 0.0
        post_hit_behavior_support_gain = 0.0
        post_hit_behavior_fragility_drag = 0.0
        post_hit_behavior_support_signal = 0.0
        post_hit_behavior_fragility_signal = 0.0
        post_hit_behavior_support_strength = 0.0
        post_hit_behavior_fragility_strength = 0.0
        post_hit_failure_recovery_bonus = 0.0
        post_hit_failed_switch_bonus = 0.0
        post_hit_failed_switch_signal = 0.0
        target_attack_window_signal = 0.0
        target_attack_window_bonus = 0.0
        target_attack_window_continuation_bonus = 0.0
        joint_collapse_signal = 0.0
        joint_collapse_bonus = 0.0
        joint_collapse_continuation_bonus = 0.0
        global_propagation_signal = 0.0
        global_propagation_bonus = 0.0
        global_propagation_continuation_bonus = 0.0
        public_reveal_bridge_signal = 0.0
        public_reveal_bridge_bonus = 0.0
        public_reveal_bridge_continuation_bonus = 0.0
        target_chain_signal = 0.0
        target_chain_bonus = 0.0
        target_chain_continuation_bonus = 0.0
        target_finish_chain_signal = 0.0
        target_finish_chain_bonus = 0.0
        target_finish_chain_continuation_bonus = 0.0
        continuation_exposure_gate = 1.0
        behavior_guidance_multiplier = self.DEFAULT_BEHAVIOR_GUIDANCE_MULTIPLIER
        behavior_guidance_support = 0.0
        behavior_guidance_stable_ratio = 0.0
        behavior_guidance_signal_count = 0.0
        behavior_match_multiplier = self.DEFAULT_BEHAVIOR_MATCH_MULTIPLIER
        behavior_match_bonus = 0.0
        behavior_match_ranking_bonus = 0.0
        behavior_match_net_structure = 0.0
        behavior_match_structure_adjustment = 0.0
        behavior_match_support = 0.0
        opening_precision_support = 0.0
        opening_precision_signal = 0.0
        opening_probability_signal = 0.0
        opening_margin_signal = 0.0
        opening_behavior_posterior_support = 0.0
        opening_joker_penalty = 0.0
        opening_numeric_competition_signal = 0.0
        initiative_recovery_signal = 0.0
        strategy_objective_core = 0.0
        strategy_objective = 0.0
        strategy_objective_breakdown: Dict[str, float] = {
            "expected_value": 0.0,
            "win_support": 0.0,
            "continuation_support": 0.0,
            "attackability_support": 0.0,
            "information_gain_support": 0.0,
            "expectimax_support": 0.0,
            "support_ratio_support": 0.0,
            "behavior_support": 0.0,
            "tree_search_support": 0.0,
            "mcts_support": 0.0,
            "recovery_support": 0.0,
            "opening_precision_support": 0.0,
            "initiative_recovery_support": 0.0,
            "confidence_drag": 0.0,
            "newly_drawn_drag": 0.0,
            "self_exposure_drag": 0.0,
            "finish_fragility_drag": 0.0,
            "total": 0.0,
        }
        behavior_match_confidence_breakdown = {
            "candidate_confidence": 1.0,
            "component_support": 1.0,
            "component_strength": 1.0,
            "component_penalty": 0.0,
            "context_focus": 1.0,
        }
        behavior_candidate_signal = {
            "signal_tags": [],
            "dominant_signal": {
                "source": "neutral",
                "reason": "neutral",
                "weight": 1.0,
            },
        }
        behavior_action_posterior_summary = {
            "probability": 0.0,
            "weight": 1.0,
            "observed_rank": 1.0,
            "support": 0.0,
        }
        self_exposure_profile = self_exposure_profile or {
            "total_exposure": 0.0,
            "max_slot_exposure": 0.0,
            "newly_drawn_exposure": 0.0,
            "average_slot_exposure": 0.0,
            "finish_fragility": 0.0,
            "hidden_count": 0.0,
        }

        success_matrix = self._success_posterior(full_probability_matrix, player_id, slot_index, card)
        if success_matrix:
            next_best_immediate_ev = self._best_immediate_ev(
                success_matrix,
                my_hidden_count,
            )
            continuation_assessment = behavior_model.estimate_continue_likelihood(
                success_matrix,
                guess_signals_by_player,
                acting_player_id,
                exclude_slot=slot_key(player_id, slot_index),
            )
            continuation_likelihood = continuation_assessment["continue_likelihood"]
            attackability_after_hit = continuation_assessment["attackability"]
            history_continue_rate = continuation_assessment["history_continue_rate"]
            if behavior_guidance_profile is not None:
                behavior_guidance_multiplier = max(
                    0.0,
                    float(behavior_guidance_profile.get("guidance_multiplier", self.DEFAULT_BEHAVIOR_GUIDANCE_MULTIPLIER)),
                )
                behavior_guidance_support = float(behavior_guidance_profile.get("average_posterior_support", 0.0))
                behavior_guidance_stable_ratio = float(behavior_guidance_profile.get("stable_signal_ratio", 0.0))
                behavior_guidance_signal_count = float(behavior_guidance_profile.get("signal_count", 0.0))
                continuation_likelihood = clamp(
                    continuation_likelihood * behavior_guidance_multiplier,
                    behavior_model.CONTINUATION_MIN,
                    behavior_model.CONTINUATION_MAX,
                )
            if rollout_depth > 0:
                post_hit_rollout = self._evaluate_post_hit_rollout(
                    success_matrix=success_matrix,
                    my_hidden_count=my_hidden_count,
                    behavior_model=behavior_model,
                    guess_signals_by_player=guess_signals_by_player,
                    acting_player_id=acting_player_id,
                    behavior_guidance_profile=behavior_guidance_profile,
                    game_state=game_state,
                    target_player_id=player_id,
                    target_slot_index=slot_index,
                    guessed_card=card,
                    rollout_depth=rollout_depth - 1,
                )
                post_hit_continue_score = post_hit_rollout["continue_score"]
                post_hit_stop_score = post_hit_rollout["stop_score"]
                post_hit_continue_margin = post_hit_rollout["continue_margin"]
                post_hit_should_continue = post_hit_rollout["should_continue"]
                post_hit_best_gap = post_hit_rollout["best_gap"]
                post_hit_guidance_multiplier = post_hit_rollout["behavior_guidance_multiplier"]
                post_hit_guidance_support = post_hit_rollout["behavior_guidance_support"]
                post_hit_guidance_stable_ratio = post_hit_rollout["behavior_guidance_stable_ratio"]
                post_hit_guidance_signal_count = post_hit_rollout["behavior_guidance_signal_count"]
                post_hit_guidance_debug = post_hit_rollout["guidance_debug"]
                post_hit_top_k_expected_continue_margin = post_hit_rollout["top_k_expected_continue_margin"]
                post_hit_top_k_continue_margin = post_hit_rollout["top_k_continue_margin"]
                post_hit_top_k_expected_support_ratio = post_hit_rollout["top_k_expected_support_ratio"]
                post_hit_top_k_support_ratio = post_hit_rollout["top_k_support_ratio"]
                post_hit_top_k_positive_count = post_hit_rollout["top_k_positive_count"]
                post_hit_branch_search_value = post_hit_rollout["branch_search_value"]
                post_hit_branch_search_margin = post_hit_rollout["branch_search_margin"]
                post_hit_branch_search_support_ratio = post_hit_rollout["branch_search_support_ratio"]
                post_hit_branch_search_signal = post_hit_rollout["branch_search_signal"]
                post_hit_tree_search_value = post_hit_rollout["tree_search_value"]
                post_hit_tree_search_margin = post_hit_rollout["tree_search_margin"]
                post_hit_tree_search_support_ratio = post_hit_rollout["tree_search_support_ratio"]
                post_hit_tree_search_signal = post_hit_rollout["tree_search_signal"]
                post_hit_mcts_value = post_hit_rollout["mcts_value"]
                post_hit_mcts_margin = post_hit_rollout["mcts_margin"]
                post_hit_mcts_support_ratio = post_hit_rollout["mcts_support_ratio"]
                post_hit_mcts_signal = post_hit_rollout["mcts_signal"]
                post_hit_mcts_node_count = post_hit_rollout["mcts_node_count"]
                post_hit_mcts_max_depth = post_hit_rollout["mcts_max_depth"]
                post_hit_mcts_search_mode = post_hit_rollout.get(
                    "mcts_search_mode",
                    "mcts",
                )
                post_hit_mcts_simulation_budget = float(
                    post_hit_rollout.get("mcts_simulation_budget", 0.0)
                )
                post_hit_expectimax_value = post_hit_rollout["expectimax_value"]
                post_hit_expectimax_margin = post_hit_rollout["expectimax_margin"]
                post_hit_expectimax_support_ratio = post_hit_rollout["expectimax_support_ratio"]
                post_hit_expectimax_signal = post_hit_rollout["expectimax_signal"]
                post_hit_game_state_for_search = post_hit_rollout.get("post_hit_game_state")
                post_hit_guess_signals_for_search = post_hit_rollout.get("post_hit_guess_signals_by_player")
                post_hit_behavior_guidance_profile_for_search = post_hit_rollout.get(
                    "post_hit_behavior_guidance_profile"
                )
                post_hit_behavior_map_hypothesis_for_search = post_hit_rollout.get(
                    "post_hit_behavior_map_hypothesis"
                )
                post_hit_failure_recovery_bonus = post_hit_rollout["failure_recovery_bonus"]
                post_hit_failed_switch_bonus = post_hit_rollout["failed_switch_bonus"]
                post_hit_failed_switch_signal = post_hit_rollout["failed_switch_signal"]
                if post_hit_continue_margin > 0.0 and post_hit_best_gap < self.POST_HIT_GAP_REFERENCE:
                    post_hit_gap_adjustment = max(
                        0.35,
                        post_hit_best_gap / self.POST_HIT_GAP_REFERENCE,
                    )
                rollout_base_weight = max(
                    0.0,
                    1.0
                    - self.CONTINUATION_TOP_K_BLEND
                    - self.CONTINUATION_BRANCH_SEARCH_BLEND
                    - self.CONTINUATION_EXPECTIMAX_BLEND,
                )
                rollout_margin_basis = (
                    (rollout_base_weight * max(0.0, post_hit_continue_margin))
                    + (self.CONTINUATION_TOP_K_BLEND * post_hit_top_k_continue_margin)
                    + (
                        self.CONTINUATION_BRANCH_SEARCH_BLEND
                        * post_hit_branch_search_margin
                    )
                    + (
                        self.CONTINUATION_EXPECTIMAX_BLEND
                        * post_hit_expectimax_margin
                    )
                )
                post_hit_continuation_value = (
                    self.CONTINUATION_DISCOUNT
                    * continuation_likelihood
                    * max(0.0, rollout_margin_basis)
                    * post_hit_gap_adjustment
                )
                post_hit_continuation_value += (
                    self.POST_HIT_FAILURE_RECOVERY_SCALE
                    * (
                        post_hit_failure_recovery_bonus
                        + post_hit_failed_switch_bonus
                    )
                )
                continuation_value = probability * post_hit_continuation_value
                continuation_exposure_gate = clamp(
                    1.0
                    - (
                        self.CONTINUATION_SELF_EXPOSURE_DRAG
                        * float(self_exposure_profile["total_exposure"])
                    )
                    - (
                        self.CONTINUATION_FINISH_FRAGILITY_DRAG
                        * float(self_exposure_profile["finish_fragility"])
                    ),
                    0.45,
                    1.0,
                )
                continuation_value *= continuation_exposure_gate
                post_hit_behavior_support_breakdown = self._post_hit_behavior_support_breakdown(
                    best_move={
                        "post_hit_stop_score": post_hit_stop_score,
                        "post_hit_continue_margin": post_hit_continue_margin,
                        "continuation_value": continuation_value,
                        "post_hit_top_k_expected_continue_margin": post_hit_top_k_expected_continue_margin,
                        "post_hit_top_k_continue_margin": post_hit_top_k_continue_margin,
                        "post_hit_top_k_expected_support_ratio": post_hit_top_k_expected_support_ratio,
                        "post_hit_top_k_support_ratio": post_hit_top_k_support_ratio,
                        "post_hit_guidance_multiplier": post_hit_guidance_multiplier,
                        "post_hit_guidance_support": post_hit_guidance_support,
                        "post_hit_guidance_stable_ratio": post_hit_guidance_stable_ratio,
                    }
                )
                post_hit_behavior_support_adjustment = post_hit_behavior_support_breakdown["adjustment"]
                post_hit_behavior_support_gain = post_hit_behavior_support_breakdown["support_gain"]
                post_hit_behavior_fragility_drag = post_hit_behavior_support_breakdown["fragility_drag"]
                post_hit_behavior_support_signal = post_hit_behavior_support_breakdown["support_signal"]
                post_hit_behavior_fragility_signal = post_hit_behavior_support_breakdown["fragility_signal"]
                post_hit_behavior_support_strength = post_hit_behavior_support_breakdown["support_strength"]
                post_hit_behavior_fragility_strength = post_hit_behavior_support_breakdown["fragility_strength"]

        self_exposure_profile = self_exposure_profile or {
            "total_exposure": 0.0,
            "max_slot_exposure": 0.0,
            "newly_drawn_exposure": 0.0,
            "average_slot_exposure": 0.0,
            "finish_fragility": 0.0,
            "hidden_count": 0.0,
        }
        failed_guess_slot_pressure = 0.0
        failed_guess_neighbor_pressure = 0.0
        failed_guess_player_pressure = 0.0
        if game_state is not None:
            failed_guess_slot_pressure = behavior_model._recent_failed_guess_pressure(
                game_state,
                player_id,
                slot_index,
            )
            failed_guess_neighbor_pressure = (
                behavior_model._recent_failed_guess_neighbor_pressure(
                    game_state,
                    player_id,
                    slot_index,
                )
            )
            failed_guess_player_pressure = behavior_model._recent_failed_guess_pressure(
                game_state,
                player_id,
            )
        failed_guess_switch_continuity_signal = (
            self._recent_failed_guess_switch_continuity_signal(
                game_state,
                acting_player_id,
                player_id,
                card,
            )
        )
        if game_state is not None:
            target_attack_window_signal = clamp(
                (
                    0.65
                    * behavior_model._slot_attack_window_pressure(
                        game_state,
                        player_id,
                        slot_index,
                    )
                )
                + (
                    0.35
                    * behavior_model._player_attack_window_pressure(
                        game_state,
                        player_id,
                        exclude_slot=slot_key(player_id, slot_index),
                    )
                ),
                0.0,
                1.0,
            )
            joint_collapse_signal = clamp(
                (
                    0.45
                    * behavior_model._recent_player_collapse_streak_pressure(
                        game_state,
                        player_id,
                    )
                )
                + (
                    0.25
                    * behavior_model._global_public_collapse_pressure(game_state)
                )
                + (
                    0.30
                    * behavior_model._global_public_propagation_pressure(game_state)
                ),
                0.0,
                1.0,
            )
            global_propagation_signal = behavior_model._global_public_propagation_pressure(
                game_state
            )
            public_reveal_bridge_signal = self._recent_public_reveal_bridge_signal(
                game_state,
                player_id,
                card,
            )
            target_chain_signal = self._recent_target_chain_signal(
                game_state,
                player_id,
            )
            target_finish_chain_signal = clamp(
                behavior_model._player_finish_pressure(
                    game_state,
                    player_id,
                    exclude_slot=slot_key(player_id, slot_index),
                ),
                0.0,
                1.0,
            )
        structural_risk_factor = risk_factor * (
            1.0
            + (self.SELF_EXPOSURE_RISK_SCALE * float(self_exposure_profile["total_exposure"]))
            + (self.SELF_NEW_DRAWN_RISK_SCALE * float(self_exposure_profile["newly_drawn_exposure"]))
        )
        hit_reward = probability * self.HIT_REWARD
        miss_penalty = (1.0 - probability) * structural_risk_factor
        info_bonus = info_gain * self.INFORMATION_GAIN_WEIGHT
        failure_collapse_bonus = (
            (self.FAILED_GUESS_SLOT_COLLAPSE_VALUE_BONUS * failed_guess_slot_pressure)
            + (
                self.FAILED_GUESS_NEIGHBOR_COLLAPSE_VALUE_BONUS
                * failed_guess_neighbor_pressure
            )
            + (
                self.FAILED_GUESS_PLAYER_COLLAPSE_VALUE_BONUS
                * max(0.0, failed_guess_player_pressure - failed_guess_slot_pressure)
            )
        ) * max(probability, attackability_after_hit, 0.25)
        failed_guess_switch_bonus = (
            self.FAILED_GUESS_SWITCH_CONTINUITY_VALUE_BONUS
            * failed_guess_switch_continuity_signal
            * max(probability, attackability_after_hit, 0.25)
        )
        target_attack_window_bonus = (
            self.TARGET_ATTACK_WINDOW_VALUE_BONUS
            * target_attack_window_signal
            * max(probability, attackability_after_hit, continuation_likelihood, 0.25)
        )
        target_attack_window_continuation_bonus = (
            self.TARGET_ATTACK_WINDOW_CONTINUATION_SCALE
            * target_attack_window_signal
            * max(continuation_likelihood, attackability_after_hit, 0.25)
        )
        continuation_value += target_attack_window_continuation_bonus
        joint_collapse_bonus = (
            self.JOINT_COLLAPSE_VALUE_BONUS
            * joint_collapse_signal
            * max(probability, attackability_after_hit, continuation_likelihood, 0.25)
        )
        joint_collapse_continuation_bonus = (
            self.JOINT_COLLAPSE_CONTINUATION_SCALE
            * joint_collapse_signal
            * max(continuation_likelihood, attackability_after_hit, 0.25)
        )
        continuation_value += joint_collapse_continuation_bonus
        global_propagation_bonus = (
            self.GLOBAL_PROPAGATION_VALUE_BONUS
            * global_propagation_signal
            * max(probability, attackability_after_hit, continuation_likelihood, 0.25)
        )
        global_propagation_continuation_bonus = (
            self.GLOBAL_PROPAGATION_CONTINUATION_SCALE
            * global_propagation_signal
            * max(continuation_likelihood, attackability_after_hit, 0.25)
        )
        continuation_value += global_propagation_continuation_bonus
        public_reveal_bridge_bonus = (
            self.PUBLIC_REVEAL_BRIDGE_VALUE_BONUS
            * public_reveal_bridge_signal
            * max(probability, attackability_after_hit, continuation_likelihood, 0.25)
        )
        public_reveal_bridge_continuation_bonus = (
            self.PUBLIC_REVEAL_BRIDGE_CONTINUATION_SCALE
            * public_reveal_bridge_signal
            * max(continuation_likelihood, attackability_after_hit, 0.25)
        )
        continuation_value += public_reveal_bridge_continuation_bonus
        target_chain_bonus = (
            self.TARGET_CHAIN_VALUE_BONUS
            * target_chain_signal
            * max(probability, attackability_after_hit, continuation_likelihood, 0.25)
        )
        target_chain_continuation_bonus = (
            self.TARGET_CHAIN_CONTINUATION_SCALE
            * target_chain_signal
            * max(continuation_likelihood, attackability_after_hit, 0.25)
        )
        continuation_value += target_chain_continuation_bonus
        target_finish_chain_bonus = (
            self.TARGET_FINISH_CHAIN_VALUE_BONUS
            * target_finish_chain_signal
            * max(probability, attackability_after_hit, continuation_likelihood, 0.25)
        )
        target_finish_chain_continuation_bonus = (
            self.TARGET_FINISH_CHAIN_CONTINUATION_SCALE
            * target_finish_chain_signal
            * max(continuation_likelihood, attackability_after_hit, 0.25)
        )
        continuation_value += target_finish_chain_continuation_bonus
        immediate_expected_value = (
            hit_reward
            - miss_penalty
            + info_bonus
            + failure_collapse_bonus
            + failed_guess_switch_bonus
            + target_attack_window_bonus
            + joint_collapse_bonus
            + global_propagation_bonus
            + public_reveal_bridge_bonus
            + target_chain_bonus
            + target_finish_chain_bonus
        )
        expected_value = immediate_expected_value + continuation_value
        if (
            behavior_guidance_profile is not None
            and game_state is not None
            and behavior_map_hypothesis is not None
            and acting_player_id is not None
        ):
            behavior_candidate_signal = self._posterior_candidate_signal(
                full_probability_matrix=full_probability_matrix,
                game_state=game_state,
                behavior_model=behavior_model,
                behavior_map_hypothesis=behavior_map_hypothesis,
                guesser_id=acting_player_id,
                target_player_id=player_id,
                target_slot_index=slot_index,
                guessed_card=card,
            )
            behavior_match_support = self._behavior_match_support(
                behavior_guidance_profile=behavior_guidance_profile,
                behavior_candidate_signal=behavior_candidate_signal,
            )
            behavior_match_multiplier = self._behavior_match_multiplier(
                behavior_match_support=behavior_match_support,
                stable_signal_ratio=float(behavior_guidance_profile.get("stable_signal_ratio", 0.0)),
            )
            behavior_match_bonus = max(0.0, immediate_expected_value) * (behavior_match_multiplier - 1.0)
            behavior_match_confidence_breakdown = self._behavior_match_candidate_confidence_breakdown(
                best_move={
                    "behavior_candidate_signal": behavior_candidate_signal,
                }
            )
            behavior_action_posterior_summary = self._behavior_action_posterior_summary(
                game_state=game_state,
                behavior_model=behavior_model,
                behavior_map_hypothesis=behavior_map_hypothesis,
                guesser_id=acting_player_id,
                target_player_id=player_id,
                target_slot_index=slot_index,
                guessed_card=card,
            )
        behavior_match_ranking_breakdown = self._behavior_match_ranking_breakdown(
            best_move={
                "behavior_match_bonus": behavior_match_bonus,
                "behavior_match_candidate_confidence": behavior_match_confidence_breakdown["candidate_confidence"],
                "behavior_match_component_support": behavior_match_confidence_breakdown["component_support"],
                "behavior_match_component_strength": behavior_match_confidence_breakdown["component_strength"],
                "behavior_match_component_penalty": behavior_match_confidence_breakdown["component_penalty"],
                "behavior_match_context_focus": behavior_match_confidence_breakdown["context_focus"],
            }
        )
        behavior_match_ranking_bonus = behavior_match_ranking_breakdown["ranking_bonus"]
        behavior_match_net_structure = behavior_match_ranking_breakdown["net_structure"]
        behavior_match_structure_adjustment = behavior_match_ranking_breakdown["structure_adjustment"]
        failure_recovery_signal = clamp(
            (
                post_hit_failure_recovery_bonus + post_hit_failed_switch_bonus
            ) / self.FAILURE_RECOVERY_REFERENCE,
            0.0,
            1.0,
        )
        opening_precision_breakdown = self._opening_precision_breakdown(
            game_state=game_state,
            player_id=player_id,
            guessed_card=card,
            probability=probability,
            slot_distribution=slot_distribution,
            information_gain=info_gain,
            behavior_confidence=behavior_match_confidence_breakdown["candidate_confidence"],
            behavior_action_posterior_support=behavior_action_posterior_summary["support"],
        )
        opening_precision_support = opening_precision_breakdown["support"]
        opening_precision_signal = opening_precision_breakdown["support"]
        opening_probability_signal = opening_precision_breakdown["probability_signal"]
        opening_margin_signal = opening_precision_breakdown["margin_signal"]
        opening_behavior_posterior_support = opening_precision_breakdown["posterior_signal"]
        opening_joker_penalty = opening_precision_breakdown["joker_penalty"]
        opening_numeric_competition_signal = opening_precision_breakdown[
            "numeric_competition_signal"
        ]
        initiative_recovery_signal = opening_precision_breakdown["initiative_recovery_signal"]
        tree_search_signal = clamp(
            max(
                post_hit_tree_search_signal,
                post_hit_mcts_signal,
                post_hit_expectimax_signal,
                post_hit_branch_search_signal,
            ),
            0.0,
            1.0,
        )
        strategy_objective_breakdown = self._strategy_objective_breakdown(
            expected_value=expected_value,
            win_probability=probability,
            continuation_likelihood=continuation_likelihood,
            attackability_after_hit=attackability_after_hit,
            information_gain=info_gain,
            expectimax_margin=post_hit_expectimax_margin,
            tree_search_signal=tree_search_signal,
            mcts_signal=post_hit_mcts_signal,
            support_ratio=max(
                post_hit_mcts_support_ratio,
                post_hit_tree_search_support_ratio,
                post_hit_expectimax_support_ratio,
                post_hit_top_k_support_ratio,
                post_hit_branch_search_support_ratio,
            ),
            behavior_match_bonus=behavior_match_ranking_bonus,
            post_hit_behavior_support_adjustment=post_hit_behavior_support_adjustment,
            failure_recovery_signal=failure_recovery_signal,
            opening_precision_signal=opening_precision_support,
            initiative_recovery_signal=initiative_recovery_signal,
            newly_drawn_exposure=float(self_exposure_profile["newly_drawn_exposure"]),
            self_public_exposure=float(self_exposure_profile["total_exposure"]),
            self_finish_fragility=float(self_exposure_profile["finish_fragility"]),
        )
        strategy_objective_core = strategy_objective_breakdown["total"]
        strategy_objective = strategy_objective_core
        ranking_score = (
            expected_value
            + behavior_match_ranking_bonus
            + (0.18 * opening_precision_support)
            + (0.12 * initiative_recovery_signal)
        )

        return {
            "target_player_id": player_id,
            "target_index": hidden_index_by_slot.get(slot_index, slot_index),
            "target_slot_index": slot_index,
            "guess_card": serialize_card(card),
            "win_probability": probability,
            "immediate_expected_value": immediate_expected_value,
            "expected_value": expected_value,
            "information_gain": info_gain,
            "continuation_value": continuation_value,
            "post_hit_continuation_value": post_hit_continuation_value,
            "next_best_immediate_value": next_best_immediate_ev,
            "continuation_likelihood": continuation_likelihood,
            "behavior_guidance_multiplier": behavior_guidance_multiplier,
            "behavior_guidance_support": behavior_guidance_support,
            "behavior_guidance_stable_ratio": behavior_guidance_stable_ratio,
            "behavior_guidance_signal_count": behavior_guidance_signal_count,
            "behavior_match_multiplier": behavior_match_multiplier,
            "behavior_match_bonus": behavior_match_bonus,
            "behavior_match_ranking_bonus": behavior_match_ranking_bonus,
            "behavior_match_net_structure": behavior_match_net_structure,
            "behavior_match_structure_adjustment": behavior_match_structure_adjustment,
            "behavior_match_support": behavior_match_support,
            "behavior_match_candidate_confidence": behavior_match_confidence_breakdown["candidate_confidence"],
            "behavior_match_component_support": behavior_match_confidence_breakdown["component_support"],
            "behavior_match_component_strength": behavior_match_confidence_breakdown["component_strength"],
            "behavior_match_component_penalty": behavior_match_confidence_breakdown["component_penalty"],
            "behavior_match_context_focus": behavior_match_confidence_breakdown["context_focus"],
            "behavior_candidate_signal": behavior_candidate_signal,
            "strategy_phase": opening_precision_breakdown["phase"],
            "opening_precision_support": opening_precision_support,
            "opening_precision_signal": opening_precision_signal,
            "opening_probability_signal": opening_probability_signal,
            "opening_margin_signal": opening_margin_signal,
            "opening_behavior_posterior_support": opening_behavior_posterior_support,
            "opening_joker_penalty": opening_joker_penalty,
            "opening_numeric_competition_signal": opening_numeric_competition_signal,
            "behavior_action_posterior_probability": behavior_action_posterior_summary["probability"],
            "behavior_action_posterior_weight": behavior_action_posterior_summary["weight"],
            "behavior_action_posterior_rank": behavior_action_posterior_summary["observed_rank"],
            "initiative_recovery_signal": initiative_recovery_signal,
            "strategy_objective_core": strategy_objective_core,
            "strategy_objective": strategy_objective,
            "strategy_objective_breakdown": strategy_objective_breakdown,
            "ranking_score": ranking_score,
            "attackability_after_hit": attackability_after_hit,
            "post_hit_continue_score": post_hit_continue_score,
            "post_hit_stop_score": post_hit_stop_score,
            "post_hit_continue_margin": post_hit_continue_margin,
            "post_hit_should_continue": post_hit_should_continue,
            "post_hit_best_gap": post_hit_best_gap,
            "post_hit_gap_adjustment": post_hit_gap_adjustment,
            "post_hit_guidance_multiplier": post_hit_guidance_multiplier,
            "post_hit_guidance_support": post_hit_guidance_support,
            "post_hit_guidance_stable_ratio": post_hit_guidance_stable_ratio,
            "post_hit_guidance_signal_count": post_hit_guidance_signal_count,
            "post_hit_guidance_debug": post_hit_guidance_debug,
            "post_hit_behavior_support_adjustment": post_hit_behavior_support_adjustment,
            "post_hit_behavior_support_gain": post_hit_behavior_support_gain,
            "post_hit_behavior_fragility_drag": post_hit_behavior_fragility_drag,
            "post_hit_behavior_support_signal": post_hit_behavior_support_signal,
            "post_hit_behavior_fragility_signal": post_hit_behavior_fragility_signal,
            "post_hit_behavior_support_strength": post_hit_behavior_support_strength,
            "post_hit_behavior_fragility_strength": post_hit_behavior_fragility_strength,
            "post_hit_failure_recovery_bonus": post_hit_failure_recovery_bonus,
            "post_hit_failed_switch_bonus": post_hit_failed_switch_bonus,
            "post_hit_failed_switch_signal": post_hit_failed_switch_signal,
            "post_hit_top_k_expected_continue_margin": post_hit_top_k_expected_continue_margin,
            "post_hit_top_k_continue_margin": post_hit_top_k_continue_margin,
            "post_hit_top_k_expected_support_ratio": post_hit_top_k_expected_support_ratio,
            "post_hit_top_k_support_ratio": post_hit_top_k_support_ratio,
            "post_hit_top_k_positive_count": post_hit_top_k_positive_count,
            "post_hit_branch_search_value": post_hit_branch_search_value,
            "post_hit_branch_search_margin": post_hit_branch_search_margin,
            "post_hit_branch_search_support_ratio": post_hit_branch_search_support_ratio,
            "post_hit_branch_search_signal": post_hit_branch_search_signal,
            "post_hit_tree_search_value": post_hit_tree_search_value,
            "post_hit_tree_search_margin": post_hit_tree_search_margin,
            "post_hit_tree_search_support_ratio": post_hit_tree_search_support_ratio,
            "post_hit_tree_search_signal": post_hit_tree_search_signal,
            "post_hit_mcts_value": post_hit_mcts_value,
            "post_hit_mcts_margin": post_hit_mcts_margin,
            "post_hit_mcts_support_ratio": post_hit_mcts_support_ratio,
            "post_hit_mcts_signal": post_hit_mcts_signal,
            "post_hit_mcts_node_count": post_hit_mcts_node_count,
            "post_hit_mcts_max_depth": post_hit_mcts_max_depth,
            "post_hit_mcts_search_mode": post_hit_mcts_search_mode,
            "post_hit_mcts_simulation_budget": post_hit_mcts_simulation_budget,
            "post_hit_expectimax_value": post_hit_expectimax_value,
            "post_hit_expectimax_margin": post_hit_expectimax_margin,
            "post_hit_expectimax_support_ratio": post_hit_expectimax_support_ratio,
            "post_hit_expectimax_signal": post_hit_expectimax_signal,
            "history_continue_rate": history_continue_rate,
            "hit_reward": hit_reward,
            "miss_penalty": miss_penalty,
            "structural_risk_factor": structural_risk_factor,
            "self_public_exposure": float(self_exposure_profile["total_exposure"]),
            "self_max_slot_exposure": float(self_exposure_profile["max_slot_exposure"]),
            "self_newly_drawn_exposure": float(self_exposure_profile["newly_drawn_exposure"]),
            "self_average_slot_exposure": float(self_exposure_profile["average_slot_exposure"]),
            "self_finish_fragility": float(self_exposure_profile["finish_fragility"]),
            "continuation_exposure_gate": continuation_exposure_gate,
            "failed_guess_slot_pressure": failed_guess_slot_pressure,
            "failed_guess_neighbor_pressure": failed_guess_neighbor_pressure,
            "failed_guess_player_pressure": failed_guess_player_pressure,
            "failure_collapse_bonus": failure_collapse_bonus,
            "failed_guess_switch_continuity_signal": failed_guess_switch_continuity_signal,
            "failed_guess_switch_bonus": failed_guess_switch_bonus,
            "target_attack_window_signal": target_attack_window_signal,
            "target_attack_window_bonus": target_attack_window_bonus,
            "target_attack_window_continuation_bonus": target_attack_window_continuation_bonus,
            "joint_collapse_signal": joint_collapse_signal,
            "joint_collapse_bonus": joint_collapse_bonus,
            "joint_collapse_continuation_bonus": joint_collapse_continuation_bonus,
            "global_propagation_signal": global_propagation_signal,
            "global_propagation_bonus": global_propagation_bonus,
            "global_propagation_continuation_bonus": global_propagation_continuation_bonus,
            "public_reveal_bridge_signal": public_reveal_bridge_signal,
            "public_reveal_bridge_bonus": public_reveal_bridge_bonus,
            "public_reveal_bridge_continuation_bonus": public_reveal_bridge_continuation_bonus,
            "target_chain_signal": target_chain_signal,
            "target_chain_bonus": target_chain_bonus,
            "target_chain_continuation_bonus": target_chain_continuation_bonus,
            "target_finish_chain_signal": target_finish_chain_signal,
            "target_finish_chain_bonus": target_finish_chain_bonus,
            "target_finish_chain_continuation_bonus": target_finish_chain_continuation_bonus,
            "_success_matrix": success_matrix,
            "_post_hit_game_state": post_hit_game_state_for_search,
            "_post_hit_guess_signals_by_player": post_hit_guess_signals_for_search,
            "_post_hit_behavior_guidance_profile": (
                post_hit_behavior_guidance_profile_for_search
            ),
            "_post_hit_behavior_map_hypothesis": (
                post_hit_behavior_map_hypothesis_for_search
            ),
            "_mcts_current_game_state": game_state,
            "_mcts_current_guess_signals_by_player": guess_signals_by_player,
            "_mcts_current_behavior_guidance_profile": behavior_guidance_profile,
            "_mcts_current_behavior_map_hypothesis": behavior_map_hypothesis,
            "_mcts_rollout_depth_remaining": max(0, rollout_depth - 1),
            "_mcts_my_hidden_count": my_hidden_count,
            "_mcts_behavior_model": behavior_model,
            "_mcts_acting_player_id": acting_player_id,
            "score_breakdown": {
                "hit_reward": hit_reward,
                "miss_penalty": miss_penalty,
                "structural_risk_factor": structural_risk_factor,
                "self_public_exposure": float(self_exposure_profile["total_exposure"]),
                "self_newly_drawn_exposure": float(self_exposure_profile["newly_drawn_exposure"]),
                "self_finish_fragility": float(self_exposure_profile["finish_fragility"]),
                "continuation_exposure_gate": continuation_exposure_gate,
                "information_gain_bonus": info_bonus,
                "failed_guess_slot_pressure": failed_guess_slot_pressure,
                "failed_guess_neighbor_pressure": failed_guess_neighbor_pressure,
                "failed_guess_player_pressure": failed_guess_player_pressure,
                "failure_collapse_bonus": failure_collapse_bonus,
                "failed_guess_switch_continuity_signal": failed_guess_switch_continuity_signal,
                "failed_guess_switch_bonus": failed_guess_switch_bonus,
                "target_attack_window_signal": target_attack_window_signal,
                "target_attack_window_bonus": target_attack_window_bonus,
                "target_attack_window_continuation_bonus": target_attack_window_continuation_bonus,
                "joint_collapse_signal": joint_collapse_signal,
                "joint_collapse_bonus": joint_collapse_bonus,
                "joint_collapse_continuation_bonus": joint_collapse_continuation_bonus,
                "global_propagation_signal": global_propagation_signal,
                "global_propagation_bonus": global_propagation_bonus,
                "global_propagation_continuation_bonus": global_propagation_continuation_bonus,
                "public_reveal_bridge_signal": public_reveal_bridge_signal,
                "public_reveal_bridge_bonus": public_reveal_bridge_bonus,
                "public_reveal_bridge_continuation_bonus": public_reveal_bridge_continuation_bonus,
                "target_chain_signal": target_chain_signal,
                "target_chain_bonus": target_chain_bonus,
                "target_chain_continuation_bonus": target_chain_continuation_bonus,
                "target_finish_chain_signal": target_finish_chain_signal,
                "target_finish_chain_bonus": target_finish_chain_bonus,
                "target_finish_chain_continuation_bonus": target_finish_chain_continuation_bonus,
                "immediate_expected_value": immediate_expected_value,
                "continuation_value": continuation_value,
                "post_hit_continuation_value": post_hit_continuation_value,
                "next_best_immediate_value": next_best_immediate_ev,
                "continuation_likelihood": continuation_likelihood,
                "behavior_guidance_multiplier": behavior_guidance_multiplier,
                "behavior_guidance_support": behavior_guidance_support,
                "behavior_guidance_stable_ratio": behavior_guidance_stable_ratio,
                "behavior_guidance_signal_count": behavior_guidance_signal_count,
                "behavior_match_multiplier": behavior_match_multiplier,
                "behavior_match_bonus": behavior_match_bonus,
                "behavior_match_ranking_bonus": behavior_match_ranking_bonus,
                "behavior_match_net_structure": behavior_match_net_structure,
                "behavior_match_structure_adjustment": behavior_match_structure_adjustment,
                "behavior_match_support": behavior_match_support,
                "behavior_match_candidate_confidence": behavior_match_confidence_breakdown["candidate_confidence"],
                "behavior_match_component_support": behavior_match_confidence_breakdown["component_support"],
                "behavior_match_component_strength": behavior_match_confidence_breakdown["component_strength"],
                "behavior_match_component_penalty": behavior_match_confidence_breakdown["component_penalty"],
                "behavior_match_context_focus": behavior_match_confidence_breakdown["context_focus"],
                "opening_precision_support": opening_precision_support,
                "opening_precision_signal": opening_precision_signal,
                "opening_probability_signal": opening_probability_signal,
                "opening_margin_signal": opening_margin_signal,
                "opening_behavior_posterior_support": opening_behavior_posterior_support,
                "opening_joker_penalty": opening_joker_penalty,
                "opening_numeric_competition_signal": opening_numeric_competition_signal,
                "behavior_action_posterior_probability": behavior_action_posterior_summary["probability"],
                "behavior_action_posterior_weight": behavior_action_posterior_summary["weight"],
                "behavior_action_posterior_rank": behavior_action_posterior_summary["observed_rank"],
                "initiative_recovery_signal": initiative_recovery_signal,
                "ranking_score": ranking_score,
                "strategy_objective_core": strategy_objective_core,
                "strategy_objective": strategy_objective,
                "strategy_objective_breakdown": dict(strategy_objective_breakdown),
                "attackability_after_hit": attackability_after_hit,
                "post_hit_continue_score": post_hit_continue_score,
                "post_hit_stop_score": post_hit_stop_score,
                "post_hit_continue_margin": post_hit_continue_margin,
                "post_hit_best_gap": post_hit_best_gap,
                "post_hit_gap_adjustment": post_hit_gap_adjustment,
                "post_hit_guidance_multiplier": post_hit_guidance_multiplier,
                "post_hit_guidance_support": post_hit_guidance_support,
                "post_hit_guidance_stable_ratio": post_hit_guidance_stable_ratio,
                "post_hit_guidance_signal_count": post_hit_guidance_signal_count,
                "post_hit_behavior_support_adjustment": post_hit_behavior_support_adjustment,
                "post_hit_behavior_support_gain": post_hit_behavior_support_gain,
                "post_hit_behavior_fragility_drag": post_hit_behavior_fragility_drag,
                "post_hit_behavior_support_signal": post_hit_behavior_support_signal,
                "post_hit_behavior_fragility_signal": post_hit_behavior_fragility_signal,
                "post_hit_behavior_support_strength": post_hit_behavior_support_strength,
                "post_hit_behavior_fragility_strength": post_hit_behavior_fragility_strength,
                "post_hit_failure_recovery_bonus": post_hit_failure_recovery_bonus,
                "post_hit_failed_switch_bonus": post_hit_failed_switch_bonus,
                "post_hit_failed_switch_signal": post_hit_failed_switch_signal,
                "post_hit_top_k_expected_continue_margin": post_hit_top_k_expected_continue_margin,
                "post_hit_top_k_continue_margin": post_hit_top_k_continue_margin,
                "post_hit_top_k_expected_support_ratio": post_hit_top_k_expected_support_ratio,
                "post_hit_top_k_support_ratio": post_hit_top_k_support_ratio,
                "post_hit_top_k_positive_count": post_hit_top_k_positive_count,
                "post_hit_branch_search_value": post_hit_branch_search_value,
                "post_hit_branch_search_margin": post_hit_branch_search_margin,
                "post_hit_branch_search_support_ratio": post_hit_branch_search_support_ratio,
                "post_hit_branch_search_signal": post_hit_branch_search_signal,
                "post_hit_mcts_value": post_hit_mcts_value,
                "post_hit_mcts_margin": post_hit_mcts_margin,
                "post_hit_mcts_support_ratio": post_hit_mcts_support_ratio,
                "post_hit_mcts_signal": post_hit_mcts_signal,
                "post_hit_mcts_node_count": post_hit_mcts_node_count,
                "post_hit_mcts_max_depth": post_hit_mcts_max_depth,
                "post_hit_expectimax_value": post_hit_expectimax_value,
                "post_hit_expectimax_margin": post_hit_expectimax_margin,
                "post_hit_expectimax_support_ratio": post_hit_expectimax_support_ratio,
                "post_hit_expectimax_signal": post_hit_expectimax_signal,
            },
            "recommendation_reason": self._build_reason(
                probability,
                info_gain,
                continuation_value,
                miss_penalty,
                continuation_likelihood,
            ),
            "target_scope": "player_slots",
        }

    def _mcts_move_prior(
        self,
        move: Dict[str, Any],
    ) -> float:
        return max(
            1e-9,
            (0.38 * float(move.get("win_probability", 0.0)))
            + (0.22 * float(move.get("continuation_likelihood", 0.0)))
            + (0.16 * float(move.get("post_hit_top_k_support_ratio", 0.0)))
            + (0.14 * float(move.get("target_attack_window_signal", 0.0)))
            + (0.10 * float(move.get("global_propagation_signal", 0.0))),
        )

    def _search_tree_node_count(
        self,
        node: SearchTreeNode,
    ) -> int:
        return 1 + sum(
            self._search_tree_node_count(child)
            for child in (node.children or ())
        )

    def _move_identity_key(
        self,
        move: Dict[str, Any],
    ) -> Tuple[Optional[str], int, Tuple[Any, ...]]:
        guess_card = move.get("guess_card")
        if isinstance(guess_card, (list, tuple)):
            normalized_guess_card = tuple(guess_card)
        else:
            normalized_guess_card = ()
        return (
            move.get("target_player_id"),
            int(move.get("target_slot_index", -1)),
            normalized_guess_card,
        )

    def _full_game_search_horizon(
        self,
        game_state: Optional[GameState],
        *,
        fallback_depth: int = 0,
    ) -> int:
        if game_state is None:
            return max(0, int(fallback_depth))
        hidden_count = sum(
            1
            for player_id in game_state.players
            for slot in game_state.resolved_ordered_slots(player_id)
            if not slot.is_revealed
        )
        return max(int(fallback_depth), int(hidden_count))

    def _adaptive_mcts_simulation_budget(
        self,
        tree_moves: Sequence[Dict[str, Any]],
    ) -> int:
        root_horizon = max(
            (
                self._full_game_search_horizon(
                    move.get(
                        "_mcts_current_game_state",
                        move.get("_post_hit_game_state"),
                    ),
                    fallback_depth=int(move.get("_mcts_rollout_depth_remaining", 0)),
                )
                for move in tree_moves
            ),
            default=0,
        )
        budget = (
            int(self.MCTS_SIMULATION_COUNT)
            + max(0, int(root_horizon) - 1) * int(self.MCTS_SIMULATION_COUNT_PER_HORIZON)
            + max(0, len(tree_moves) - 1) * int(self.MCTS_SIMULATION_COUNT_PER_ROOT_MOVE)
        )
        return max(
            int(self.MCTS_SIMULATION_COUNT),
            min(int(self.MCTS_SIMULATION_COUNT_CAP), int(budget)),
        )

    def _should_run_exhaustive_tree_search(
        self,
        tree_moves: Sequence[Dict[str, Any]],
    ) -> bool:
        if not tree_moves:
            return False
        root_horizon = max(
            (
                self._full_game_search_horizon(
                    move.get(
                        "_mcts_current_game_state",
                        move.get("_post_hit_game_state"),
                    ),
                    fallback_depth=int(move.get("_mcts_rollout_depth_remaining", 0)),
                )
                for move in tree_moves
            ),
            default=0,
        )
        if root_horizon > self.EXACT_TREE_SEARCH_HORIZON_THRESHOLD:
            return False
        branching_factor = max(
            len(tree_moves),
            int(self.MCTS_DEEP_CHILD_TOP_K) + int(self.MCTS_TOP_K),
        )
        estimated_node_count = 1.0 + float(len(tree_moves))
        frontier_width = float(max(1, len(tree_moves)))
        for _ in range(max(0, int(root_horizon))):
            frontier_width *= float(max(1, branching_factor))
            estimated_node_count += frontier_width
            if estimated_node_count > float(self.EXACT_TREE_SEARCH_NODE_BUDGET):
                return False
        return True

    def _build_hit_search_context(
        self,
        node: SearchTreeNode,
    ) -> Dict[str, Any]:
        move = node.move or {}
        success_matrix = node.success_matrix or move.get("_success_matrix")
        behavior_model = node.behavior_model
        game_state = node.game_state
        acting_player_id = node.acting_player_id
        guidance_profile = node.behavior_guidance_profile
        behavior_map_hypothesis = node.behavior_map_hypothesis
        guess_signals = node.guess_signals_by_player
        if (
            not success_matrix
            or behavior_model is None
            or game_state is None
            or acting_player_id is None
            or "target_player_id" not in move
            or "target_slot_index" not in move
            or "guess_card" not in move
        ):
            return {
                "success_matrix": success_matrix,
                "game_state": game_state,
                "guess_signals_by_player": guess_signals,
                "behavior_guidance_profile": guidance_profile,
                "behavior_map_hypothesis": behavior_map_hypothesis,
            }
        guess_card = move.get("guess_card")
        if not isinstance(guess_card, (list, tuple)) or len(guess_card) != 2:
            return {
                "success_matrix": success_matrix,
                "game_state": game_state,
                "guess_signals_by_player": guess_signals,
                "behavior_guidance_profile": guidance_profile,
                "behavior_map_hypothesis": behavior_map_hypothesis,
            }
        try:
            post_hit_context = self._build_post_hit_behavior_context(
                game_state=game_state,
                success_matrix=success_matrix,
                behavior_model=behavior_model,
                acting_player_id=acting_player_id,
                target_player_id=str(move.get("target_player_id")),
                target_slot_index=int(move.get("target_slot_index", -1)),
                guessed_card=(guess_card[0], guess_card[1]),
            )
            next_game_state = post_hit_context["game_state"]
            next_guess_signals = post_hit_context["guess_signals_by_player"]
            next_behavior_map_hypothesis = post_hit_context["behavior_map_hypothesis"]
            guidance_rebuild = self._rebuild_post_hit_behavior_guidance_profile(
                base_profile=guidance_profile,
                behavior_model=behavior_model,
                full_probability_matrix=success_matrix,
                game_state=next_game_state,
                guess_signals_by_player=next_guess_signals,
                behavior_map_hypothesis=next_behavior_map_hypothesis,
                acting_player_id=acting_player_id,
            )
            next_guidance_profile = guidance_rebuild["profile"]
        except ValueError:
            return {
                "success_matrix": success_matrix,
                "game_state": game_state,
                "guess_signals_by_player": guess_signals,
                "behavior_guidance_profile": guidance_profile,
                "behavior_map_hypothesis": behavior_map_hypothesis,
            }
        return {
            "success_matrix": success_matrix,
            "game_state": next_game_state,
            "guess_signals_by_player": next_guess_signals,
            "behavior_guidance_profile": next_guidance_profile,
            "behavior_map_hypothesis": next_behavior_map_hypothesis,
        }

    def _select_miss_reveal_slot(
        self,
        game_state: GameState,
        acting_player_id: str,
    ) -> Optional[CardSlot]:
        hidden_self_slots = [
            slot
            for slot in game_state.resolved_ordered_slots(acting_player_id)
            if slot.known_card() is not None and not slot.is_revealed
        ]
        newly_drawn_slot = next(
            (slot for slot in hidden_self_slots if getattr(slot, "is_newly_drawn", False)),
            None,
        )
        if newly_drawn_slot is not None:
            return newly_drawn_slot
        return (
            hidden_self_slots[-1]
            if hidden_self_slots
            else None
        )

    def _build_post_miss_behavior_context(
        self,
        *,
        game_state: GameState,
        acting_player_id: str,
        target_player_id: str,
        target_slot_index: int,
        guessed_card: Card,
    ) -> Optional[Dict[str, Any]]:
        revealed_slot = self._select_miss_reveal_slot(game_state, acting_player_id)
        players: Dict[str, PlayerState] = {}
        revealed_card: Optional[Card] = None
        revealed_slot_index: Optional[int] = None

        for player_id in game_state.players:
            slots: List[CardSlot] = []
            for slot in game_state.resolved_ordered_slots(player_id):
                reveal_current_slot = (
                    player_id == acting_player_id
                    and revealed_slot is not None
                    and slot.slot_index == revealed_slot.slot_index
                )
                slot_card = slot.known_card()
                if reveal_current_slot:
                    revealed_card = slot_card
                    revealed_slot_index = slot.slot_index
                slots.append(
                    CardSlot(
                        slot_index=slot.slot_index,
                        color=slot_card[0] if slot_card is not None else slot.color,
                        value=slot_card[1] if (reveal_current_slot and slot_card is not None) else slot.value,
                        is_revealed=slot.is_revealed or reveal_current_slot,
                        is_newly_drawn=False,
                    )
                )
            players[player_id] = PlayerState(player_id=player_id, slots=slots)

        next_self_player_id = target_player_id
        next_target_player_id = acting_player_id
        if next_self_player_id not in players or next_target_player_id not in players:
            return None

        post_miss_actions = list(game_state.actions)
        post_miss_actions.append(
            GuessAction(
                guesser_id=acting_player_id,
                target_player_id=target_player_id,
                target_slot_index=target_slot_index,
                guessed_color=guessed_card[0],
                guessed_value=guessed_card[1],
                result=False,
                continued_turn=False,
                revealed_player_id=acting_player_id,
                revealed_slot_index=revealed_slot_index,
                revealed_color=(
                    revealed_card[0] if revealed_card is not None else None
                ),
                revealed_value=(
                    revealed_card[1] if revealed_card is not None else None
                ),
            )
        )
        return {
            "game_state": GameState(
                self_player_id=next_self_player_id,
                target_player_id=next_target_player_id,
                players=players,
                actions=post_miss_actions,
            ),
            "revealed_card": revealed_card,
            "revealed_slot_index": revealed_slot_index,
        }

    def _miss_response_children(
        self,
        node: SearchTreeNode,
        *,
        miss_prior: float,
    ) -> List[SearchTreeNode]:
        move = node.move or {}
        game_state = node.game_state
        acting_player_id = node.acting_player_id
        guess_card = move.get("guess_card")
        target_player_id = move.get("target_player_id")
        target_slot_index = move.get("target_slot_index")
        if (
            game_state is None
            or acting_player_id is None
            or not isinstance(target_player_id, str)
            or not isinstance(target_slot_index, int)
            or not isinstance(guess_card, (list, tuple))
            or len(guess_card) != 2
        ):
            return []

        miss_context = self._build_post_miss_behavior_context(
            game_state=game_state,
            acting_player_id=acting_player_id,
            target_player_id=target_player_id,
            target_slot_index=target_slot_index,
            guessed_card=(guess_card[0], guess_card[1]),
        )
        if miss_context is None:
            return []

        response_game_state = miss_context["game_state"]
        try:
            response_controller = GameController(response_game_state)
        except ValueError:
            return []
        response_controller.decision_engine.DEEP_ROLLOUT_DEPTH = min(
            1,
            self.DEEP_ROLLOUT_DEPTH,
        )
        try:
            response_result = response_controller.run_turn(
                include_draw_color_summary=False,
            )
        except ValueError:
            return []
        response_moves = list(response_result.get("top_moves", ()))
        if (
            not response_moves
            and response_result.get("strategy_phase") == "post_draw_opening"
            and response_result.get("best_move") is not None
        ):
            response_moves = [response_result["best_move"]]
        if node.search_mode != "exhaustive":
            response_moves = response_moves[: self.MCTS_DEEP_CHILD_TOP_K]
        if not response_moves:
            return []

        response_stop_score = float(
            (response_result.get("decision_summary") or {}).get("stop_score", 0.0)
        )
        total_prior = sum(self._mcts_move_prior(move) for move in response_moves)
        if total_prior <= 0.0:
            total_prior = float(len(response_moves))
        response_children: List[SearchTreeNode] = []
        for child_index, response_move in enumerate(response_moves):
            response_move.setdefault("_mcts_stop_score", response_stop_score)
            response_children.append(
                SearchTreeNode(
                    label=f"{node.label}:miss:response:{child_index}",
                    prior=max(
                        1e-9,
                        miss_prior * self._mcts_move_prior(response_move) / total_prior,
                    ),
                    rollout_value=(
                        -node.perspective_sign
                        * float(
                            response_move.get(
                                "strategy_objective",
                                response_move.get(
                                    "ranking_score",
                                    response_move.get("expected_value", 0.0),
                                ),
                            )
                        )
                    ),
                    move=response_move,
                    success_matrix=response_move.get("_success_matrix"),
                    my_hidden_count=int(response_move.get("_mcts_my_hidden_count", 0)),
                    hidden_index_by_player=self._hidden_index_by_player_from_matrix(
                        response_move.get("_success_matrix", {})
                    )
                    if response_move.get("_success_matrix")
                    else None,
                    behavior_model=response_move.get("_mcts_behavior_model"),
                    guess_signals_by_player=response_move.get(
                        "_post_hit_guess_signals_by_player"
                    ),
                    acting_player_id=response_move.get("_mcts_acting_player_id"),
                    behavior_guidance_profile=response_move.get(
                        "_post_hit_behavior_guidance_profile"
                    ),
                    game_state=response_move.get(
                        "_post_hit_game_state",
                        response_game_state,
                    ),
                    behavior_map_hypothesis=response_move.get(
                        "_post_hit_behavior_map_hypothesis"
                    ),
                    rollout_depth_remaining=self._full_game_search_horizon(
                        response_game_state,
                        fallback_depth=max(
                            0,
                            int(node.rollout_depth_remaining) - 1,
                        ),
                    ),
                    perspective_sign=(-node.perspective_sign),
                    search_mode=node.search_mode,
                )
            )
        return response_children

    def _expand_post_hit_mcts_node(
        self,
        node: SearchTreeNode,
        *,
        stop_score: float,
    ) -> None:
        if node.is_terminal or node.children is not None:
            return
        move = node.move or {}
        local_stop_score = float(
            move.get(
                "_mcts_stop_score",
                move.get("post_hit_stop_score", stop_score),
            )
        )
        perspective_sign = float(node.perspective_sign or 1.0)
        future_value = max(
            0.0,
            float(
                move.get(
                    "post_hit_mcts_value",
                    move.get(
                        "post_hit_tree_search_value",
                        move.get(
                            "post_hit_expectimax_value",
                            move.get("post_hit_branch_search_value", 0.0),
                        ),
                    ),
                )
            ),
        )
        immediate_value = float(
            move.get(
                "strategy_objective",
                move.get("ranking_score", move.get("expected_value", 0.0)),
            )
        )
        hit_value = perspective_sign * (
            immediate_value + (
                self.MCTS_FUTURE_SCALE
                * float(move.get("win_probability", 0.0))
                * future_value
            )
        )
        miss_penalty = float(
            move.get(
                "miss_penalty",
                max(0.0, move.get("structural_risk_factor", 0.0)),
            )
        )
        miss_value = perspective_sign * (
            immediate_value
            - miss_penalty
            + (0.16 * float(move.get("information_gain", 0.0)))
            - (0.08 * local_stop_score)
        )
        hit_prior = clamp(float(move.get("win_probability", 0.0)), 0.05, 0.95)
        miss_prior = clamp(1.0 - hit_prior, 0.05, 0.95)
        deeper_success_matrix = node.success_matrix or move.get("_success_matrix")
        deeper_behavior_model = node.behavior_model
        hit_context = self._build_hit_search_context(node)
        deeper_guess_signals = hit_context.get(
            "guess_signals_by_player",
            node.guess_signals_by_player,
        )
        deeper_guidance_profile = hit_context.get(
            "behavior_guidance_profile",
            node.behavior_guidance_profile,
        )
        deeper_game_state = hit_context.get("game_state", node.game_state)
        deeper_behavior_map_hypothesis = hit_context.get(
            "behavior_map_hypothesis",
            node.behavior_map_hypothesis,
        )
        deeper_success_matrix = hit_context.get("success_matrix", deeper_success_matrix)
        deeper_hidden_index_by_player = self._hidden_index_by_player_from_matrix(
            deeper_success_matrix or {}
        ) if deeper_success_matrix else node.hidden_index_by_player
        deeper_my_hidden_count = node.my_hidden_count
        deeper_depth_remaining = self._full_game_search_horizon(
            deeper_game_state,
            fallback_depth=max(0, int(node.rollout_depth_remaining)),
        )

        future_children: List[SearchTreeNode] = []
        if (
            deeper_success_matrix
            and deeper_behavior_model is not None
            and deeper_hidden_index_by_player is not None
            and deeper_depth_remaining > 0
        ):
            future_moves, _ = self.evaluate_all_moves(
                full_probability_matrix=deeper_success_matrix,
                my_hidden_count=deeper_my_hidden_count,
                hidden_index_by_player=deeper_hidden_index_by_player,
                behavior_model=deeper_behavior_model,
                guess_signals_by_player=deeper_guess_signals or {},
                acting_player_id=node.acting_player_id,
                behavior_guidance_profile=deeper_guidance_profile,
                game_state=deeper_game_state,
                behavior_map_hypothesis=deeper_behavior_map_hypothesis,
                blocked_slots=set(),
                rollout_depth=0,
            )
            if node.search_mode != "exhaustive":
                future_moves = future_moves[: self.MCTS_DEEP_CHILD_TOP_K]
            if future_moves:
                total_child_prior = sum(
                    self._mcts_move_prior(child_move)
                    for child_move in future_moves
                )
                if total_child_prior <= 0.0:
                    total_child_prior = float(len(future_moves))
                for child_index, child_move in enumerate(future_moves):
                    child_prior = (
                        hit_prior
                        * self._mcts_move_prior(child_move)
                        / total_child_prior
                    )
                    future_children.append(
                        SearchTreeNode(
                            label=f"{node.label}:hit:{child_index}",
                            prior=max(1e-9, child_prior),
                            rollout_value=float(
                                child_move.get(
                                    "strategy_objective",
                                    child_move.get(
                                        "ranking_score",
                                        child_move.get("expected_value", 0.0),
                                    ),
                                )
                            ),
                            move=child_move,
                            success_matrix=child_move.get("_success_matrix"),
                            my_hidden_count=deeper_my_hidden_count,
                            hidden_index_by_player=self._hidden_index_by_player_from_matrix(
                                child_move.get("_success_matrix", {})
                            )
                            if child_move.get("_success_matrix")
                            else None,
                            behavior_model=deeper_behavior_model,
                            guess_signals_by_player=child_move.get(
                                "_post_hit_guess_signals_by_player",
                                deeper_guess_signals,
                            ),
                            acting_player_id=node.acting_player_id,
                            behavior_guidance_profile=child_move.get(
                                "_post_hit_behavior_guidance_profile",
                                deeper_guidance_profile,
                            ),
                            game_state=child_move.get(
                                "_post_hit_game_state",
                                deeper_game_state,
                            ),
                            behavior_map_hypothesis=child_move.get(
                                "_post_hit_behavior_map_hypothesis",
                                deeper_behavior_map_hypothesis,
                            ),
                            rollout_depth_remaining=self._full_game_search_horizon(
                                deeper_game_state,
                                fallback_depth=max(
                                    0,
                                    deeper_depth_remaining - 1,
                                ),
                            ),
                            perspective_sign=perspective_sign,
                            search_mode=node.search_mode,
                        )
                    )

        miss_children: List[SearchTreeNode] = []
        if deeper_depth_remaining > 0:
            miss_children = self._miss_response_children(
                node,
                miss_prior=miss_prior,
            )

        if future_children or miss_children:
            node.children = future_children + (
                miss_children
                if miss_children
                else [
                    SearchTreeNode(
                        label=f"{node.label}:miss",
                        prior=miss_prior,
                        rollout_value=miss_value,
                        is_terminal=True,
                        perspective_sign=perspective_sign,
                    ),
                ]
            )
            return

        node.children = [
            SearchTreeNode(
                label=f"{node.label}:hit",
                prior=hit_prior,
                rollout_value=hit_value,
                is_terminal=True,
                perspective_sign=perspective_sign,
            ),
            SearchTreeNode(
                label=f"{node.label}:miss",
                prior=miss_prior,
                rollout_value=miss_value,
                is_terminal=True,
                perspective_sign=perspective_sign,
            ),
        ]

    def _mcts_ucb_score(
        self,
        node: SearchTreeNode,
        *,
        total_visits: float,
    ) -> float:
        mean_value = (
            node.value_sum / node.visits
            if node.visits > 0.0
            else node.rollout_value
        )
        exploration_bonus = (
            self.MCTS_EXPLORATION_SCALE
            * node.prior
            * sqrt(log2(total_visits + 1.0) / (1.0 + node.visits))
        )
        return mean_value + exploration_bonus

    def _build_post_hit_search_root(
        self,
        *,
        tree_moves: Sequence[Dict[str, Any]],
        search_mode: str,
    ) -> SearchTreeNode:
        return SearchTreeNode(
            label="root",
            prior=1.0,
            rollout_value=0.0,
            children=[
                SearchTreeNode(
                    label=f"move_{index}",
                    prior=self._mcts_move_prior(move),
                    rollout_value=float(
                        move.get(
                            "strategy_objective",
                            move.get("ranking_score", move.get("expected_value", 0.0)),
                        )
                    ),
                    move=move,
                    success_matrix=move.get("_success_matrix"),
                    my_hidden_count=int(move.get("_mcts_my_hidden_count", 0)),
                    hidden_index_by_player=self._hidden_index_by_player_from_matrix(
                        move.get("_success_matrix", {})
                    )
                    if move.get("_success_matrix")
                    else None,
                    behavior_model=move.get("_mcts_behavior_model"),
                    guess_signals_by_player=move.get(
                        "_mcts_current_guess_signals_by_player",
                        move.get("_post_hit_guess_signals_by_player"),
                    ),
                    acting_player_id=move.get("_mcts_acting_player_id"),
                    behavior_guidance_profile=move.get(
                        "_mcts_current_behavior_guidance_profile",
                        move.get("_post_hit_behavior_guidance_profile"),
                    ),
                    game_state=move.get(
                        "_mcts_current_game_state",
                        move.get("_post_hit_game_state"),
                    ),
                    behavior_map_hypothesis=move.get(
                        "_mcts_current_behavior_map_hypothesis",
                        move.get("_post_hit_behavior_map_hypothesis"),
                    ),
                    rollout_depth_remaining=self._full_game_search_horizon(
                        move.get(
                            "_mcts_current_game_state",
                            move.get("_post_hit_game_state"),
                        ),
                        fallback_depth=int(
                            move.get("_mcts_rollout_depth_remaining", 0)
                        ),
                    ),
                    perspective_sign=1.0,
                    search_mode=search_mode,
                )
                for index, move in enumerate(tree_moves)
            ],
        )

    def _rollup_exhaustive_tree_search(
        self,
        node: SearchTreeNode,
        *,
        stop_score: float,
    ) -> Dict[str, float]:
        if not node.is_terminal and node.children is None:
            self._expand_post_hit_mcts_node(node, stop_score=stop_score)

        if node.is_terminal or not node.children:
            node_value = float(node.rollout_value)
            return {
                "value": node_value,
                "support_ratio": 1.0 if node_value > stop_score else 0.0,
                "peak_value": node_value,
                "max_depth": 0.0,
                "node_count": 1.0,
                "leaf_count": 1.0,
            }

        child_rollups = [
            self._rollup_exhaustive_tree_search(
                child,
                stop_score=stop_score,
            )
            for child in node.children
        ]
        total_prior = sum(max(1e-9, float(child.prior)) for child in node.children)
        if total_prior <= 0.0:
            total_prior = float(len(node.children))
        value = sum(
            max(1e-9, float(child.prior)) * rollup["value"]
            for child, rollup in zip(node.children, child_rollups)
        ) / total_prior
        support_ratio = sum(
            max(1e-9, float(child.prior)) * rollup["support_ratio"]
            for child, rollup in zip(node.children, child_rollups)
        ) / total_prior
        peak_value = max(rollup["peak_value"] for rollup in child_rollups)
        max_depth = 1.0 + max(rollup["max_depth"] for rollup in child_rollups)
        node_count = 1.0 + sum(rollup["node_count"] for rollup in child_rollups)
        leaf_count = sum(rollup["leaf_count"] for rollup in child_rollups)
        return {
            "value": value,
            "support_ratio": support_ratio,
            "peak_value": peak_value,
            "max_depth": max_depth,
            "node_count": node_count,
            "leaf_count": leaf_count,
        }

    def _run_post_hit_exhaustive_search(
        self,
        *,
        tree_moves: Sequence[Dict[str, Any]],
        stop_score: float,
    ) -> Dict[str, Any]:
        if not tree_moves:
            return {
                "value": 0.0,
                "margin": 0.0,
                "support_ratio": 0.0,
                "signal": 0.0,
                "peak_value": 0.0,
                "max_depth": 0.0,
                "node_count": 0.0,
                "root_children": [],
                "search_mode": "exhaustive",
                "simulation_budget": 0.0,
            }

        root = self._build_post_hit_search_root(
            tree_moves=tree_moves,
            search_mode="exhaustive",
        )
        root_children = root.children or []
        total_root_prior = sum(max(1e-9, float(child.prior)) for child in root_children)
        if total_root_prior <= 0.0:
            total_root_prior = float(len(root_children))

        root_child_summaries: List[Dict[str, Any]] = []
        aggregated_value = 0.0
        aggregated_support_ratio = 0.0
        peak_value = 0.0
        max_depth = 0.0
        total_node_count = 1.0
        for child in root_children:
            child_rollup = self._rollup_exhaustive_tree_search(
                child,
                stop_score=stop_score,
            )
            child_weight = max(1e-9, float(child.prior))
            aggregated_value += child_weight * child_rollup["value"]
            aggregated_support_ratio += child_weight * child_rollup["support_ratio"]
            peak_value = max(peak_value, child_rollup["peak_value"])
            max_depth = max(max_depth, 1.0 + child_rollup["max_depth"])
            total_node_count += child_rollup["node_count"]
            child_margin = max(0.0, child_rollup["value"] - stop_score)
            child_signal = clamp(
                (0.42 * child_rollup["support_ratio"])
                + (
                    0.34
                    * clamp(
                        child_margin
                        / self.DRAW_ROLLOUT_CONTINUE_MARGIN_REFERENCE,
                        0.0,
                        1.0,
                    )
                )
                + (
                    0.24
                    * clamp(
                        max(0.0, child_rollup["peak_value"] - stop_score)
                        / self.DRAW_ROLLOUT_CONTINUE_MARGIN_REFERENCE,
                        0.0,
                        1.0,
                    )
                ),
                0.0,
                1.0,
            )
            root_child_summaries.append(
                {
                    "move_key": self._move_identity_key(child.move or {}),
                    "value": child_rollup["value"],
                    "margin": child_margin,
                    "support_ratio": child_rollup["support_ratio"],
                    "signal": child_signal,
                    "visit_share": child_weight / total_root_prior,
                    "peak_value": child_rollup["peak_value"],
                    "visits": child_rollup["leaf_count"],
                }
            )

        value = aggregated_value / total_root_prior
        support_ratio = aggregated_support_ratio / total_root_prior
        margin = max(0.0, value - stop_score)
        best_child_margin = max(
            (summary["margin"] for summary in root_child_summaries),
            default=0.0,
        )
        signal = clamp(
            (0.34 * support_ratio)
            + (
                0.28
                * clamp(
                    margin / self.DRAW_ROLLOUT_CONTINUE_MARGIN_REFERENCE,
                    0.0,
                    1.0,
                )
            )
            + (
                0.22
                * clamp(
                    max(0.0, peak_value - stop_score)
                    / self.DRAW_ROLLOUT_CONTINUE_MARGIN_REFERENCE,
                    0.0,
                    1.0,
                )
            )
            + (
                0.16
                * clamp(
                    best_child_margin / self.DRAW_ROLLOUT_CONTINUE_MARGIN_REFERENCE,
                    0.0,
                    1.0,
                )
            ),
            0.0,
            1.0,
        )
        return {
            "value": value,
            "margin": margin,
            "support_ratio": support_ratio,
            "signal": signal,
            "peak_value": peak_value,
            "max_depth": max_depth,
            "node_count": total_node_count,
            "root_children": root_child_summaries,
            "search_mode": "exhaustive",
            "simulation_budget": float(total_node_count),
        }

    def _run_hybrid_full_game_search(
        self,
        *,
        tree_moves: Sequence[Dict[str, Any]],
        stop_score: float,
    ) -> Dict[str, Any]:
        return self._run_post_hit_exhaustive_search(
            tree_moves=tree_moves,
            stop_score=stop_score,
        )

    def _run_post_hit_mcts_search(
        self,
        *,
        tree_moves: Sequence[Dict[str, Any]],
        stop_score: float,
    ) -> Dict[str, Any]:
        if not tree_moves:
            return {
                "value": 0.0,
                "margin": 0.0,
                "support_ratio": 0.0,
                "signal": 0.0,
                "peak_value": 0.0,
                "max_depth": 0.0,
                "node_count": 0.0,
                "root_children": [],
                "search_mode": "mcts",
                "simulation_budget": 0.0,
            }

        root = self._build_post_hit_search_root(
            tree_moves=tree_moves,
            search_mode="mcts",
        )
        simulation_budget = self._adaptive_mcts_simulation_budget(tree_moves)

        positive_terminal_visits = 0.0
        total_terminal_visits = 0.0
        max_depth = 0.0
        for _ in range(simulation_budget):
            path = [root]
            current = root
            depth = 0.0
            while True:
                if current.is_terminal:
                    rollout_value = current.rollout_value
                    break
                if current.children is None:
                    self._expand_post_hit_mcts_node(current, stop_score=stop_score)
                if not current.children:
                    rollout_value = current.rollout_value
                    break
                total_visits = 1.0 + sum(child.visits for child in current.children)
                current = max(
                    current.children,
                    key=lambda child: (
                        self._mcts_ucb_score(child, total_visits=total_visits),
                        child.prior,
                        child.label,
                    ),
                )
                path.append(current)
                depth += 1.0
            max_depth = max(max_depth, depth)
            for node in path:
                node.visits += 1.0
                node.value_sum += rollout_value
                node.peak_value = max(node.peak_value, rollout_value)
                if rollout_value > stop_score:
                    node.positive_value_count += 1.0
            total_terminal_visits += 1.0
            if rollout_value > stop_score:
                positive_terminal_visits += 1.0

        root_children = root.children or []
        total_move_visits = sum(child.visits for child in root_children)
        if total_move_visits > 0.0:
            value = sum(child.value_sum for child in root_children) / total_move_visits
        else:
            value = sum(child.rollout_value for child in root_children) / float(
                len(root_children)
            )
        support_ratio = (
            positive_terminal_visits / total_terminal_visits
            if total_terminal_visits > 0.0
            else 0.0
        )
        peak_value = max(
            (child.peak_value or child.rollout_value) for child in root_children
        )
        best_child_margin = max(
            0.0,
            max(
                (
                    (child.value_sum / child.visits)
                    if child.visits > 0.0
                    else child.rollout_value
                )
                for child in root_children
            )
            - stop_score,
        )
        margin = max(0.0, value - stop_score)
        signal = clamp(
            (0.34 * support_ratio)
            + (
                0.28
                * clamp(
                    margin / self.DRAW_ROLLOUT_CONTINUE_MARGIN_REFERENCE,
                    0.0,
                    1.0,
                )
            )
            + (
                0.22
                * clamp(
                    max(0.0, peak_value - stop_score)
                    / self.DRAW_ROLLOUT_CONTINUE_MARGIN_REFERENCE,
                    0.0,
                    1.0,
                )
            )
            + (
                0.16
                * clamp(
                    best_child_margin / self.DRAW_ROLLOUT_CONTINUE_MARGIN_REFERENCE,
                    0.0,
                    1.0,
                )
            ),
            0.0,
            1.0,
        )
        root_child_summaries = []
        for child in root_children:
            child_value = (
                (child.value_sum / child.visits)
                if child.visits > 0.0
                else child.rollout_value
            )
            child_support_ratio = (
                child.positive_value_count / child.visits
                if child.visits > 0.0
                else 0.0
            )
            child_margin = max(0.0, child_value - stop_score)
            child_signal = clamp(
                (0.42 * child_support_ratio)
                + (
                    0.34
                    * clamp(
                        child_margin
                        / self.DRAW_ROLLOUT_CONTINUE_MARGIN_REFERENCE,
                        0.0,
                        1.0,
                    )
                )
                + (
                    0.24
                    * clamp(
                        max(0.0, (child.peak_value or child_value) - stop_score)
                        / self.DRAW_ROLLOUT_CONTINUE_MARGIN_REFERENCE,
                        0.0,
                        1.0,
                    )
                ),
                0.0,
                1.0,
            )
            root_child_summaries.append(
                {
                    "move_key": self._move_identity_key(child.move or {}),
                    "value": child_value,
                    "margin": child_margin,
                    "support_ratio": child_support_ratio,
                    "signal": child_signal,
                    "visit_share": (
                        child.visits / total_move_visits
                        if total_move_visits > 0.0
                        else 0.0
                    ),
                    "peak_value": float(child.peak_value or child_value),
                    "visits": float(child.visits),
                }
            )
        return {
            "value": value,
            "margin": margin,
            "support_ratio": support_ratio,
            "signal": signal,
            "peak_value": peak_value,
            "max_depth": max_depth,
            "node_count": float(self._search_tree_node_count(root)),
            "root_children": root_child_summaries,
            "search_mode": "mcts",
            "simulation_budget": float(simulation_budget),
        }

    def _evaluate_post_hit_rollout(
        self,
        *,
        success_matrix: FullProbabilityMatrix,
        my_hidden_count: int,
        behavior_model: BehavioralLikelihoodModel,
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        acting_player_id: Optional[str],
        behavior_guidance_profile: Optional[Dict[str, float]],
        game_state: Optional[GameState],
        target_player_id: str,
        target_slot_index: int,
        guessed_card: Card,
        rollout_depth: int,
    ) -> Dict[str, Any]:
        hidden_index_by_player = self._hidden_index_by_player_from_matrix(success_matrix)
        post_hit_game_state = None
        post_hit_guess_signals_by_player = guess_signals_by_player
        post_hit_behavior_map_hypothesis = None
        post_hit_behavior_guidance_profile = behavior_guidance_profile
        post_hit_behavior_guidance_debug = self._default_post_hit_guidance_rebuild_debug(
            base_profile=behavior_guidance_profile,
            acting_player_id=acting_player_id,
        )
        if game_state is not None and acting_player_id is not None:
            post_hit_context = self._build_post_hit_behavior_context(
                game_state=game_state,
                success_matrix=success_matrix,
                behavior_model=behavior_model,
                acting_player_id=acting_player_id,
                target_player_id=target_player_id,
                target_slot_index=target_slot_index,
                guessed_card=guessed_card,
            )
            post_hit_game_state = post_hit_context["game_state"]
            post_hit_guess_signals_by_player = post_hit_context["guess_signals_by_player"]
            post_hit_behavior_map_hypothesis = post_hit_context["behavior_map_hypothesis"]
            post_hit_guidance_rebuild = self._rebuild_post_hit_behavior_guidance_profile(
                base_profile=behavior_guidance_profile,
                behavior_model=behavior_model,
                full_probability_matrix=success_matrix,
                game_state=post_hit_game_state,
                guess_signals_by_player=post_hit_guess_signals_by_player,
                behavior_map_hypothesis=post_hit_behavior_map_hypothesis,
                acting_player_id=acting_player_id,
            )
            post_hit_behavior_guidance_profile = post_hit_guidance_rebuild["profile"]
            post_hit_behavior_guidance_debug = post_hit_guidance_rebuild["debug"]
        next_moves, next_risk_factor = self.evaluate_all_moves(
            full_probability_matrix=success_matrix,
            my_hidden_count=my_hidden_count,
            hidden_index_by_player=hidden_index_by_player,
            behavior_model=behavior_model,
            guess_signals_by_player=post_hit_guess_signals_by_player,
            acting_player_id=acting_player_id,
            behavior_guidance_profile=post_hit_behavior_guidance_profile,
            game_state=post_hit_game_state,
            behavior_map_hypothesis=post_hit_behavior_map_hypothesis,
            blocked_slots=set(),
            rollout_depth=rollout_depth,
        )
        next_best_move, next_summary = self.choose_best_move(
            next_moves,
            risk_factor=next_risk_factor,
            my_hidden_count=my_hidden_count,
        )
        top_k_moves = next_moves[: self.POST_HIT_TOP_K_COUNT]
        stop_score = next_summary.get("stop_score", next_summary.get("stop_threshold", 0.0))
        top_k_expected_continue_edges = [
            max(0.0, move["expected_value"] - stop_score)
            for move in top_k_moves
        ]
        top_k_continue_edges = [
            max(
                0.0,
                move.get(
                    "strategy_objective",
                    move.get("ranking_score", move["expected_value"]),
                )
                - stop_score,
            )
            for move in top_k_moves
        ]
        top_k_expected_continue_margin = (
            sum(top_k_expected_continue_edges) / len(top_k_expected_continue_edges)
            if top_k_expected_continue_edges
            else 0.0
        )
        top_k_continue_margin = (
            sum(top_k_continue_edges) / len(top_k_continue_edges)
            if top_k_continue_edges
            else 0.0
        )
        top_k_expected_positive_count = float(
            sum(1 for edge in top_k_expected_continue_edges if edge > 0.0)
        )
        top_k_positive_count = float(sum(1 for edge in top_k_continue_edges if edge > 0.0))
        top_k_expected_support_ratio = (
            top_k_expected_positive_count / len(top_k_expected_continue_edges)
            if top_k_expected_continue_edges
            else 0.0
        )
        top_k_support_ratio = (
            top_k_positive_count / len(top_k_continue_edges)
            if top_k_continue_edges
            else 0.0
        )
        branch_search_value = 0.0
        branch_search_margin = 0.0
        branch_search_support_ratio = 0.0
        branch_search_signal = 0.0
        tree_search_value = 0.0
        tree_search_margin = 0.0
        tree_search_support_ratio = 0.0
        tree_search_signal = 0.0
        mcts_value = 0.0
        mcts_margin = 0.0
        mcts_support_ratio = 0.0
        mcts_signal = 0.0
        mcts_node_count = 0.0
        mcts_max_depth = 0.0
        mcts_search_mode = "mcts"
        mcts_simulation_budget = 0.0
        expectimax_value = 0.0
        expectimax_margin = 0.0
        expectimax_support_ratio = 0.0
        expectimax_signal = 0.0
        branch_weight_sum = 0.0
        branch_positive_count = 0.0
        tree_weight_sum = 0.0
        tree_positive_mass = 0.0
        tree_peak_value = 0.0
        tree_candidates: List[float] = []
        mcts_visit_sum = 0.0
        mcts_positive_visits = 0.0
        mcts_peak_value = 0.0
        expectimax_positive_mass = 0.0
        expectimax_weight_sum = 0.0
        expectimax_peak_value = 0.0
        expectimax_candidates: List[float] = []
        for move in top_k_moves:
            immediate_branch_margin = max(
                0.0,
                move.get(
                    "strategy_objective",
                    move.get("ranking_score", move["expected_value"]),
                )
                - stop_score,
            )
            future_branch_margin = max(
                0.0,
                float(move.get("post_hit_top_k_continue_margin", 0.0)),
            )
            combined_branch_value = immediate_branch_margin + (
                self.BRANCH_SEARCH_FUTURE_MARGIN_BLEND * future_branch_margin
            )
            branch_weight = max(
                0.0,
                (0.45 * float(move.get("win_probability", 0.0)))
                + (0.25 * float(move.get("continuation_likelihood", 0.0)))
                + (0.20 * float(move.get("post_hit_top_k_support_ratio", 0.0)))
                + (0.10 * float(move.get("target_attack_window_signal", 0.0))),
            )
            branch_search_value += branch_weight * combined_branch_value
            branch_search_margin += branch_weight * immediate_branch_margin
            branch_weight_sum += branch_weight
            if combined_branch_value > 0.0:
                branch_positive_count += 1.0
        tree_moves = top_k_moves[: self.TREE_SEARCH_TOP_K]
        full_search_moves = list(next_moves)
        for move in tree_moves:
            immediate_tree_value = float(
                move.get(
                    "strategy_objective",
                    move.get("ranking_score", move["expected_value"]),
                )
            )
            future_tree_value = max(
                0.0,
                float(
                    move.get(
                        "post_hit_tree_search_value",
                        move.get(
                            "post_hit_expectimax_value",
                            move.get("post_hit_branch_search_value", 0.0),
                        ),
                    )
                ),
            )
            branch_tree_value = immediate_tree_value + (
                self.TREE_SEARCH_FUTURE_SCALE
                * float(move.get("win_probability", 0.0))
                * future_tree_value
            )
            branch_tree_weight = max(
                0.0,
                (0.34 * float(move.get("win_probability", 0.0)))
                + (0.22 * float(move.get("continuation_likelihood", 0.0)))
                + (0.16 * float(move.get("post_hit_tree_search_support_ratio", 0.0)))
                + (0.14 * float(move.get("post_hit_expectimax_support_ratio", 0.0)))
                + (0.14 * float(move.get("target_attack_window_signal", 0.0))),
            )
            tree_candidates.append(branch_tree_value)
            tree_peak_value = max(tree_peak_value, branch_tree_value)
            tree_search_value += branch_tree_weight * branch_tree_value
            tree_search_margin += branch_tree_weight * max(
                0.0,
                immediate_tree_value - stop_score,
            )
            tree_weight_sum += branch_tree_weight
            if branch_tree_value > stop_score:
                tree_positive_mass += branch_tree_weight
        for move in top_k_moves:
            future_expectimax_value = max(
                0.0,
                float(
                    move.get(
                        "post_hit_expectimax_value",
                        move.get("post_hit_branch_search_value", 0.0),
                    )
                ),
            )
            branch_expectimax_value = float(
                move.get(
                    "strategy_objective",
                    move.get("ranking_score", move["expected_value"]),
                )
            ) + (
                self.EXPECTIMAX_FUTURE_SCALE
                * float(move.get("win_probability", 0.0))
                * future_expectimax_value
            )
            branch_expectimax_weight = max(
                0.0,
                (0.40 * float(move.get("win_probability", 0.0)))
                + (0.25 * float(move.get("continuation_likelihood", 0.0)))
                + (0.20 * float(move.get("post_hit_top_k_support_ratio", 0.0)))
                + (0.15 * float(move.get("target_attack_window_signal", 0.0))),
            )
            expectimax_candidates.append(branch_expectimax_value)
            expectimax_peak_value = max(expectimax_peak_value, branch_expectimax_value)
            expectimax_value += branch_expectimax_weight * branch_expectimax_value
            expectimax_weight_sum += branch_expectimax_weight
            if branch_expectimax_value > stop_score:
                expectimax_positive_mass += branch_expectimax_weight
        if branch_weight_sum > 0.0:
            branch_search_value /= branch_weight_sum
            branch_search_margin /= branch_weight_sum
        if top_k_moves:
            branch_search_support_ratio = branch_positive_count / len(top_k_moves)
        branch_search_signal = clamp(
            (0.55 * branch_search_support_ratio)
            + (
                0.45
                * clamp(
                    branch_search_margin
                    / self.DRAW_ROLLOUT_CONTINUE_MARGIN_REFERENCE,
                    0.0,
                    1.0,
                )
            ),
            0.0,
            1.0,
        )
        if tree_candidates:
            if tree_weight_sum > 0.0:
                tree_search_value /= tree_weight_sum
                tree_search_margin /= tree_weight_sum
                tree_search_support_ratio = tree_positive_mass / tree_weight_sum
            else:
                tree_search_value = sum(tree_candidates) / len(tree_candidates)
                tree_search_margin = max(0.0, tree_search_value - stop_score)
                tree_search_support_ratio = float(
                    sum(1 for value in tree_candidates if value > stop_score)
                ) / len(tree_candidates)
            tree_search_signal = clamp(
                (0.38 * tree_search_support_ratio)
                + (
                    0.32
                    * clamp(
                        tree_search_margin
                        / self.DRAW_ROLLOUT_CONTINUE_MARGIN_REFERENCE,
                        0.0,
                        1.0,
                    )
                )
                + (
                    0.30
                    * clamp(
                        max(0.0, tree_peak_value - stop_score)
                        / self.DRAW_ROLLOUT_CONTINUE_MARGIN_REFERENCE,
                        0.0,
                        1.0,
                    )
                ),
                0.0,
                1.0,
            )
        if full_search_moves:
            mcts_search = self._run_hybrid_full_game_search(
                tree_moves=full_search_moves,
                stop_score=stop_score,
            )
            mcts_value = float(mcts_search["value"])
            mcts_margin = float(mcts_search["margin"])
            mcts_support_ratio = float(mcts_search["support_ratio"])
            mcts_signal = float(mcts_search["signal"])
            mcts_node_count = float(mcts_search["node_count"])
            mcts_max_depth = float(mcts_search["max_depth"])
            mcts_search_mode = str(mcts_search.get("search_mode", "mcts"))
            mcts_simulation_budget = float(
                mcts_search.get("simulation_budget", 0.0)
            )
        if expectimax_candidates:
            if expectimax_weight_sum > 0.0:
                expectimax_value /= expectimax_weight_sum
                expectimax_support_ratio = (
                    expectimax_positive_mass / expectimax_weight_sum
                )
            else:
                expectimax_value = sum(expectimax_candidates) / len(expectimax_candidates)
                expectimax_support_ratio = float(
                    sum(1 for value in expectimax_candidates if value > stop_score)
                ) / len(expectimax_candidates)
            expectimax_margin = max(0.0, expectimax_value - stop_score)
            expectimax_signal = clamp(
                (0.40 * expectimax_support_ratio)
                + (
                    0.35
                    * clamp(
                        expectimax_margin
                        / self.DRAW_ROLLOUT_CONTINUE_MARGIN_REFERENCE,
                        0.0,
                        1.0,
                    )
                ),
                0.0,
                1.0,
            )
            expectimax_signal = clamp(
                expectimax_signal
                + (
                    0.25
                    * clamp(
                        max(0.0, expectimax_peak_value - stop_score)
                        / self.DRAW_ROLLOUT_CONTINUE_MARGIN_REFERENCE,
                        0.0,
                        1.0,
                    )
                ),
                0.0,
                1.0,
            )
        return {
            "should_continue": next_best_move is not None,
            "continue_score": next_summary.get("continue_score", 0.0),
            "stop_score": stop_score,
            "continue_margin": next_summary.get("continue_margin", 0.0),
            "best_gap": next_summary.get("best_gap", 0.0),
            "failure_recovery_bonus": next_summary.get("best_failure_collapse_bonus", 0.0),
            "failed_switch_bonus": next_summary.get("best_failed_guess_switch_bonus", 0.0),
            "failed_switch_signal": next_summary.get(
                "best_failed_guess_switch_continuity_signal",
                0.0,
            ),
            "behavior_guidance_multiplier": float(
                (post_hit_behavior_guidance_profile or {}).get(
                    "guidance_multiplier",
                    self.DEFAULT_BEHAVIOR_GUIDANCE_MULTIPLIER,
                )
            ),
            "behavior_guidance_support": float(
                (post_hit_behavior_guidance_profile or {}).get(
                    "average_posterior_support",
                    0.0,
                )
            ),
            "behavior_guidance_stable_ratio": float(
                (post_hit_behavior_guidance_profile or {}).get(
                    "stable_signal_ratio",
                    0.0,
                )
            ),
            "behavior_guidance_signal_count": float(
                (post_hit_behavior_guidance_profile or {}).get(
                    "signal_count",
                    0.0,
                )
            ),
            "top_k_expected_continue_margin": top_k_expected_continue_margin,
            "top_k_continue_margin": top_k_continue_margin,
            "top_k_expected_support_ratio": top_k_expected_support_ratio,
            "top_k_support_ratio": top_k_support_ratio,
            "top_k_positive_count": top_k_positive_count,
            "branch_search_value": branch_search_value,
            "branch_search_margin": branch_search_margin,
            "branch_search_support_ratio": branch_search_support_ratio,
            "branch_search_signal": branch_search_signal,
            "tree_search_value": tree_search_value,
            "tree_search_margin": tree_search_margin,
            "tree_search_support_ratio": tree_search_support_ratio,
            "tree_search_signal": tree_search_signal,
            "mcts_value": mcts_value,
            "mcts_margin": mcts_margin,
            "mcts_support_ratio": mcts_support_ratio,
            "mcts_signal": mcts_signal,
            "mcts_node_count": mcts_node_count,
            "mcts_max_depth": mcts_max_depth,
            "mcts_search_mode": mcts_search_mode,
            "mcts_simulation_budget": mcts_simulation_budget,
            "expectimax_value": expectimax_value,
            "expectimax_margin": expectimax_margin,
            "expectimax_support_ratio": expectimax_support_ratio,
            "expectimax_signal": expectimax_signal,
            "guidance_debug": post_hit_behavior_guidance_debug,
            "post_hit_game_state": post_hit_game_state,
            "post_hit_guess_signals_by_player": post_hit_guess_signals_by_player,
            "post_hit_behavior_guidance_profile": post_hit_behavior_guidance_profile,
            "post_hit_behavior_map_hypothesis": post_hit_behavior_map_hypothesis,
        }

    def _build_post_hit_behavior_context(
        self,
        *,
        game_state: GameState,
        success_matrix: FullProbabilityMatrix,
        behavior_model: BehavioralLikelihoodModel,
        acting_player_id: str,
        target_player_id: str,
        target_slot_index: int,
        guessed_card: Card,
    ) -> Dict[str, Any]:
        players: Dict[str, PlayerState] = {}
        for player_id in game_state.players:
            slots: List[CardSlot] = []
            for slot in game_state.resolved_ordered_slots(player_id):
                slot_card = slot.known_card()
                if player_id == target_player_id and slot.slot_index == target_slot_index:
                    slot_card = guessed_card
                slots.append(
                    CardSlot(
                        slot_index=slot.slot_index,
                        color=slot_card[0] if slot_card is not None else slot.color,
                        value=slot_card[1] if slot_card is not None else slot.value,
                        is_revealed=(slot_card is not None) or slot.is_revealed,
                        is_newly_drawn=slot.is_newly_drawn,
                    )
                )
            players[player_id] = PlayerState(player_id=player_id, slots=slots)

        post_hit_actions = list(game_state.actions)
        post_hit_actions.append(
            GuessAction(
                guesser_id=acting_player_id,
                target_player_id=target_player_id,
                target_slot_index=target_slot_index,
                guessed_color=guessed_card[0],
                guessed_value=guessed_card[1],
                result=True,
                continued_turn=True,
                revealed_player_id=target_player_id,
                revealed_slot_index=target_slot_index,
                revealed_color=guessed_card[0],
                revealed_value=guessed_card[1],
            )
        )
        post_hit_game_state = GameState(
            self_player_id=game_state.self_player_id,
            target_player_id=game_state.target_player_id,
            players=players,
            actions=post_hit_actions,
        )
        return {
            "game_state": post_hit_game_state,
            "guess_signals_by_player": behavior_model.build_guess_signals(post_hit_game_state),
            "behavior_map_hypothesis": self._map_hypothesis_from_matrix(success_matrix),
        }

    def _rebuild_post_hit_behavior_guidance_profile(
        self,
        *,
        base_profile: Optional[Dict[str, float]],
        behavior_model: BehavioralLikelihoodModel,
        full_probability_matrix: FullProbabilityMatrix,
        game_state: GameState,
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        behavior_map_hypothesis: Dict[str, Dict[int, Card]],
        acting_player_id: str,
    ) -> Dict[str, Any]:
        fallback_profile = {
            "signal_count": float((base_profile or {}).get("signal_count", 0.0)),
            "average_posterior_support": float((base_profile or {}).get("average_posterior_support", 0.0)),
            "average_weighted_strength": float((base_profile or {}).get("average_weighted_strength", 0.0)),
            "stable_signal_ratio": float((base_profile or {}).get("stable_signal_ratio", 0.0)),
            "guidance_multiplier": float(
                (base_profile or {}).get(
                    "guidance_multiplier",
                    self.DEFAULT_BEHAVIOR_GUIDANCE_MULTIPLIER,
                )
            ),
            "source_support_progressive": float((base_profile or {}).get("source_support_progressive", 0.0)),
            "source_support_same_color_anchor": float((base_profile or {}).get("source_support_same_color_anchor", 0.0)),
            "source_support_local_boundary": float((base_profile or {}).get("source_support_local_boundary", 0.0)),
        }
        default_debug = self._default_post_hit_guidance_rebuild_debug(
            base_profile=fallback_profile,
            acting_player_id=acting_player_id,
        )
        acting_signals = guess_signals_by_player.get(acting_player_id, ())
        if not acting_signals:
            return {
                "profile": fallback_profile,
                "debug": default_debug,
            }

        guidance_matrix_context = self._augment_behavior_debug_matrix(
            full_probability_matrix=full_probability_matrix,
            game_state=game_state,
            guess_signals_by_player=guess_signals_by_player,
        )
        guidance_matrix = guidance_matrix_context["matrix"]
        guidance_map_hypothesis = self._map_hypothesis_from_matrix(guidance_matrix)
        guidance_controller = GameController(game_state)
        behavior_debug = guidance_controller._build_behavior_debug(
            full_probability_matrix=guidance_matrix,
            guess_signals_by_player=guess_signals_by_player,
            map_hypothesis=guidance_map_hypothesis or behavior_map_hypothesis,
        )
        rebuilt_profile = guidance_controller._build_behavior_guidance_profile(
            behavior_debug_signals=behavior_debug["signals"],
            acting_player_id=acting_player_id,
        )
        if float(rebuilt_profile.get("signal_count", 0.0)) <= 0.0:
            return {
                "profile": fallback_profile,
                "debug": default_debug,
            }
        blended_profile = self._blend_behavior_guidance_profiles(
            base_profile=fallback_profile,
            updated_profile=rebuilt_profile,
        )
        signal_summaries = self._summarize_guidance_rebuild_signals(
            behavior_debug_signals=behavior_debug["signals"],
            acting_player_id=acting_player_id,
        )
        debug_payload = {
            "rebuild_applied": True,
            "acting_player_id": acting_player_id,
            "acting_signal_count": float(len(acting_signals)),
            "rebuilt_signal_count": float(rebuilt_profile.get("signal_count", 0.0)),
            "augmented_known_slot_count": float(len(guidance_matrix_context["augmented_known_slots"])),
            "augmented_known_slots": guidance_matrix_context["augmented_known_slots"],
            "signal_summaries": signal_summaries,
            "base_profile": dict(fallback_profile),
            "rebuilt_profile": dict(rebuilt_profile),
            "blended_profile": dict(blended_profile),
            "rebuilt_delta_from_base": self._behavior_guidance_profile_delta(
                base_profile=fallback_profile,
                updated_profile=rebuilt_profile,
            ),
            "blended_delta_from_base": self._behavior_guidance_profile_delta(
                base_profile=fallback_profile,
                updated_profile=blended_profile,
            ),
            "rebuilt_dominant_source_shift": self._dominant_guidance_source_shift(
                self._behavior_guidance_profile_delta(
                    base_profile=fallback_profile,
                    updated_profile=rebuilt_profile,
                )
            ),
            "rebuilt_dominant_source_shift_strength": self._dominant_guidance_source_shift_strength(
                self._behavior_guidance_profile_delta(
                    base_profile=fallback_profile,
                    updated_profile=rebuilt_profile,
                )
            ),
            "blended_dominant_source_shift": self._dominant_guidance_source_shift(
                self._behavior_guidance_profile_delta(
                    base_profile=fallback_profile,
                    updated_profile=blended_profile,
                )
            ),
            "blended_dominant_source_shift_strength": self._dominant_guidance_source_shift_strength(
                self._behavior_guidance_profile_delta(
                    base_profile=fallback_profile,
                    updated_profile=blended_profile,
                )
            ),
        }
        return {
            "profile": blended_profile,
            "debug": debug_payload,
        }

    def _augment_behavior_debug_matrix(
        self,
        *,
        full_probability_matrix: FullProbabilityMatrix,
        game_state: GameState,
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
    ) -> Dict[str, Any]:
        augmented_matrix: FullProbabilityMatrix = {
            player_id: {
                slot_index: dict(slot_distribution)
                for slot_index, slot_distribution in probability_matrix.items()
            }
            for player_id, probability_matrix in full_probability_matrix.items()
        }
        augmented_known_slots: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for signals in guess_signals_by_player.values():
            for signal in signals:
                try:
                    slot = game_state.get_slot(signal.target_player_id, signal.target_slot_index)
                except ValueError:
                    continue
                known_card = slot.known_card()
                if known_card is None:
                    continue
                augmented_matrix.setdefault(signal.target_player_id, {})[signal.target_slot_index] = {
                    known_card: 1.0,
                }
                augmented_known_slots[(signal.target_player_id, signal.target_slot_index)] = {
                    "player_id": signal.target_player_id,
                    "slot_index": signal.target_slot_index,
                    "card": serialize_card(known_card),
                }
        return {
            "matrix": augmented_matrix,
            "augmented_known_slots": [
                augmented_known_slots[key]
                for key in sorted(augmented_known_slots)
            ],
        }

    def _default_post_hit_guidance_rebuild_debug(
        self,
        *,
        base_profile: Optional[Dict[str, float]],
        acting_player_id: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "rebuild_applied": False,
            "acting_player_id": acting_player_id,
            "acting_signal_count": 0.0,
            "rebuilt_signal_count": 0.0,
            "augmented_known_slot_count": 0.0,
            "augmented_known_slots": [],
            "signal_summaries": [],
            "base_profile": dict(base_profile or {}),
            "rebuilt_profile": {
                "signal_count": 0.0,
                "average_posterior_support": 0.0,
                "average_weighted_strength": 0.0,
                "stable_signal_ratio": 0.0,
                "guidance_multiplier": self.DEFAULT_BEHAVIOR_GUIDANCE_MULTIPLIER,
                "source_support_progressive": 0.0,
                "source_support_same_color_anchor": 0.0,
                "source_support_local_boundary": 0.0,
            },
            "blended_profile": dict(base_profile or {}),
            "rebuilt_delta_from_base": self._behavior_guidance_profile_delta(
                base_profile=base_profile or {},
                updated_profile=None,
            ),
            "blended_delta_from_base": self._behavior_guidance_profile_delta(
                base_profile=base_profile or {},
                updated_profile=base_profile or {},
            ),
            "rebuilt_dominant_source_shift": "neutral",
            "blended_dominant_source_shift": "neutral",
            "rebuilt_dominant_source_shift_strength": 0.0,
            "blended_dominant_source_shift_strength": 0.0,
        }

    def _summarize_guidance_rebuild_signals(
        self,
        *,
        behavior_debug_signals: Sequence[Dict[str, Any]],
        acting_player_id: str,
    ) -> List[Dict[str, Any]]:
        signal_summaries: List[Dict[str, Any]] = []
        for signal in behavior_debug_signals:
            if signal.get("guesser_id") != acting_player_id:
                continue
            dominant_signal = signal.get("value_selection", {}).get("dominant_signal", {})
            signal_summaries.append(
                {
                    "target_player_id": signal.get("target_player_id"),
                    "target_slot_index": signal.get("target_slot_index"),
                    "result": signal.get("result"),
                    "continued_turn": signal.get("continued_turn"),
                    "dominant_source": dominant_signal.get("source"),
                    "posterior_support": float(dominant_signal.get("posterior_support", 0.0)),
                    "weighted_strength": float(dominant_signal.get("weighted_strength", 0.0)),
                    "covered_probability": float(
                        signal.get("value_selection", {}).get("covered_probability", 0.0)
                    ),
                }
            )
        return signal_summaries

    def _behavior_guidance_profile_delta(
        self,
        *,
        base_profile: Dict[str, float],
        updated_profile: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        keys = (
            "signal_count",
            "average_posterior_support",
            "average_weighted_strength",
            "stable_signal_ratio",
            "guidance_multiplier",
            "source_support_progressive",
            "source_support_same_color_anchor",
            "source_support_local_boundary",
        )
        reference_profile = updated_profile or {}
        return {
            key: float(reference_profile.get(key, 0.0)) - float(base_profile.get(key, 0.0))
            for key in keys
        }

    def _dominant_guidance_source_shift(
        self,
        delta_profile: Dict[str, float],
    ) -> str:
        source_key_mapping = {
            "progressive": float(delta_profile.get("source_support_progressive", 0.0)),
            "same_color_anchor": float(delta_profile.get("source_support_same_color_anchor", 0.0)),
            "local_boundary": float(delta_profile.get("source_support_local_boundary", 0.0)),
        }
        dominant_source, dominant_delta = max(
            source_key_mapping.items(),
            key=lambda item: (item[1], item[0]),
        )
        if dominant_delta <= 0.0:
            return "neutral"
        return dominant_source

    def _dominant_guidance_source_shift_strength(
        self,
        delta_profile: Dict[str, float],
    ) -> float:
        return max(
            0.0,
            float(delta_profile.get("source_support_progressive", 0.0)),
            float(delta_profile.get("source_support_same_color_anchor", 0.0)),
            float(delta_profile.get("source_support_local_boundary", 0.0)),
        )

    def _build_post_hit_guidance_rebuild_reason(
        self,
        guidance_debug: Dict[str, Any],
    ) -> str:
        if not guidance_debug.get("rebuild_applied", False):
            return "post-hit rebuild unavailable"
        source_shift = str(guidance_debug.get("blended_dominant_source_shift", "neutral"))
        shift_strength = float(guidance_debug.get("blended_dominant_source_shift_strength", 0.0))
        multiplier_delta = float(
            guidance_debug.get("blended_delta_from_base", {}).get("guidance_multiplier", 0.0)
        )
        rebuilt_signal_count = float(guidance_debug.get("rebuilt_signal_count", 0.0))
        return (
            f"post-hit rebuild favors {source_shift}, "
            f"strength {shift_strength:.3f}, "
            f"signals {rebuilt_signal_count:.0f}, "
            f"multiplier delta {multiplier_delta:.3f}"
        )

    def _blend_behavior_guidance_profiles(
        self,
        *,
        base_profile: Dict[str, float],
        updated_profile: Dict[str, float],
    ) -> Dict[str, float]:
        base_count = max(0.0, float(base_profile.get("signal_count", 0.0)))
        updated_count = max(0.0, float(updated_profile.get("signal_count", 0.0)))
        if base_count <= 0.0:
            return dict(updated_profile)
        if updated_count <= 0.0:
            return dict(base_profile)

        total_count = base_count + updated_count
        blended_profile = {
            "signal_count": total_count,
            "average_posterior_support": (
                (float(base_profile.get("average_posterior_support", 0.0)) * base_count)
                + (float(updated_profile.get("average_posterior_support", 0.0)) * updated_count)
            ) / total_count,
            "average_weighted_strength": (
                (float(base_profile.get("average_weighted_strength", 0.0)) * base_count)
                + (float(updated_profile.get("average_weighted_strength", 0.0)) * updated_count)
            ) / total_count,
            "stable_signal_ratio": (
                (float(base_profile.get("stable_signal_ratio", 0.0)) * base_count)
                + (float(updated_profile.get("stable_signal_ratio", 0.0)) * updated_count)
            ) / total_count,
            "source_support_progressive": (
                (float(base_profile.get("source_support_progressive", 0.0)) * base_count)
                + (float(updated_profile.get("source_support_progressive", 0.0)) * updated_count)
            ) / total_count,
            "source_support_same_color_anchor": (
                (float(base_profile.get("source_support_same_color_anchor", 0.0)) * base_count)
                + (float(updated_profile.get("source_support_same_color_anchor", 0.0)) * updated_count)
            ) / total_count,
            "source_support_local_boundary": (
                (float(base_profile.get("source_support_local_boundary", 0.0)) * base_count)
                + (float(updated_profile.get("source_support_local_boundary", 0.0)) * updated_count)
            ) / total_count,
        }
        blended_profile["guidance_multiplier"] = clamp(
            0.95
            + (0.08 * blended_profile["average_posterior_support"])
            + (0.07 * blended_profile["stable_signal_ratio"])
            + (0.14 * blended_profile["average_weighted_strength"]),
            0.93,
            1.12,
        )
        return blended_profile

    def _map_hypothesis_from_matrix(
        self,
        full_probability_matrix: FullProbabilityMatrix,
    ) -> Dict[str, Dict[int, Card]]:
        hypothesis_by_player: Dict[str, Dict[int, Card]] = {}
        for player_id, probability_matrix in full_probability_matrix.items():
            player_hypothesis: Dict[int, Card] = {}
            for slot_index, slot_distribution in probability_matrix.items():
                if not slot_distribution:
                    continue
                best_card = max(
                    slot_distribution.items(),
                    key=lambda item: (item[1], -card_sort_key(item[0])[0], -card_sort_key(item[0])[1]),
                )[0]
                player_hypothesis[slot_index] = best_card
            if player_hypothesis:
                hypothesis_by_player[player_id] = player_hypothesis
        return hypothesis_by_player

    def _behavior_match_support(
        self,
        *,
        behavior_guidance_profile: Dict[str, float],
        behavior_candidate_signal: Dict[str, Any],
    ) -> float:
        source_weight_mapping = {
            "progressive": max(
                0.0,
                float(behavior_candidate_signal["progressive"].get("weight", 1.0)) - 1.0,
            ),
            "same_color_anchor": max(
                0.0,
                float(behavior_candidate_signal["anchor"].get("weight", 1.0)) - 1.0,
            ),
            "local_boundary": max(
                0.0,
                float(behavior_candidate_signal["boundary"].get("weight", 1.0)) - 1.0,
            ),
        }
        return sum(
            float(behavior_guidance_profile.get(f"source_support_{source}", 0.0)) * weight
            for source, weight in source_weight_mapping.items()
        )

    def _posterior_candidate_signal(
        self,
        *,
        full_probability_matrix: FullProbabilityMatrix,
        game_state: GameState,
        behavior_model: BehavioralLikelihoodModel,
        behavior_map_hypothesis: Dict[str, Dict[int, Card]],
        guesser_id: str,
        target_player_id: str,
        target_slot_index: int,
        guessed_card: Card,
    ) -> Dict[str, Any]:
        map_signal = behavior_model.describe_candidate_value_signal(
            game_state=game_state,
            hypothesis_by_player=behavior_map_hypothesis,
            guesser_id=guesser_id,
            target_player_id=target_player_id,
            target_slot_index=target_slot_index,
            guessed_card=guessed_card,
        )
        context_slots = self._neighbor_context_slot_indices(
            game_state,
            target_player_id=target_player_id,
            target_slot_index=target_slot_index,
        )
        context_candidates: List[Tuple[int, List[Tuple[Card, float]], float]] = []
        covered_probability = 1.0

        for context_slot_index in context_slots:
            slot_candidates, slot_covered_probability = self._context_slot_candidates(
                full_probability_matrix=full_probability_matrix,
                game_state=game_state,
                behavior_map_hypothesis=behavior_map_hypothesis,
                player_id=target_player_id,
                slot_index=context_slot_index,
            )
            if len(slot_candidates) <= 1:
                continue
            context_candidates.append(
                (context_slot_index, slot_candidates, slot_covered_probability)
            )
            covered_probability *= slot_covered_probability

        if not context_candidates:
            return {
                **map_signal,
                "mode": "map_context_fallback",
                "context_candidate_count": 0,
                "context_covered_probability": 1.0,
                "map_signal": map_signal,
            }

        hypothesis_variants: List[Tuple[Dict[str, Dict[int, Card]], float]] = [
            (behavior_map_hypothesis, 1.0),
        ]
        for context_slot_index, slot_candidates, _ in context_candidates:
            next_variants: List[Tuple[Dict[str, Dict[int, Card]], float]] = []
            for hypothesis_variant, variant_weight in hypothesis_variants:
                for card, candidate_weight in slot_candidates:
                    next_variants.append(
                        (
                            self._replace_hypothesis_card(
                                hypothesis_variant,
                                target_player_id,
                                context_slot_index,
                                card,
                            ),
                            variant_weight * candidate_weight,
                        )
                    )
            hypothesis_variants = next_variants

        total_variant_weight = max(
            behavior_model.EPSILON,
            sum(weight for _, weight in hypothesis_variants),
        )
        component_weight_sums: DefaultDict[str, float] = defaultdict(float)
        reason_support_sums: Dict[str, DefaultDict[str, float]] = {
            "progressive": defaultdict(float),
            "anchor": defaultdict(float),
            "boundary": defaultdict(float),
        }

        for hypothesis_variant, variant_weight in hypothesis_variants:
            normalized_weight = variant_weight / total_variant_weight
            variant_signal = behavior_model.describe_candidate_value_signal(
                game_state=game_state,
                hypothesis_by_player=hypothesis_variant,
                guesser_id=guesser_id,
                target_player_id=target_player_id,
                target_slot_index=target_slot_index,
                guessed_card=guessed_card,
            )
            for component_name in ("progressive", "anchor", "boundary"):
                component_weight_sums[component_name] += (
                    normalized_weight * float(variant_signal[component_name]["weight"])
                )
                component_reason = str(variant_signal[component_name].get("reason", "neutral"))
                reason_support_sums[component_name][component_reason] += normalized_weight

        aggregated_components = {
            component_name: self._aggregate_candidate_signal_component(
                component_weight=component_weight_sums[component_name],
                reason_support=reason_support_sums[component_name],
            )
            for component_name in ("progressive", "anchor", "boundary")
        }
        source_mapping = {
            "progressive": aggregated_components["progressive"],
            "same_color_anchor": aggregated_components["anchor"],
            "local_boundary": aggregated_components["boundary"],
        }
        dominant_signal = behavior_model._dominant_signal(source_mapping)
        dominant_component_name = {
            "progressive": "progressive",
            "same_color_anchor": "anchor",
            "local_boundary": "boundary",
        }.get(dominant_signal["source"])
        if dominant_component_name is not None:
            dominant_signal["posterior_support"] = aggregated_components[dominant_component_name]["posterior_support"]
        signal_tags = [
            component["reason"]
            for component in aggregated_components.values()
            if component["reason"] != "neutral"
        ]
        return {
            "signal_tags": signal_tags,
            "dominant_signal": dominant_signal,
            "progressive": aggregated_components["progressive"],
            "anchor": aggregated_components["anchor"],
            "boundary": aggregated_components["boundary"],
            "mode": "neighbor_top_k_posterior",
            "context_candidate_count": len(hypothesis_variants),
            "context_covered_probability": covered_probability,
            "map_signal": map_signal,
        }

    def _behavior_action_posterior_summary(
        self,
        *,
        game_state: GameState,
        behavior_model: BehavioralLikelihoodModel,
        behavior_map_hypothesis: Dict[str, Dict[int, Card]],
        guesser_id: str,
        target_player_id: str,
        target_slot_index: int,
        guessed_card: Card,
    ) -> Dict[str, float]:
        signal = GuessSignal(
            action_index=max(
                0,
                sum(
                    1
                    for action in getattr(game_state, "actions", ())
                    if getattr(action, "action_type", None) == "guess"
                ),
            ),
            guesser_id=guesser_id,
            target_player_id=target_player_id,
            target_slot_index=target_slot_index,
            guessed_card=guessed_card,
            result=False,
            continued_turn=None,
        )
        explanation = behavior_model.explain_signal(
            behavior_map_hypothesis,
            game_state,
            signal,
        )
        posterior = explanation.get("joint_action_posterior", {})
        weight = float(posterior.get("weight", 1.0))
        probability = float(posterior.get("probability", 0.0))
        observed_rank = float(posterior.get("observed_rank", 1.0))
        top_actions = list(posterior.get("top_actions", ()))
        top_probability = float(top_actions[0].get("probability", probability)) if top_actions else probability
        rank_signal = clamp(
            1.0 - ((observed_rank - 1.0) / max(1.0, float(len(top_actions)))),
            0.0,
            1.0,
        )
        support = clamp(
            (0.44 * clamp(probability / max(1e-9, top_probability), 0.0, 1.0))
            + (0.34 * clamp((weight - 1.0) / max(1e-9, behavior_model.JOINT_ACTION_POSTERIOR_MAX_WEIGHT - 1.0), 0.0, 1.0))
            + (0.22 * rank_signal),
            0.0,
            1.0,
        )
        return {
            "probability": probability,
            "weight": weight,
            "observed_rank": observed_rank,
            "support": support,
        }

    def _aggregate_candidate_signal_component(
        self,
        *,
        component_weight: float,
        reason_support: DefaultDict[str, float],
    ) -> Dict[str, Any]:
        sorted_reasons = sorted(
            reason_support.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if not sorted_reasons:
            return {
                "weight": component_weight,
                "reason": "neutral",
                "posterior_support": 0.0,
            }

        top_reason, top_support = sorted_reasons[0]
        if top_reason == "neutral" and len(sorted_reasons) > 1:
            top_reason, top_support = sorted_reasons[1]
        if top_reason == "neutral":
            top_support = 0.0
        return {
            "weight": component_weight,
            "reason": top_reason,
            "posterior_support": top_support,
        }

    def _neighbor_context_slot_indices(
        self,
        game_state: GameState,
        *,
        target_player_id: str,
        target_slot_index: int,
    ) -> List[int]:
        slots = game_state.resolved_ordered_slots(target_player_id)
        index_by_slot = {slot.slot_index: idx for idx, slot in enumerate(slots)}
        order_index = index_by_slot.get(target_slot_index)
        if order_index is None:
            return []

        context_slot_indices: List[int] = []
        if order_index > 0:
            context_slot_indices.append(slots[order_index - 1].slot_index)
        if order_index + 1 < len(slots):
            context_slot_indices.append(slots[order_index + 1].slot_index)
        return context_slot_indices

    def _context_slot_candidates(
        self,
        *,
        full_probability_matrix: FullProbabilityMatrix,
        game_state: GameState,
        behavior_map_hypothesis: Dict[str, Dict[int, Card]],
        player_id: str,
        slot_index: int,
    ) -> Tuple[List[Tuple[Card, float]], float]:
        try:
            slot = game_state.get_slot(player_id, slot_index)
        except ValueError:
            return [], 0.0

        known_card = slot.known_card()
        if known_card is not None:
            return [(known_card, 1.0)], 1.0

        slot_distribution = full_probability_matrix.get(player_id, {}).get(slot_index, {})
        if slot_distribution:
            ranked_candidates = sorted(
                slot_distribution.items(),
                key=lambda item: (-item[1], card_sort_key(item[0])),
            )[: self.BEHAVIOR_MATCH_CONTEXT_TOP_K]
            covered_probability = sum(probability for _, probability in ranked_candidates)
            if covered_probability > 0.0:
                return (
                    [
                        (card, probability / covered_probability)
                        for card, probability in ranked_candidates
                    ],
                    covered_probability,
                )

        fallback_card = behavior_map_hypothesis.get(player_id, {}).get(slot_index)
        if fallback_card is not None:
            return [(fallback_card, 1.0)], 1.0
        return [], 0.0

    def _replace_hypothesis_card(
        self,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        player_id: str,
        slot_index: int,
        card: Card,
    ) -> Dict[str, Dict[int, Card]]:
        cloned = {
            hypothesis_player_id: dict(cards_by_slot)
            for hypothesis_player_id, cards_by_slot in hypothesis_by_player.items()
        }
        player_hypothesis = cloned.setdefault(player_id, {})
        player_hypothesis[slot_index] = card
        return cloned

    def _behavior_match_multiplier(
        self,
        *,
        behavior_match_support: float,
        stable_signal_ratio: float,
    ) -> float:
        if behavior_match_support <= 0.0:
            if stable_signal_ratio >= 0.65:
                return 0.99
            return self.DEFAULT_BEHAVIOR_MATCH_MULTIPLIER
        return clamp(
            1.0 + ((0.32 + (0.16 * stable_signal_ratio)) * behavior_match_support),
            0.99,
            1.08,
        )

    def _hidden_index_by_player_from_matrix(
        self,
        full_probability_matrix: FullProbabilityMatrix,
    ) -> Dict[str, Dict[int, int]]:
        return {
            player_id: {
                slot_index: hidden_index
                for hidden_index, slot_index in enumerate(sorted(probability_matrix))
            }
            for player_id, probability_matrix in full_probability_matrix.items()
        }

    def _strategy_phase_for_state(
        self,
        game_state: Optional[GameState],
    ) -> str:
        if game_state is None:
            return "pre_draw"
        self_slots = game_state.resolved_ordered_slots(game_state.self_player_id)
        if any(getattr(slot, "is_newly_drawn", False) for slot in self_slots):
            latest_self_action = next(
                (
                    action
                    for action in reversed(game_state.actions)
                    if getattr(action, "guesser_id", None) == game_state.self_player_id
                ),
                None,
            )
            if latest_self_action is not None and getattr(latest_self_action, "result", False):
                return "post_hit_chain"
            return "post_draw_opening"
        return "pre_draw"

    def _initiative_recovery_signal(
        self,
        game_state: Optional[GameState],
        *,
        player_id: str,
    ) -> float:
        if game_state is None:
            return 0.0
        self_slots = game_state.resolved_ordered_slots(game_state.self_player_id)
        target_slots = game_state.resolved_ordered_slots(player_id)
        self_revealed = sum(1 for slot in self_slots if slot.is_revealed)
        target_revealed = sum(1 for slot in target_slots if slot.is_revealed)
        revealed_deficit = max(0.0, float(self_revealed - target_revealed))
        self_hidden = sum(1 for slot in self_slots if not slot.is_revealed)
        target_hidden = sum(1 for slot in target_slots if not slot.is_revealed)
        hidden_deficit = max(0.0, float(target_hidden - self_hidden))
        action_pressure = 0.0
        for action in reversed(getattr(game_state, "actions", ())):
            if getattr(action, "action_type", None) != "guess":
                continue
            if getattr(action, "guesser_id", None) == player_id and getattr(action, "result", False):
                action_pressure += 0.5
                break
            if getattr(action, "guesser_id", None) == game_state.self_player_id and getattr(action, "result", False):
                break
        return clamp(
            (
                (0.48 * (revealed_deficit / self.INITIATIVE_RECOVERY_REFERENCE))
                + (0.38 * (hidden_deficit / self.INITIATIVE_RECOVERY_REFERENCE))
                + (0.14 * action_pressure)
            ),
            0.0,
            1.0,
        )

    def _opening_precision_breakdown(
        self,
        *,
        game_state: Optional[GameState],
        player_id: str,
        guessed_card: Card,
        probability: float,
        slot_distribution: Dict[Card, float],
        information_gain: float,
        behavior_confidence: float,
        behavior_action_posterior_support: float,
    ) -> Dict[str, float]:
        phase = self._strategy_phase_for_state(game_state)
        if phase != "post_draw_opening":
            return {
                "phase": phase,
                "support": 0.0,
                "probability_signal": 0.0,
                "margin_signal": 0.0,
                "confidence_signal": 0.0,
                "posterior_signal": 0.0,
                "joker_penalty": 0.0,
                "numeric_competition_signal": 0.0,
                "initiative_recovery_signal": 0.0,
            }
        ranked = sorted(slot_distribution.values(), reverse=True)
        second_probability = float(ranked[1]) if len(ranked) > 1 else 0.0
        best_numeric_probability = max(
            (
                float(candidate_probability)
                for candidate_card, candidate_probability in slot_distribution.items()
                if candidate_card[1] != JOKER
            ),
            default=0.0,
        )
        probability_signal = clamp(probability, 0.0, 1.0)
        margin_signal = clamp(
            (probability - second_probability) / self.OPENING_PRECISION_MARGIN_REFERENCE,
            0.0,
            1.0,
        )
        confidence_signal = clamp(
            (0.55 * behavior_confidence) + (0.45 * clamp(information_gain / 2.0, 0.0, 1.0)),
            0.0,
            1.0,
        )
        posterior_signal = clamp(behavior_action_posterior_support, 0.0, 1.0)
        initiative_recovery_signal = self._initiative_recovery_signal(
            game_state,
            player_id=player_id,
        )
        joker_penalty = 0.0
        numeric_competition_signal = 0.0
        if guessed_card[1] == JOKER and best_numeric_probability > 0.0:
            numeric_competition_signal = clamp(
                (
                    best_numeric_probability
                    + self.OPENING_JOKER_SUPPRESSION_MARGIN
                    - probability
                )
                / max(1e-9, self.OPENING_JOKER_SUPPRESSION_MARGIN),
                0.0,
                1.0,
            )
            joker_penalty = max(
                clamp(
                    (best_numeric_probability - max(0.0, probability - 0.015))
                    / max(1e-9, self.OPENING_PRECISION_MARGIN_REFERENCE),
                    0.0,
                    1.0,
                ),
                numeric_competition_signal,
            )
            posterior_signal *= 1.0 - (
                self.OPENING_JOKER_POSTERIOR_DAMPING * joker_penalty
            )
        support = clamp(
            (0.42 * probability_signal)
            + (0.24 * margin_signal)
            + (0.10 * confidence_signal)
            + (0.12 * posterior_signal)
            + (0.12 * initiative_recovery_signal),
            0.0,
            1.0,
        )
        support *= 1.0 - (self.OPENING_JOKER_PENALTY_SCALE * joker_penalty)
        return {
            "phase": phase,
            "support": support,
            "probability_signal": probability_signal,
            "margin_signal": margin_signal,
            "confidence_signal": confidence_signal,
            "posterior_signal": posterior_signal,
            "joker_penalty": joker_penalty,
            "numeric_competition_signal": numeric_competition_signal,
            "initiative_recovery_signal": initiative_recovery_signal,
        }

    def _evaluate_continue_decision(
        self,
        *,
        best_move: Dict[str, Any],
        second_move: Optional[Dict[str, Any]],
        stop_threshold: float,
        stop_threshold_breakdown: Optional[Dict[str, float]],
        my_hidden_count: int,
    ) -> Dict[str, Any]:
        best_gap = best_move["expected_value"] - (second_move["expected_value"] if second_move else 0.0)
        behavior_match_confidence_breakdown = self._behavior_match_candidate_confidence_breakdown(
            best_move=best_move,
        )
        behavior_match_candidate_confidence = behavior_match_confidence_breakdown["candidate_confidence"]
        behavior_match_component_support = behavior_match_confidence_breakdown["component_support"]
        behavior_match_component_strength = behavior_match_confidence_breakdown["component_strength"]
        behavior_match_component_penalty = behavior_match_confidence_breakdown["component_penalty"]
        behavior_match_context_focus = behavior_match_confidence_breakdown["context_focus"]
        behavior_match_net_structure = clamp(
            behavior_match_component_strength - behavior_match_component_penalty,
            -1.0,
            1.0,
        )
        behavior_match_decision_bonus = self._behavior_match_decision_bonus(
            best_move=best_move,
            candidate_confidence=behavior_match_candidate_confidence,
        )
        self_exposure_level = max(
            best_move.get("self_public_exposure", 0.0),
            best_move.get("self_newly_drawn_exposure", 0.0),
        )

        edge_pressure = 0.0
        if second_move is not None and best_gap < self.STOP_EDGE_REFERENCE:
            edge_pressure = self.STOP_MARGIN_WEAK_EDGE * (
                (self.STOP_EDGE_REFERENCE - max(0.0, best_gap)) / self.STOP_EDGE_REFERENCE
            )
            edge_pressure *= 1.0 + (
                self.STOP_MARGIN_EDGE_SELF_EXPOSURE_BOOST * self_exposure_level
            )

        post_hit_continue_margin = best_move.get("post_hit_continue_margin", 0.0)
        rollout_pressure = 0.0
        if best_move.get("post_hit_stop_score", 0.0) > 0.0 and post_hit_continue_margin < 0.0:
            rollout_pressure = self.STOP_MARGIN_WEAK_ROLLOUT * min(
                1.0,
                (-post_hit_continue_margin) / self.ROLLOUT_MARGIN_REFERENCE,
            )

        post_hit_best_gap = best_move.get("post_hit_best_gap", 0.0)
        fragile_rollout_pressure = 0.0
        if (
            best_move.get("post_hit_stop_score", 0.0) > 0.0
            and post_hit_continue_margin > 0.0
            and post_hit_best_gap < self.POST_HIT_GAP_REFERENCE
        ):
            fragile_rollout_pressure = self.STOP_MARGIN_FRAGILE_POST_HIT * (
                (self.POST_HIT_GAP_REFERENCE - max(0.0, post_hit_best_gap)) / self.POST_HIT_GAP_REFERENCE
            )

        post_hit_top_k_continue_margin = best_move.get("post_hit_top_k_continue_margin", 0.0)
        top_k_rollout_pressure = 0.0
        if (
            best_move.get("post_hit_stop_score", 0.0) > 0.0
            and post_hit_continue_margin > 0.0
            and post_hit_top_k_continue_margin < post_hit_continue_margin
        ):
            top_k_rollout_pressure = self.STOP_MARGIN_TOP_K_SUPPORT * min(
                1.0,
                1.0 - (
                    post_hit_top_k_continue_margin
                    / max(1e-9, post_hit_continue_margin)
                ),
            )

        attackability_after_hit = best_move.get("attackability_after_hit", 0.0)
        attackability_pressure = 0.0
        if best_move.get("post_hit_stop_score", 0.0) <= 0.0 and attackability_after_hit < self.ATTACKABILITY_REFERENCE:
            attackability_pressure = self.STOP_MARGIN_LOW_ATTACKABILITY * (
                (self.ATTACKABILITY_REFERENCE - attackability_after_hit)
                / max(self.ATTACKABILITY_REFERENCE, 1e-9)
            )

        behavior_rollout_pressure = 0.0
        if (
            best_move.get("post_hit_stop_score", 0.0) > 0.0
            and post_hit_continue_margin > 0.0
            and best_move.get("continuation_value", 0.0) > 0.0
        ):
            behavior_rollout_pressure = self.STOP_MARGIN_BEHAVIOR_ROLLOUT * (
                1.0 - behavior_match_candidate_confidence
            )

        stop_score = (
            stop_threshold
            + edge_pressure
            + rollout_pressure
            + fragile_rollout_pressure
            + top_k_rollout_pressure
            + attackability_pressure
            + behavior_rollout_pressure
        )
        strategy_objective_support = best_move.get(
            "strategy_objective",
            best_move.get(
                "strategy_objective_core",
                best_move["expected_value"],
            ),
        )
        continue_score = strategy_objective_support
        strategy_core_support = best_move.get(
            "strategy_objective_core",
            best_move["expected_value"],
        )
        continue_score += max(0.0, strategy_core_support - strategy_objective_support)
        continue_score += behavior_match_decision_bonus
        failure_recovery_signal = clamp(
            (
                best_move.get("post_hit_failure_recovery_bonus", 0.0)
                + best_move.get("post_hit_failed_switch_bonus", 0.0)
            ) / self.FAILURE_RECOVERY_REFERENCE,
            0.0,
            1.0,
        )
        failure_recovery_pressure = (
            self.STOP_MARGIN_FAILURE_RECOVERY * failure_recovery_signal
        )
        continue_score += failure_recovery_pressure
        attack_window_support = (
            self.STOP_MARGIN_ATTACK_WINDOW
            * float(best_move.get("target_attack_window_signal", 0.0))
        )
        continue_score += attack_window_support
        joint_collapse_support = (
            self.STOP_MARGIN_JOINT_COLLAPSE
            * float(best_move.get("joint_collapse_signal", 0.0))
        )
        continue_score += joint_collapse_support
        global_propagation_support = (
            self.STOP_MARGIN_GLOBAL_PROPAGATION
            * float(best_move.get("global_propagation_signal", 0.0))
        )
        continue_score += global_propagation_support
        public_reveal_bridge_support = (
            self.STOP_MARGIN_PUBLIC_BRIDGE
            * float(best_move.get("public_reveal_bridge_signal", 0.0))
        )
        continue_score += public_reveal_bridge_support
        target_chain_support = (
            self.STOP_MARGIN_TARGET_CHAIN
            * float(best_move.get("target_chain_signal", 0.0))
        )
        continue_score += target_chain_support
        target_finish_chain_support = (
            self.STOP_MARGIN_FINISH_CHAIN
            * float(best_move.get("target_finish_chain_signal", 0.0))
        )
        continue_score += target_finish_chain_support
        branch_search_support = (
            self.STOP_MARGIN_BRANCH_SEARCH
            * float(best_move.get("post_hit_branch_search_signal", 0.0))
        )
        continue_score += branch_search_support
        expectimax_support = (
            self.STOP_MARGIN_EXPECTIMAX
            * float(best_move.get("post_hit_expectimax_signal", 0.0))
        )
        continue_score += expectimax_support
        mcts_support = (
            self.STOP_MARGIN_MCTS
            * float(best_move.get("post_hit_mcts_signal", 0.0))
        )
        continue_score += mcts_support
        global_mcts_support = (
            self.STOP_MARGIN_MCTS
            * float(best_move.get("global_mcts_signal", 0.0))
        )
        continue_score += global_mcts_support
        post_hit_behavior_support_breakdown = self._post_hit_behavior_support_breakdown(
            best_move=best_move,
        )
        post_hit_behavior_support_adjustment = post_hit_behavior_support_breakdown["adjustment"]
        continue_score += post_hit_behavior_support_adjustment
        pre_structure_continue_margin = continue_score - stop_score
        behavior_match_decision_structure_adjustment = self._behavior_match_decision_structure_adjustment(
            decision_bonus=behavior_match_decision_bonus,
            net_structure=behavior_match_net_structure,
            component_support=behavior_match_component_support,
            context_focus=behavior_match_context_focus,
            pre_structure_continue_margin=pre_structure_continue_margin,
        )
        continue_score += behavior_match_decision_structure_adjustment
        continue_margin = continue_score - stop_score
        guard_relief_signal = float(
            (stop_threshold_breakdown or {}).get("guard_relief_signal", 0.0)
        )
        low_confidence_guard_margin = (
            self.LOW_CONFIDENCE_GUARD_MARGIN
            + (self.LOW_CONFIDENCE_SELF_EXPOSURE_BOOST * self_exposure_level)
        ) * (1.0 - (0.35 * guard_relief_signal))
        weak_edge_guard_margin = (
            self.WEAK_EDGE_GUARD_MARGIN
            + (self.WEAK_EDGE_SELF_EXPOSURE_BOOST * self_exposure_level)
        ) * (1.0 - (0.35 * guard_relief_signal))
        self_exposure_guard_margin = self.SELF_EXPOSURE_GUARD_MARGIN * (
            1.0 - (0.25 * guard_relief_signal)
        )
        epsilon = self.EPSILON
        margin_relief_signal = clamp(
            continue_margin / self.DRAW_ROLLOUT_CONTINUE_MARGIN_REFERENCE,
            0.0,
            1.0,
        )

        low_confidence_guard_signal = 0.0
        if my_hidden_count <= 2:
            low_confidence_guard_signal = clamp(
                (0.45 - float(best_move["win_probability"])) / 0.20,
                0.0,
                1.0,
            ) * clamp(
                (0.55 - float(best_move.get("continuation_likelihood", 0.0)))
                / 0.20,
                0.0,
                1.0,
            )
        weak_edge_guard_signal = 0.0
        if second_move is not None:
            weak_edge_guard_signal = clamp(
                (0.10 - best_gap) / 0.10,
                0.0,
                1.0,
            ) * clamp(
                (
                    second_move.get("strategy_objective", second_move["expected_value"])
                    / max(
                        epsilon,
                        best_move.get(
                            "strategy_objective",
                            best_move["expected_value"],
                        ),
                    )
                ),
                0.0,
                1.0,
            )
        self_exposure_guard_signal = clamp(
            (
                max(
                    best_move.get("self_public_exposure", 0.0),
                    best_move.get("self_newly_drawn_exposure", 0.0),
                )
                - self.SELF_EXPOSURE_GUARD_REFERENCE
            )
            / max(
                epsilon,
                1.0 - self.SELF_EXPOSURE_GUARD_REFERENCE,
            ),
            0.0,
            1.0,
        )
        guard_stop_logit = (
            self.GUARD_POLICY_BIAS
            + (
                self.GUARD_POLICY_LOW_CONFIDENCE_WEIGHT
                * low_confidence_guard_signal
            )
            + (
                self.GUARD_POLICY_WEAK_EDGE_WEIGHT
                * weak_edge_guard_signal
            )
            + (
                self.GUARD_POLICY_SELF_EXPOSURE_WEIGHT
                * self_exposure_guard_signal
            )
            - (
                self.GUARD_POLICY_MARGIN_RELIEF_WEIGHT
                * margin_relief_signal
            )
            - (
                self.GUARD_POLICY_SEARCH_RELIEF_WEIGHT
                * guard_relief_signal
            )
        )
        guard_stop_probability = 1.0 / (1.0 + exp(-guard_stop_logit))
        guard_continue_probability = 1.0 - guard_stop_probability
        low_confidence_guard = (
            low_confidence_guard_signal >= 0.55 and guard_stop_probability >= 0.50
        )
        weak_edge_guard = (
            weak_edge_guard_signal >= 0.55 and guard_stop_probability >= 0.50
        )
        self_exposure_guard = (
            self_exposure_guard_signal >= 0.55 and guard_stop_probability >= 0.50
        )

        guard_policy_stop_boost = (
            self.STOP_MARGIN_GUARD_POLICY_SCALE * guard_stop_probability
        )
        total_guard_signal = (
            low_confidence_guard_signal
            + weak_edge_guard_signal
            + self_exposure_guard_signal
        )
        low_confidence_guard_boost = (
            guard_policy_stop_boost
            * low_confidence_guard_signal
            / max(epsilon, total_guard_signal)
            if total_guard_signal > 0.0
            else 0.0
        )
        weak_edge_guard_boost = (
            guard_policy_stop_boost
            * weak_edge_guard_signal
            / max(epsilon, total_guard_signal)
            if total_guard_signal > 0.0
            else 0.0
        )
        self_exposure_guard_boost = (
            guard_policy_stop_boost
            * self_exposure_guard_signal
            / max(epsilon, total_guard_signal)
            if total_guard_signal > 0.0
            else 0.0
        )
        continue_strategy_objective = continue_score
        stop_strategy_objective = (
            stop_score
            + guard_policy_stop_boost
        )
        strategy_action_margin = (
            continue_strategy_objective - stop_strategy_objective
        )
        continue_margin = strategy_action_margin
        should_continue = strategy_action_margin > 0.0
        return {
            "should_continue": should_continue,
            "continue_score": continue_score,
            "stop_score": stop_score,
            "continue_margin": continue_margin,
            "strategy_objective_continue": continue_strategy_objective,
            "strategy_objective_stop": stop_strategy_objective,
            "strategy_action_margin": strategy_action_margin,
            "best_gap": best_gap,
            "behavior_match_decision_bonus": behavior_match_decision_bonus,
            "behavior_match_decision_structure_adjustment": behavior_match_decision_structure_adjustment,
            "behavior_match_net_structure": behavior_match_net_structure,
            "behavior_match_candidate_confidence": behavior_match_candidate_confidence,
            "behavior_match_component_support": behavior_match_component_support,
            "behavior_match_component_strength": behavior_match_component_strength,
            "behavior_match_component_penalty": behavior_match_component_penalty,
            "behavior_match_context_focus": behavior_match_context_focus,
            "low_confidence_guard": low_confidence_guard,
            "weak_edge_guard": weak_edge_guard,
            "self_exposure_guard": self_exposure_guard,
            "low_confidence_guard_margin": low_confidence_guard_margin,
            "weak_edge_guard_margin": weak_edge_guard_margin,
            "self_exposure_guard_margin": self_exposure_guard_margin,
            "guard_relief_signal": guard_relief_signal,
            "decision_score_breakdown": {
                "base_stop_threshold": float(
                    (stop_threshold_breakdown or {}).get("base_stop_threshold", stop_threshold)
                ),
                "short_hand_threshold": float(
                    (stop_threshold_breakdown or {}).get("short_hand_threshold", 0.0)
                ),
                "self_exposure_threshold": float(
                    (stop_threshold_breakdown or {}).get("self_exposure_threshold", 0.0)
                ),
                "newly_drawn_threshold": float(
                    (stop_threshold_breakdown or {}).get("newly_drawn_threshold", 0.0)
                ),
                "finish_fragility_threshold": float(
                    (stop_threshold_breakdown or {}).get("finish_fragility_threshold", 0.0)
                ),
                "low_confidence_threshold": float(
                    (stop_threshold_breakdown or {}).get("low_confidence_threshold", 0.0)
                ),
                "weak_continuation_threshold": float(
                    (stop_threshold_breakdown or {}).get("weak_continuation_threshold", 0.0)
                ),
                "strategy_objective_credit": float(
                    (stop_threshold_breakdown or {}).get("strategy_objective_credit", 0.0)
                ),
                "search_credit": float(
                    (stop_threshold_breakdown or {}).get("search_credit", 0.0)
                ),
                "search_depth_credit": float(
                    (stop_threshold_breakdown or {}).get("search_depth_credit", 0.0)
                ),
                "guard_relief_signal": guard_relief_signal,
                "total_stop_threshold": stop_threshold,
                "self_exposure_guard_signal": (
                    self_exposure_guard_signal
                ),
                "low_confidence_guard_signal": low_confidence_guard_signal,
                "weak_edge_guard_signal": weak_edge_guard_signal,
                "low_confidence_guard_margin": low_confidence_guard_margin,
                "weak_edge_guard_margin": weak_edge_guard_margin,
                "self_exposure_guard_margin": self_exposure_guard_margin,
                "edge_pressure": edge_pressure,
                "rollout_pressure": rollout_pressure,
                "fragile_rollout_pressure": fragile_rollout_pressure,
                "top_k_rollout_pressure": top_k_rollout_pressure,
                "attackability_pressure": attackability_pressure,
                "behavior_rollout_pressure": behavior_rollout_pressure,
                "post_hit_behavior_support_adjustment": post_hit_behavior_support_adjustment,
                "post_hit_behavior_support_gain": post_hit_behavior_support_breakdown["support_gain"],
                "post_hit_behavior_fragility_drag": post_hit_behavior_support_breakdown["fragility_drag"],
                "post_hit_behavior_support_signal": post_hit_behavior_support_breakdown["support_signal"],
                "post_hit_behavior_fragility_signal": post_hit_behavior_support_breakdown["fragility_signal"],
                "post_hit_behavior_support_strength": post_hit_behavior_support_breakdown["support_strength"],
                "post_hit_behavior_fragility_strength": post_hit_behavior_support_breakdown["fragility_strength"],
                "behavior_match_decision_bonus": behavior_match_decision_bonus,
                "behavior_match_decision_structure_adjustment": behavior_match_decision_structure_adjustment,
                "behavior_match_net_structure": behavior_match_net_structure,
                "behavior_match_candidate_confidence": behavior_match_candidate_confidence,
                "behavior_match_component_support": behavior_match_component_support,
                "behavior_match_component_strength": behavior_match_component_strength,
                "behavior_match_component_penalty": behavior_match_component_penalty,
                "behavior_match_context_focus": behavior_match_context_focus,
                "failure_recovery_pressure": failure_recovery_pressure,
                "strategy_objective_support": strategy_objective_support,
                "strategy_objective_continue": continue_strategy_objective,
                "strategy_objective_stop": stop_strategy_objective,
                "strategy_action_margin": strategy_action_margin,
                "attack_window_support": attack_window_support,
                "joint_collapse_support": joint_collapse_support,
                "global_propagation_support": global_propagation_support,
                "public_reveal_bridge_support": public_reveal_bridge_support,
                "target_chain_support": target_chain_support,
                "target_finish_chain_support": target_finish_chain_support,
                "branch_search_support": branch_search_support,
                "expectimax_support": expectimax_support,
                "mcts_support": mcts_support,
                "global_mcts_support": global_mcts_support,
                "low_confidence_guard_boost": low_confidence_guard_boost,
                "weak_edge_guard_boost": weak_edge_guard_boost,
                "self_exposure_guard_boost": self_exposure_guard_boost,
                "guard_policy_stop_boost": guard_policy_stop_boost,
                "guard_stop_logit": guard_stop_logit,
                "guard_stop_probability": guard_stop_probability,
                "guard_continue_probability": guard_continue_probability,
                "margin_relief_signal": margin_relief_signal,
            },
        }

    def _behavior_match_candidate_confidence_breakdown(
        self,
        *,
        best_move: Dict[str, Any],
    ) -> Dict[str, float]:
        precomputed_keys = (
            "behavior_match_candidate_confidence",
            "behavior_match_component_support",
            "behavior_match_component_strength",
            "behavior_match_component_penalty",
            "behavior_match_context_focus",
        )
        if all(key in best_move for key in precomputed_keys):
            return {
                "candidate_confidence": float(best_move["behavior_match_candidate_confidence"]),
                "component_support": float(best_move["behavior_match_component_support"]),
                "component_strength": float(best_move["behavior_match_component_strength"]),
                "component_penalty": float(best_move["behavior_match_component_penalty"]),
                "context_focus": float(best_move["behavior_match_context_focus"]),
            }

        candidate_signal = best_move.get("behavior_candidate_signal")
        if not isinstance(candidate_signal, dict):
            return {
                "candidate_confidence": 1.0,
                "component_support": 1.0,
                "component_strength": 1.0,
                "component_penalty": 0.0,
                "context_focus": 1.0,
            }

        mode = str(candidate_signal.get("mode", ""))
        if mode == "map_context_fallback":
            return {
                "candidate_confidence": 1.0,
                "component_support": 1.0,
                "component_strength": 1.0,
                "component_penalty": 0.0,
                "context_focus": 1.0,
            }

        dominant_signal = candidate_signal.get("dominant_signal", {})
        posterior_support = clamp(
            float(dominant_signal.get("posterior_support", 0.0)),
            0.0,
            1.0,
        )
        context_covered_probability = clamp(
            float(candidate_signal.get("context_covered_probability", 0.0)),
            0.0,
            1.0,
        )
        component_strength = self._behavior_match_component_strength(
            candidate_signal=candidate_signal,
        )
        component_penalty = self._behavior_match_component_penalty(
            candidate_signal=candidate_signal,
        )
        component_support = self._behavior_match_component_support(
            candidate_signal=candidate_signal,
            dominant_signal_support=posterior_support,
            component_strength=component_strength,
            component_penalty=component_penalty,
        )
        context_focus = self._behavior_match_context_focus(candidate_signal=candidate_signal)
        if (
            mode != "neighbor_top_k_posterior"
            and posterior_support <= 0.0
            and context_covered_probability <= 0.0
        ):
            return {
                "candidate_confidence": 1.0,
                "component_support": 1.0,
                "component_strength": 1.0,
                "component_penalty": 0.0,
                "context_focus": 1.0,
            }
        candidate_confidence = clamp(
            (0.30 * posterior_support)
            + (0.30 * context_covered_probability)
            + (0.25 * component_support)
            + (0.15 * context_focus),
            0.0,
            1.0,
        )
        return {
            "candidate_confidence": candidate_confidence,
            "component_support": component_support,
            "component_strength": component_strength,
            "component_penalty": component_penalty,
            "context_focus": context_focus,
        }

    def _behavior_match_ranking_breakdown(
        self,
        *,
        best_move: Dict[str, Any],
    ) -> Dict[str, float]:
        confidence_breakdown = self._behavior_match_candidate_confidence_breakdown(
            best_move=best_move,
        )
        raw_bonus = max(0.0, float(best_move.get("behavior_match_bonus", 0.0)))
        net_structure = clamp(
            confidence_breakdown["component_strength"]
            - confidence_breakdown["component_penalty"],
            -1.0,
            1.0,
        )
        structure_adjustment = (
            raw_bonus
            * self.BEHAVIOR_MATCH_NET_STRUCTURE_SCALE
            * net_structure
        )
        ranking_bonus = max(
            0.0,
            (raw_bonus * confidence_breakdown["candidate_confidence"])
            + structure_adjustment,
        )
        return {
            "ranking_bonus": ranking_bonus,
            "net_structure": net_structure,
            "structure_adjustment": structure_adjustment,
        }

    def _behavior_match_component_strength(
        self,
        *,
        candidate_signal: Dict[str, Any],
    ) -> float:
        component_strengths = []
        for component_name in ("progressive", "anchor", "boundary"):
            component = candidate_signal.get(component_name)
            if not isinstance(component, dict):
                continue
            component_strengths.append(
                clamp(
                    (float(component.get("weight", 1.0)) - 1.0)
                    / self.BEHAVIOR_MATCH_COMPONENT_WEIGHT_REFERENCE,
                    0.0,
                    1.0,
                )
            )
        if component_strengths:
            return sum(component_strengths) / len(component_strengths)
        return 0.0

    def _behavior_match_component_penalty(
        self,
        *,
        candidate_signal: Dict[str, Any],
    ) -> float:
        component_penalties = []
        for component_name in ("progressive", "anchor", "boundary"):
            component = candidate_signal.get(component_name)
            if not isinstance(component, dict):
                continue
            component_penalties.append(
                clamp(
                    (1.0 - float(component.get("weight", 1.0)))
                    / self.BEHAVIOR_MATCH_COMPONENT_WEIGHT_REFERENCE,
                    0.0,
                    1.0,
                )
            )
        if component_penalties:
            return sum(component_penalties) / len(component_penalties)
        return 0.0

    def _behavior_match_component_support(
        self,
        *,
        candidate_signal: Dict[str, Any],
        dominant_signal_support: float,
        component_strength: float,
        component_penalty: float,
    ) -> float:
        component_supports = []
        for component_name in ("progressive", "anchor", "boundary"):
            component = candidate_signal.get(component_name)
            if not isinstance(component, dict):
                continue
            component_supports.append(
                clamp(float(component.get("posterior_support", 0.0)), 0.0, 1.0)
            )
        if component_supports:
            average_support = sum(component_supports) / len(component_supports)
        else:
            average_support = dominant_signal_support
        return clamp(
            (0.75 * average_support)
            + (0.25 * component_strength)
            - (self.BEHAVIOR_MATCH_COMPONENT_PENALTY_WEIGHT * component_penalty),
            0.0,
            1.0,
        )

    def _behavior_match_context_focus(
        self,
        *,
        candidate_signal: Dict[str, Any],
    ) -> float:
        context_candidate_count = max(
            1.0,
            float(candidate_signal.get("context_candidate_count", 1.0)),
        )
        return min(
            1.0,
            sqrt(self.BEHAVIOR_MATCH_CONTEXT_COUNT_REFERENCE / context_candidate_count),
        )

    def _behavior_match_decision_bonus(
        self,
        *,
        best_move: Dict[str, Any],
        candidate_confidence: float,
    ) -> float:
        raw_bonus = max(0.0, best_move.get("behavior_match_bonus", 0.0))
        if raw_bonus <= 0.0:
            return 0.0

        support_scale = min(
            1.0,
            max(0.0, float(best_move.get("behavior_match_support", 0.0)))
            / self.BEHAVIOR_MATCH_SUPPORT_REFERENCE,
        )
        stable_ratio = clamp(
            float(best_move.get("behavior_guidance_stable_ratio", 0.0)),
            0.0,
            1.0,
        )
        return (
            raw_bonus
            * self.BEHAVIOR_MATCH_DECISION_SCALE
            * support_scale
            * stable_ratio
            * clamp(candidate_confidence, 0.0, 1.0)
        )

    def _post_hit_behavior_support_breakdown(
        self,
        *,
        best_move: Dict[str, Any],
    ) -> Dict[str, float]:
        if (
            float(best_move.get("post_hit_stop_score", 0.0)) <= 0.0
            or float(best_move.get("post_hit_continue_margin", 0.0)) <= 0.0
            or float(best_move.get("continuation_value", 0.0)) <= 0.0
        ):
            return {
                "adjustment": 0.0,
                "support_gain": 0.0,
                "fragility_drag": 0.0,
                "support_signal": 0.0,
                "fragility_signal": 0.0,
                "support_strength": 0.0,
                "fragility_strength": 0.0,
            }

        ranking_edge = clamp(
            (
                float(best_move.get("post_hit_top_k_continue_margin", 0.0))
                - float(best_move.get("post_hit_top_k_expected_continue_margin", 0.0))
            ) / self.POST_HIT_BEHAVIOR_SUPPORT_REFERENCE,
            -1.0,
            1.0,
        )
        guidance_edge = clamp(
            (
                float(best_move.get("post_hit_guidance_multiplier", self.DEFAULT_BEHAVIOR_GUIDANCE_MULTIPLIER))
                - 1.0
            ) / 0.12,
            -1.0,
            1.0,
        )
        support_ratio_edge = clamp(
            float(best_move.get("post_hit_top_k_support_ratio", 0.0))
            - float(best_move.get("post_hit_top_k_expected_support_ratio", 0.0)),
            -1.0,
            1.0,
        )
        support_strength = clamp(
            (0.5 * float(best_move.get("post_hit_guidance_stable_ratio", 0.0)))
            + (0.5 * float(best_move.get("post_hit_guidance_support", 0.0))),
            0.0,
            1.0,
        )
        fragility_strength = clamp(
            (0.60 * (1.0 - float(best_move.get("post_hit_guidance_stable_ratio", 0.0))))
            + (0.40 * (1.0 - float(best_move.get("post_hit_guidance_support", 0.0)))),
            0.0,
            1.0,
        )
        support_signal = clamp(
            (0.55 * max(0.0, ranking_edge))
            + (0.25 * max(0.0, guidance_edge))
            + (0.20 * max(0.0, support_ratio_edge)),
            0.0,
            1.0,
        )
        fragility_signal = clamp(
            (0.50 * max(0.0, -ranking_edge))
            + (0.20 * max(0.0, -guidance_edge))
            + (0.30 * max(0.0, -support_ratio_edge)),
            0.0,
            1.0,
        )
        support_gain = self.POST_HIT_BEHAVIOR_SUPPORT_SCALE * support_signal * support_strength
        fragility_drag = self.POST_HIT_BEHAVIOR_SUPPORT_SCALE * fragility_signal * fragility_strength
        return {
            "adjustment": support_gain - fragility_drag,
            "support_gain": support_gain,
            "fragility_drag": fragility_drag,
            "support_signal": support_signal,
            "fragility_signal": fragility_signal,
            "support_strength": support_strength,
            "fragility_strength": fragility_strength,
        }

    def _behavior_match_decision_structure_adjustment(
        self,
        *,
        decision_bonus: float,
        net_structure: float,
        component_support: float,
        context_focus: float,
        pre_structure_continue_margin: float,
    ) -> float:
        decision_bonus = max(0.0, float(decision_bonus))
        if decision_bonus <= 0.0:
            return 0.0

        margin = abs(float(pre_structure_continue_margin))
        if margin >= self.BEHAVIOR_MATCH_DECISION_WINDOW:
            return 0.0

        support_gate = (
            clamp(component_support, 0.0, 1.0)
            * clamp(context_focus, 0.0, 1.0)
        )
        edge_scale = 1.0 - (margin / self.BEHAVIOR_MATCH_DECISION_WINDOW)
        return (
            decision_bonus
            * self.BEHAVIOR_MATCH_DECISION_NET_STRUCTURE_SCALE
            * clamp(net_structure, -1.0, 1.0)
            * support_gate
            * edge_scale
        )

    def _stop_threshold_breakdown(
        self,
        *,
        risk_factor: float,
        my_hidden_count: int,
        best_move: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        base_stop_threshold = self.STOP_MARGIN_BASE + max(
            0.0,
            (risk_factor - self.HIT_REWARD) * self.STOP_MARGIN_RISK_SCALE,
        )
        short_hand_threshold = 0.0
        if my_hidden_count <= 2:
            short_hand_threshold += self.STOP_MARGIN_SHORT_HAND
        if my_hidden_count <= 1:
            short_hand_threshold += self.STOP_MARGIN_SHORT_HAND

        self_exposure_threshold = 0.0
        newly_drawn_threshold = 0.0
        finish_fragility_threshold = 0.0
        low_confidence_threshold = 0.0
        weak_continuation_threshold = 0.0
        strategy_objective_credit = 0.0
        search_credit = 0.0
        search_depth_credit = 0.0
        guard_relief_signal = 0.0

        if best_move is not None:
            self_exposure_threshold = self.STOP_MARGIN_SELF_EXPOSURE * clamp(
                float(best_move.get("self_public_exposure", 0.0)),
                0.0,
                1.0,
            )
            newly_drawn_threshold = self.STOP_MARGIN_NEW_DRAWN_EXPOSURE * clamp(
                float(best_move.get("self_newly_drawn_exposure", 0.0)),
                0.0,
                1.0,
            )
            finish_fragility_threshold = self.STOP_MARGIN_FINISH_FRAGILITY * clamp(
                float(best_move.get("self_finish_fragility", 0.0)),
                0.0,
                1.0,
            )
            if best_move["win_probability"] < 0.5:
                low_confidence_threshold = self.STOP_MARGIN_LOW_CONFIDENCE * (
                    (0.5 - best_move["win_probability"]) / 0.5
                )
            if best_move.get("continuation_likelihood", 0.0) < 0.5:
                weak_continuation_threshold = self.STOP_MARGIN_WEAK_CONTINUATION * (
                    (0.5 - best_move.get("continuation_likelihood", 0.0)) / 0.5
                )
            strategy_objective_credit = self.STOP_MARGIN_OBJECTIVE_CREDIT * clamp(
                float(best_move.get("strategy_objective_core", best_move.get("strategy_objective", 0.0)))
                / self.HIT_REWARD,
                0.0,
                1.0,
            )
            search_signal = clamp(
                max(
                    float(best_move.get("post_hit_tree_search_signal", 0.0)),
                    float(best_move.get("post_hit_expectimax_signal", 0.0)),
                    float(best_move.get("post_hit_mcts_signal", 0.0)),
                    float(best_move.get("post_hit_branch_search_signal", 0.0)),
                    float(best_move.get("global_mcts_signal", 0.0)),
                ),
                0.0,
                1.0,
            )
            search_depth_signal = clamp(
                max(
                    float(best_move.get("post_hit_mcts_max_depth", 0.0))
                    / max(1.0, float(self.DEEP_ROLLOUT_DEPTH)),
                    float(best_move.get("post_hit_mcts_node_count", 0.0))
                    / max(
                        1.0,
                        float(
                            best_move.get(
                                "post_hit_mcts_simulation_budget",
                                self.MCTS_SIMULATION_COUNT,
                            )
                            * self.MCTS_TOP_K
                            * self.MCTS_DEEP_CHILD_TOP_K
                        ),
                    ),
                    float(best_move.get("global_mcts_max_depth", 0.0))
                    / max(1.0, float(self.DEEP_ROLLOUT_DEPTH + 1)),
                    float(best_move.get("global_mcts_node_count", 0.0))
                    / max(
                        1.0,
                        float(
                            best_move.get(
                                "global_mcts_simulation_budget",
                                self.MCTS_SIMULATION_COUNT,
                            )
                            * self.GLOBAL_MCTS_TOP_K
                            * self.MCTS_DEEP_CHILD_TOP_K
                        ),
                    ),
                ),
                0.0,
                1.0,
            )
            search_credit = self.STOP_MARGIN_SEARCH_CREDIT * search_signal
            search_depth_credit = (
                self.STOP_MARGIN_SEARCH_DEPTH_CREDIT * search_depth_signal
            )
            guard_relief_signal = clamp(
                (
                    0.45
                    * clamp(
                        float(
                            best_move.get(
                                "strategy_objective_core",
                                best_move.get("strategy_objective", 0.0),
                            )
                        )
                        / self.HIT_REWARD,
                        0.0,
                        1.0,
                    )
                    + (0.35 * search_signal)
                    + (0.20 * search_depth_signal)
                ),
                0.0,
                1.0,
            )
            low_confidence_threshold *= (
                1.0 - (self.STOP_MARGIN_GUARD_RELIEF_SCALE * guard_relief_signal)
            )
            weak_continuation_threshold *= (
                1.0 - (self.STOP_MARGIN_GUARD_RELIEF_SCALE * guard_relief_signal)
            )

        threshold = (
            base_stop_threshold
            + short_hand_threshold
            + self_exposure_threshold
            + newly_drawn_threshold
            + finish_fragility_threshold
            + low_confidence_threshold
            + weak_continuation_threshold
            - strategy_objective_credit
            - search_credit
            - search_depth_credit
        )
        return {
            "base_stop_threshold": base_stop_threshold,
            "short_hand_threshold": short_hand_threshold,
            "self_exposure_threshold": self_exposure_threshold,
            "newly_drawn_threshold": newly_drawn_threshold,
            "finish_fragility_threshold": finish_fragility_threshold,
            "low_confidence_threshold": low_confidence_threshold,
            "weak_continuation_threshold": weak_continuation_threshold,
            "strategy_objective_credit": strategy_objective_credit,
            "search_credit": search_credit,
            "search_depth_credit": search_depth_credit,
            "guard_relief_signal": guard_relief_signal,
            "threshold": threshold,
        }

    def _stop_threshold(
        self,
        *,
        risk_factor: float,
        my_hidden_count: int,
        best_move: Optional[Dict[str, Any]],
    ) -> float:
        return self._stop_threshold_breakdown(
            risk_factor=risk_factor,
            my_hidden_count=my_hidden_count,
            best_move=best_move,
        )["threshold"]

    def _build_stop_reason(
        self,
        *,
        should_continue: bool,
        best_move: Dict[str, Any],
        stop_threshold: float,
        stop_score: float,
        continue_score: float,
        continue_margin: float,
        low_confidence_guard: bool,
        weak_edge_guard: bool,
        self_exposure_guard: bool,
    ) -> str:
        if should_continue:
            if best_move.get("continuation_value", 0.0) > 0.0:
                return (
                    f"继续评分 {continue_score:.2f} 高于停手评分 {stop_score:.2f}，"
                    "命中后继续收益也支持进攻。"
                )
            return f"继续评分 {continue_score:.2f} 高于停手评分 {stop_score:.2f}，可以继续进攻。"
        if low_confidence_guard:
            return "当前属于高暴露局面，命中率与续压潜力都偏弱，建议停手。"
        if weak_edge_guard:
            return "最佳动作与次优动作差距过小，当前进攻边际优势不足，建议停手。"
        if self_exposure_guard:
            return "当前自手牌公开结构已经偏尖，继续进攻的自曝风险过高，建议停手。"
        if stop_score > stop_threshold:
            return (
                f"继续评分 {continue_score:.2f} 仍低于停手评分 {stop_score:.2f}；"
                f"当前续压空间不足，净优势 {continue_margin:.2f}，建议停手。"
            )
        return f"最佳动作期望收益 {best_move['expected_value']:.2f} 未超过停手阈值 {stop_threshold:.2f}。"

    def _build_reason(
        self,
        probability: float,
        info_gain: float,
        continuation_value: float,
        miss_penalty: float,
        continuation_likelihood: float,
    ) -> str:
        if continuation_value > max(0.35, info_gain * self.INFORMATION_GAIN_WEIGHT):
            if continuation_likelihood >= 0.60:
                return "命中后继续收益较高，且续压概率也偏高，适合主动压回合。"
            return "命中后存在续压价值，但更依赖后续局面兑现。"
        if probability >= 0.62 and miss_penalty < self.HIT_REWARD:
            return "当前命中率较高，且风险处于可接受范围。"
        if info_gain * self.INFORMATION_GAIN_WEIGHT >= max(0.45, continuation_value):
            return "即便不是最稳的一猜，也能显著压缩该位置的不确定性。"
        return "这步主要依靠当前命中率与风险平衡取得正收益。"

    def _best_immediate_ev(
        self,
        full_probability_matrix: FullProbabilityMatrix,
        my_hidden_count: int,
    ) -> float:
        risk_factor = self.calculate_risk_factor(my_hidden_count)
        best_value = 0.0
        for probability_matrix in full_probability_matrix.values():
            for slot_distribution in probability_matrix.values():
                for card, probability in slot_distribution.items():
                    info_gain = self._expected_slot_entropy_reduction(slot_distribution, card)
                    value = (probability * self.HIT_REWARD) - ((1.0 - probability) * risk_factor) + (info_gain * self.INFORMATION_GAIN_WEIGHT)
                    best_value = max(best_value, value)
        return best_value

    def _success_posterior(
        self,
        full_probability_matrix: FullProbabilityMatrix,
        target_player_id: str,
        target_slot_index: int,
        guessed_card: Card,
    ) -> FullProbabilityMatrix:
        success_matrix: FullProbabilityMatrix = {}
        for player_id, probability_matrix in full_probability_matrix.items():
            player_positions: ProbabilityMatrix = {}
            for slot_index, slot_distribution in probability_matrix.items():
                if player_id == target_player_id and slot_index == target_slot_index:
                    continue

                adjusted = dict(slot_distribution)
                if guessed_card in adjusted:
                    adjusted.pop(guessed_card, None)
                    adjusted = normalize_card_distribution(adjusted)

                if adjusted:
                    player_positions[slot_index] = adjusted
            if player_positions:
                success_matrix[player_id] = player_positions
        return success_matrix

    def _expected_slot_entropy_reduction(
        self,
        slot_distribution: Dict[Card, float],
        guessed_card: Card,
    ) -> float:
        before = self._entropy(slot_distribution.values())
        hit_probability = slot_distribution.get(guessed_card, 0.0)
        if hit_probability <= 0.0:
            return 0.0
        if hit_probability >= 1.0:
            return before

        miss_distribution = {
            card: probability
            for card, probability in slot_distribution.items()
            if card != guessed_card
        }
        miss_distribution = normalize_card_distribution(miss_distribution)
        after_miss = self._entropy(miss_distribution.values())
        return before - ((1.0 - hit_probability) * after_miss)

    def _public_self_exposure_profile(
        self,
        game_state: Optional[GameState],
    ) -> Dict[str, float]:
        if game_state is None:
            return {
                "total_exposure": 0.0,
                "max_slot_exposure": 0.0,
                "newly_drawn_exposure": 0.0,
                "average_slot_exposure": 0.0,
                "finish_fragility": 0.0,
                "hidden_count": 0.0,
            }

        self_slots = game_state.self_player().ordered_slots()
        slot_exposures: List[float] = []
        newly_drawn_exposure = 0.0
        for slot in self_slots:
            if slot.is_revealed:
                continue
            slot_exposure = self._public_self_slot_exposure(self_slots, slot.slot_index)
            slot_exposures.append(slot_exposure)
            if slot.is_newly_drawn:
                newly_drawn_exposure = max(newly_drawn_exposure, slot_exposure)

        if not slot_exposures:
            return {
                "total_exposure": 0.0,
                "max_slot_exposure": 0.0,
                "newly_drawn_exposure": 0.0,
                "average_slot_exposure": 0.0,
                "finish_fragility": 0.0,
                "hidden_count": 0.0,
            }

        slot_exposures.sort(reverse=True)
        total_exposure = slot_exposures[0]
        if len(slot_exposures) >= 2:
            total_exposure += self.SELF_EXPOSURE_SECONDARY_BLEND * slot_exposures[1]
        hidden_count = float(len(slot_exposures))
        finish_fragility = clamp(
            ((3.0 - hidden_count) / 3.0) * slot_exposures[0],
            0.0,
            1.0,
        )
        return {
            "total_exposure": total_exposure,
            "max_slot_exposure": slot_exposures[0],
            "newly_drawn_exposure": newly_drawn_exposure,
            "average_slot_exposure": sum(slot_exposures) / float(len(slot_exposures)),
            "finish_fragility": finish_fragility,
            "hidden_count": hidden_count,
        }

    def _public_self_slot_exposure(
        self,
        self_slots: Sequence[CardSlot],
        slot_index: int,
    ) -> float:
        slot_by_index = {slot.slot_index: slot for slot in self_slots}
        slot = slot_by_index.get(slot_index)
        if slot is None or slot.is_revealed:
            return 0.0

        low, high, width = self._public_self_slot_interval(self_slots, slot_index)
        exposure = (
            self.SELF_EXPOSURE_COLOR_BONUS
            if getattr(slot, "color", None) is not None
            else 1.0
        ) / max(1.0, float(width + 1))
        if low is not None and high is not None and width <= 2:
            exposure *= self.SELF_EXPOSURE_NARROW_BONUS
        if (
            getattr(slot, "color", None) is not None
            and (
                (low is None and high is not None and high <= 4)
                or (high is None and low is not None and low >= (MAX_CARD_VALUE - 4))
            )
        ):
            exposure *= self.SELF_EXPOSURE_EDGE_BONUS
        anchor_match_count = self._public_self_slot_same_color_anchor_count(
            self_slots,
            slot_index,
            getattr(slot, "color", None),
        )
        if anchor_match_count >= 2:
            exposure *= self.SELF_EXPOSURE_DOUBLE_COLOR_ANCHOR_BONUS
        elif anchor_match_count == 1:
            exposure *= self.SELF_EXPOSURE_SAME_COLOR_ANCHOR_BONUS
        if slot.is_newly_drawn:
            exposure *= self.SELF_EXPOSURE_NEW_DRAWN_BONUS
        return exposure

    def _public_self_slot_same_color_anchor_count(
        self,
        self_slots: Sequence[CardSlot],
        slot_index: int,
        slot_color: Optional[str],
    ) -> int:
        if slot_color not in CARD_COLORS:
            return 0

        ordered_slots = sorted(self_slots, key=lambda slot: slot.slot_index)
        index_by_slot = {slot.slot_index: idx for idx, slot in enumerate(ordered_slots)}
        order_index = index_by_slot.get(slot_index)
        if order_index is None:
            return 0

        same_color_anchor_count = 0
        cursor = order_index - 1
        while cursor >= 0:
            left_slot = ordered_slots[cursor]
            if left_slot.is_revealed and left_slot.color in CARD_COLORS:
                if left_slot.color == slot_color:
                    same_color_anchor_count += 1
                break
            cursor -= 1

        cursor = order_index + 1
        while cursor < len(ordered_slots):
            right_slot = ordered_slots[cursor]
            if right_slot.is_revealed and right_slot.color in CARD_COLORS:
                if right_slot.color == slot_color:
                    same_color_anchor_count += 1
                break
            cursor += 1

        return same_color_anchor_count

    def _public_self_slot_interval(
        self,
        self_slots: Sequence[CardSlot],
        slot_index: int,
    ) -> Tuple[Optional[int], Optional[int], int]:
        ordered_slots = sorted(self_slots, key=lambda slot: slot.slot_index)
        index_by_slot = {slot.slot_index: idx for idx, slot in enumerate(ordered_slots)}
        order_index = index_by_slot.get(slot_index)
        if order_index is None:
            return None, None, MAX_CARD_VALUE

        lower: Optional[int] = None
        upper: Optional[int] = None

        cursor = order_index - 1
        while cursor >= 0:
            left_slot = ordered_slots[cursor]
            if left_slot.is_revealed:
                left_value = numeric_card_value(left_slot.known_card())
                if left_value is not None:
                    lower = left_value
                    break
            cursor -= 1

        cursor = order_index + 1
        while cursor < len(ordered_slots):
            right_slot = ordered_slots[cursor]
            if right_slot.is_revealed:
                right_value = numeric_card_value(right_slot.known_card())
                if right_value is not None:
                    upper = right_value
                    break
            cursor += 1

        effective_low = 0 if lower is None else lower
        effective_high = MAX_CARD_VALUE if upper is None else upper
        if effective_high < effective_low:
            effective_high = effective_low
        width = max(0, effective_high - effective_low)
        return lower, upper, width

    def _entropy(self, probabilities: Iterable[float]) -> float:
        entropy = 0.0
        for probability in probabilities:
            if probability <= 0.0:
                continue
            entropy -= probability * log2(probability)
        return entropy


class GameController:
    """Coordinate hard inference, soft weighting, posterior blending, and move scoring."""

    BEHAVIOR_DEBUG_TOP_K = 3
    DRAW_ROLLOUT_SAMPLE_LIMIT = 4
    DRAW_ROLLOUT_EXACT_ENUMERATION_LIMIT = 8
    DRAW_ROLLOUT_ENDGAME_ENUMERATION_THRESHOLD = 2
    DRAW_ROLLOUT_VALUE_REFERENCE = 8.0
    DRAW_ROLLOUT_EDGE_WINDOW = 0.80
    DRAW_ROLLOUT_CONTINUATION_VALUE_REFERENCE = 3.0
    DRAW_ROLLOUT_ATTACKABILITY_REFERENCE = 0.40
    DRAW_ROLLOUT_BEST_GAP_REFERENCE = 1.5
    DRAW_ROLLOUT_STOP_THRESHOLD_REFERENCE = 0.8
    DRAW_ROLLOUT_CONTINUE_MARGIN_REFERENCE = 1.0

    def __init__(self, game_state: GameState):
        self.game_state = game_state
        self.inference_engine = DaVinciInferenceEngine(game_state)
        self.behavior_model = BehavioralLikelihoodModel()
        self.decision_engine = DaVinciDecisionEngine()

    def _normalized_entropy(self, probabilities: Sequence[float]) -> float:
        positive = [float(probability) for probability in probabilities if probability > 0.0]
        if len(positive) <= 1:
            return 0.0
        entropy = 0.0
        for probability in positive:
            entropy -= probability * log2(probability)
        return entropy / max(1e-9, log2(len(positive)))

    def _color_entropy_pressure(
        self,
        full_probability_matrix: Optional[FullProbabilityMatrix],
        *,
        target_player_only: bool = False,
    ) -> Dict[str, float]:
        if not full_probability_matrix:
            return {"B": 0.0, "W": 0.0}

        entropy_pressure = {"B": 0.0, "W": 0.0}
        relevant_matrices: Sequence[ProbabilityMatrix]
        if target_player_only:
            target_player_id = getattr(self.game_state, "target_player_id", None)
            relevant_matrices = [full_probability_matrix.get(target_player_id, {})]
        else:
            relevant_matrices = list(full_probability_matrix.values())

        total_positions = 0.0
        for probability_matrix in relevant_matrices:
            for slot_distribution in probability_matrix.values():
                total_positions += 1.0
                for color in CARD_COLORS:
                    color_probabilities = [
                        float(probability)
                        for card, probability in slot_distribution.items()
                        if card[0] == color and probability > 0.0
                    ]
                    color_mass = sum(color_probabilities)
                    if color_mass <= 0.0:
                        continue
                    normalized_color_probabilities = [
                        probability / color_mass
                        for probability in color_probabilities
                    ]
                    entropy_pressure[color] += (
                        color_mass * self._normalized_entropy(normalized_color_probabilities)
                    )

        if total_positions <= 0.0:
            return {"B": 0.0, "W": 0.0}
        return {
            color: entropy_pressure[color] / total_positions
            for color in CARD_COLORS
        }

    def _self_flexibility_pressure(self) -> Dict[str, float]:
        self_numeric_anchors = {
            color: sorted(
                numeric_card_value(slot.known_card())
                for slot in self.game_state.self_player().ordered_slots()
                if getattr(slot, "color", None) == color
                and numeric_card_value(slot.known_card()) is not None
            )
            for color in CARD_COLORS
        }
        available_numeric_values = {
            color: sorted(
                numeric_card_value(card)
                for card in self.inference_engine.available_cards
                if card[0] == color and numeric_card_value(card) is not None
            )
            for color in CARD_COLORS
        }
        flexibility_pressure = {"B": 0.0, "W": 0.0}
        for color in CARD_COLORS:
            anchors = self_numeric_anchors[color]
            values = available_numeric_values[color]
            if not values:
                continue
            if not anchors:
                flexibility_pressure[color] = 1.0
                continue

            slack_sum = 0.0
            for value in values:
                left_anchor = -1
                right_anchor = MAX_CARD_VALUE + 1
                for anchor in anchors:
                    if anchor < value:
                        left_anchor = anchor
                    elif anchor > value:
                        right_anchor = anchor
                        break
                slack_sum += max(0.0, (right_anchor - left_anchor - 1) / (MAX_CARD_VALUE + 1))
                flexibility_pressure[color] = slack_sum / len(values)
        return flexibility_pressure

    def _hidden_defense_pressure(self) -> Dict[str, float]:
        hidden_counts = {
            color: sum(
                1
                for slot in self.game_state.resolved_ordered_slots(self.game_state.self_player_id)
                if not slot.is_revealed and getattr(slot, "color", None) == color
            )
            for color in CARD_COLORS
        }
        total_hidden = max(1, sum(hidden_counts.values()))
        return {
            "B": (hidden_counts["W"] - hidden_counts["B"]) / total_hidden,
            "W": (hidden_counts["B"] - hidden_counts["W"]) / total_hidden,
        }

    def _target_boundary_pressure(
        self,
        full_probability_matrix: Optional[FullProbabilityMatrix],
    ) -> Dict[str, float]:
        if not full_probability_matrix:
            return {"B": 0.0, "W": 0.0}
        target_player_id = getattr(self.game_state, "target_player_id", None)
        if target_player_id is None:
            return {"B": 0.0, "W": 0.0}
        target_probability_matrix = full_probability_matrix.get(target_player_id, {})
        boundary_pressure = {"B": 0.0, "W": 0.0}
        observation_counts = {"B": 0.0, "W": 0.0}
        for slot_distribution in target_probability_matrix.values():
            for color in CARD_COLORS:
                color_candidates = [
                    card
                    for card, probability in slot_distribution.items()
                    if card[0] == color and probability > 0.0
                ]
                if not color_candidates:
                    continue
                color_mass = sum(
                    float(probability)
                    for card, probability in slot_distribution.items()
                    if card[0] == color and probability > 0.0
                )
                boundary_pressure[color] += color_mass / len(color_candidates)
                observation_counts[color] += 1.0
        return {
            color: (
                boundary_pressure[color] / observation_counts[color]
                if observation_counts[color] > 0.0
                else 0.0
            )
            for color in CARD_COLORS
        }

    def _target_finish_pressure(
        self,
        full_probability_matrix: Optional[FullProbabilityMatrix],
    ) -> Dict[str, float]:
        if not full_probability_matrix:
            return {"B": 0.0, "W": 0.0}
        target_player_id = getattr(self.game_state, "target_player_id", None)
        if target_player_id is None:
            return {"B": 0.0, "W": 0.0}
        target_probability_matrix = full_probability_matrix.get(target_player_id, {})
        if not target_probability_matrix:
            return {"B": 0.0, "W": 0.0}

        hidden_slot_count = max(1, len(target_probability_matrix))
        finish_scale = clamp(2.0 / hidden_slot_count, 0.75, 2.0)
        finish_pressure = {"B": 0.0, "W": 0.0}
        for slot_distribution in target_probability_matrix.values():
            for color in CARD_COLORS:
                color_peak = max(
                    (
                        float(probability)
                        for card, probability in slot_distribution.items()
                        if card[0] == color and probability > 0.0
                    ),
                    default=0.0,
                )
                finish_pressure[color] += color_peak
        return {
            color: (finish_pressure[color] / hidden_slot_count) * finish_scale
            for color in CARD_COLORS
        }

    def _target_focus_pressure(
        self,
        full_probability_matrix: Optional[FullProbabilityMatrix],
    ) -> Dict[str, float]:
        if not full_probability_matrix:
            return {"B": 0.0, "W": 0.0}
        target_player_id = getattr(self.game_state, "target_player_id", None)
        if target_player_id is None:
            return {"B": 0.0, "W": 0.0}

        total_hidden_color_mass = {"B": 0.0, "W": 0.0}
        for probability_matrix in full_probability_matrix.values():
            for slot_distribution in probability_matrix.values():
                for card, probability in slot_distribution.items():
                    total_hidden_color_mass[card[0]] += float(probability)

        target_hidden_color_mass = {"B": 0.0, "W": 0.0}
        for slot_distribution in full_probability_matrix.get(target_player_id, {}).values():
            for card, probability in slot_distribution.items():
                target_hidden_color_mass[card[0]] += float(probability)

        return {
            color: (
                target_hidden_color_mass[color] / total_hidden_color_mass[color]
                if total_hidden_color_mass[color] > 0.0
                else 0.0
            )
            for color in CARD_COLORS
        }

    def _target_recent_momentum_pressure(self) -> Dict[str, float]:
        target_player_id = getattr(self.game_state, "target_player_id", None)
        if target_player_id is None:
            return {"B": 0.0, "W": 0.0}

        target_actions = [
            action
            for action in getattr(self.game_state, "actions", ())
            if getattr(action, "action_type", None) == "guess"
            and getattr(action, "target_player_id", None) == target_player_id
            and getattr(action, "guessed_color", None) in CARD_COLORS
        ]
        if not target_actions:
            return {"B": 0.0, "W": 0.0}

        momentum = {"B": 0.0, "W": 0.0}
        total_recency = 0.0
        for reverse_index, action in enumerate(reversed(target_actions)):
            color = getattr(action, "guessed_color", None)
            if color not in CARD_COLORS:
                continue
            recency_weight = 1.0 / (reverse_index + 1.0)
            signed_outcome = 1.0 if getattr(action, "result", False) else -0.55
            if getattr(action, "result", False) and getattr(action, "continued_turn", False):
                signed_outcome *= 1.15
            momentum[color] += recency_weight * signed_outcome
            total_recency += recency_weight

        if total_recency <= 0.0:
            return {"B": 0.0, "W": 0.0}
        return {
            color: momentum[color] / total_recency
            for color in CARD_COLORS
        }

    def _recent_self_exposure_pressure(self) -> Dict[str, float]:
        self_player_id = getattr(self.game_state, "self_player_id", None)
        if self_player_id is None:
            return {"B": 0.0, "W": 0.0}

        exposure = {"B": 0.0, "W": 0.0}
        total_recency = 0.0
        relevant_actions = [
            action
            for action in getattr(self.game_state, "actions", ())
            if getattr(action, "revealed_player_id", None) == self_player_id
            and getattr(action, "revealed_color", None) in CARD_COLORS
        ]
        for reverse_index, action in enumerate(reversed(relevant_actions)):
            color = getattr(action, "revealed_color", None)
            if color not in CARD_COLORS:
                continue
            recency_weight = 1.0 / (reverse_index + 1.0)
            exposure[color] += recency_weight
            total_recency += recency_weight

        if total_recency <= 0.0:
            return {"B": 0.0, "W": 0.0}
        return {
            "B": (exposure["W"] - exposure["B"]) / total_recency,
            "W": (exposure["B"] - exposure["W"]) / total_recency,
        }

    def _target_attack_window_factor(self) -> float:
        target_hidden_count = len(self.game_state.target_hidden_slots())
        my_hidden_count = self.game_state.my_hidden_count()
        finish_bonus = clamp((2 - target_hidden_count) * 0.18, 0.0, 0.36)
        safety_gap = clamp((my_hidden_count - target_hidden_count) * 0.08, -0.18, 0.24)
        return clamp(1.0 + finish_bonus + safety_gap, 0.78, 1.55)

    def _draw_rollout_phase_gate(self) -> float:
        target_hidden_count = len(self.game_state.target_hidden_slots())
        my_hidden_count = self.game_state.my_hidden_count()
        late_target_bonus = clamp((2 - target_hidden_count) * 0.16, 0.0, 0.32)
        safety_bonus = clamp((my_hidden_count - target_hidden_count) * 0.08, -0.18, 0.24)
        return clamp(0.62 + late_target_bonus + safety_bonus, 0.32, 1.0)

    def _draw_rollout_plan_gate(
        self,
        draw_rollout: Dict[str, Any],
    ) -> float:
        active_opening_signal = max(
            draw_rollout["active_opening_ratio"]["B"],
            draw_rollout["active_opening_ratio"]["W"],
        )
        opening_plan_signal = max(
            draw_rollout["opening_plan"]["B"]["support_ratio"],
            draw_rollout["opening_plan"]["W"]["support_ratio"],
        )
        information_signal = max(
            abs(draw_rollout["information_gain_pressure"]["B"]),
            abs(draw_rollout["information_gain_pressure"]["W"]),
        )
        return clamp(
            (0.50 * active_opening_signal)
            + (0.35 * opening_plan_signal)
            + (0.15 * information_signal),
            0.18,
            1.0,
        )

    def _representative_draw_cards(self, color: str) -> List[Card]:
        cards = sorted(
            (card for card in self.inference_engine.available_cards if card[0] == color),
            key=card_sort_key,
        )
        exact_endgame = (
            self.game_state.my_hidden_count()
            <= self.DRAW_ROLLOUT_ENDGAME_ENUMERATION_THRESHOLD
            or len(self.game_state.target_hidden_slots())
            <= self.DRAW_ROLLOUT_ENDGAME_ENUMERATION_THRESHOLD
        )
        if (
            exact_endgame
            or len(cards) <= self.DRAW_ROLLOUT_SAMPLE_LIMIT
            or len(cards) <= self.DRAW_ROLLOUT_EXACT_ENUMERATION_LIMIT
        ):
            return cards

        sampled_indices = {
            0,
            len(cards) // 3,
            (2 * len(cards)) // 3,
            len(cards) - 1,
        }
        return [cards[index] for index in sorted(sampled_indices)]

    def _simulated_draw_game_state(self, drawn_card: Card) -> GameState:
        self_player = self.game_state.self_player()
        simulated_self_slots = [
            replace(slot, is_newly_drawn=False) for slot in self_player.ordered_slots()
        ]
        simulated_self_slots.append(
            CardSlot(
                slot_index=-1,
                color=drawn_card[0],
                value=drawn_card[1],
                is_revealed=False,
                is_newly_drawn=True,
            )
        )
        simulated_self_slots = [
            replace(slot, slot_index=slot_index)
            for slot_index, slot in enumerate(
                sorted(
                    simulated_self_slots,
                    key=lambda slot: card_sort_key(
                        slot.known_card()
                        if slot.known_card() is not None
                        else (
                            slot.color if slot.color in CARD_COLORS else "B",
                            slot.value if slot.value is not None else JOKER,
                        )
                    ),
                )
            )
        ]
        simulated_players = dict(self.game_state.players)
        simulated_players[self.game_state.self_player_id] = PlayerState(
            player_id=self.game_state.self_player_id,
            slots=simulated_self_slots,
        )
        return GameState(
            self_player_id=self.game_state.self_player_id,
            target_player_id=self.game_state.target_player_id,
            players=simulated_players,
            actions=list(self.game_state.actions),
        )

    def _draw_rollout_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "expected_best_value": {"B": 0.0, "W": 0.0},
            "expected_immediate_value": {"B": 0.0, "W": 0.0},
            "expected_continuation_value": {"B": 0.0, "W": 0.0},
            "expected_continuation_likelihood": {"B": 0.0, "W": 0.0},
            "expected_self_public_exposure": {"B": 0.0, "W": 0.0},
            "expected_self_newly_drawn_exposure": {"B": 0.0, "W": 0.0},
            "expected_self_finish_fragility": {"B": 0.0, "W": 0.0},
            "expected_stop_threshold": {"B": 0.0, "W": 0.0},
            "expected_continue_margin": {"B": 0.0, "W": 0.0},
            "expected_failure_collapse_bonus": {"B": 0.0, "W": 0.0},
            "expected_failed_guess_switch_bonus": {"B": 0.0, "W": 0.0},
            "expected_failed_guess_switch_signal": {"B": 0.0, "W": 0.0},
            "expected_post_hit_failure_recovery_bonus": {"B": 0.0, "W": 0.0},
            "expected_post_hit_failed_switch_bonus": {"B": 0.0, "W": 0.0},
            "expected_target_attack_window_signal": {"B": 0.0, "W": 0.0},
            "expected_target_attack_window_bonus": {"B": 0.0, "W": 0.0},
            "expected_target_attack_window_continuation_bonus": {"B": 0.0, "W": 0.0},
            "expected_joint_collapse_signal": {"B": 0.0, "W": 0.0},
            "expected_joint_collapse_bonus": {"B": 0.0, "W": 0.0},
            "expected_joint_collapse_continuation_bonus": {"B": 0.0, "W": 0.0},
            "expected_public_reveal_bridge_signal": {"B": 0.0, "W": 0.0},
            "expected_public_reveal_bridge_bonus": {"B": 0.0, "W": 0.0},
            "expected_public_reveal_bridge_continuation_bonus": {"B": 0.0, "W": 0.0},
            "expected_target_chain_signal": {"B": 0.0, "W": 0.0},
            "expected_target_chain_bonus": {"B": 0.0, "W": 0.0},
            "expected_target_chain_continuation_bonus": {"B": 0.0, "W": 0.0},
            "expected_target_finish_chain_signal": {"B": 0.0, "W": 0.0},
            "expected_target_finish_chain_bonus": {"B": 0.0, "W": 0.0},
            "expected_target_finish_chain_continuation_bonus": {"B": 0.0, "W": 0.0},
            "expected_win_probability": {"B": 0.0, "W": 0.0},
            "expected_attackability_after_hit": {"B": 0.0, "W": 0.0},
            "target_retention_ratio": {"B": 0.0, "W": 0.0},
            "color_alignment_ratio": {"B": 0.0, "W": 0.0},
            "expected_best_gap": {"B": 0.0, "W": 0.0},
            "expected_strategy_objective": {"B": 0.0, "W": 0.0},
            "active_opening_ratio": {"B": 0.0, "W": 0.0},
            "best_value_stddev": {"B": 0.0, "W": 0.0},
            "win_probability_stddev": {"B": 0.0, "W": 0.0},
            "best_value_floor": {"B": 0.0, "W": 0.0},
            "win_probability_floor": {"B": 0.0, "W": 0.0},
            "expected_information_gain": {"B": 0.0, "W": 0.0},
            "information_gain_floor": {"B": 0.0, "W": 0.0},
            "opening_plan": {
                "B": {
                    "target_player_id": None,
                    "target_slot_index": None,
                    "guess_card": None,
                    "support_ratio": 0.0,
                    "guess_support_ratio": 0.0,
                    "expected_value": 0.0,
                    "strategy_objective": 0.0,
                    "win_probability": 0.0,
                    "information_gain": 0.0,
                    "continuation_likelihood": 0.0,
                },
                "W": {
                    "target_player_id": None,
                    "target_slot_index": None,
                    "guess_card": None,
                    "support_ratio": 0.0,
                    "guess_support_ratio": 0.0,
                    "expected_value": 0.0,
                    "strategy_objective": 0.0,
                    "win_probability": 0.0,
                    "information_gain": 0.0,
                    "continuation_likelihood": 0.0,
                },
            },
            "sample_count": {"B": 0.0, "W": 0.0},
        }
        target_player_id = getattr(self.game_state, "target_player_id", None)

        for color in CARD_COLORS:
            sample_cards = self._representative_draw_cards(color)
            summary["sample_count"][color] = float(len(sample_cards))
            if not sample_cards:
                continue

            best_value_sum = 0.0
            immediate_value_sum = 0.0
            continuation_value_sum = 0.0
            continuation_likelihood_sum = 0.0
            self_public_exposure_sum = 0.0
            self_newly_drawn_exposure_sum = 0.0
            self_finish_fragility_sum = 0.0
            stop_threshold_sum = 0.0
            continue_margin_sum = 0.0
            failure_collapse_bonus_sum = 0.0
            failed_guess_switch_bonus_sum = 0.0
            failed_guess_switch_signal_sum = 0.0
            post_hit_failure_recovery_bonus_sum = 0.0
            post_hit_failed_switch_bonus_sum = 0.0
            target_attack_window_signal_sum = 0.0
            target_attack_window_bonus_sum = 0.0
            target_attack_window_continuation_bonus_sum = 0.0
            joint_collapse_signal_sum = 0.0
            joint_collapse_bonus_sum = 0.0
            joint_collapse_continuation_bonus_sum = 0.0
            public_reveal_bridge_signal_sum = 0.0
            public_reveal_bridge_bonus_sum = 0.0
            public_reveal_bridge_continuation_bonus_sum = 0.0
            target_chain_signal_sum = 0.0
            target_chain_bonus_sum = 0.0
            target_chain_continuation_bonus_sum = 0.0
            target_finish_chain_signal_sum = 0.0
            target_finish_chain_bonus_sum = 0.0
            target_finish_chain_continuation_bonus_sum = 0.0
            win_probability_sum = 0.0
            attackability_sum = 0.0
            strategy_objective_sum = 0.0
            target_retention_count = 0.0
            color_alignment_count = 0.0
            best_gap_sum = 0.0
            active_opening_count = 0.0
            sampled_best_values: List[float] = []
            sampled_win_probabilities: List[float] = []
            sampled_information_gains: List[float] = []
            opening_plan_stats: Dict[Tuple[str, int], Dict[str, Any]] = {}
            for drawn_card in sample_cards:
                simulated_state = self._simulated_draw_game_state(drawn_card)
                simulated_result = GameController(simulated_state).run_turn(
                    include_draw_color_summary=False,
                )
                simulated_decision = simulated_result.get("decision_summary", {})
                simulated_best_move = simulated_result.get("best_move")
                sampled_best_value = float(
                    simulated_decision.get("best_expected_value", 0.0)
                )
                sampled_win_probability = float(
                    simulated_decision.get("best_win_probability", 0.0)
                )
                sampled_information_gain = float(
                    simulated_decision.get("best_information_gain", 0.0)
                )
                sampled_strategy_objective = float(
                    simulated_decision.get("best_strategy_objective", 0.0)
                )
                best_value_sum += sampled_best_value
                immediate_value_sum += float(
                    simulated_decision.get("best_immediate_value", 0.0)
                )
                continuation_value_sum += float(
                    simulated_decision.get("best_continuation_value", 0.0)
                )
                continuation_likelihood_sum += float(
                    simulated_decision.get("best_continuation_likelihood", 0.0)
                )
                self_public_exposure_sum += float(
                    simulated_decision.get("best_self_public_exposure", 0.0)
                )
                self_newly_drawn_exposure_sum += float(
                    simulated_decision.get("best_self_newly_drawn_exposure", 0.0)
                )
                self_finish_fragility_sum += float(
                    simulated_decision.get("best_self_finish_fragility", 0.0)
                )
                stop_threshold_sum += float(
                    simulated_decision.get("stop_threshold", 0.0)
                )
                continue_margin_sum += float(
                    simulated_decision.get("continue_margin", 0.0)
                )
                failure_collapse_bonus_sum += float(
                    simulated_decision.get("best_failure_collapse_bonus", 0.0)
                )
                failed_guess_switch_bonus_sum += float(
                    simulated_decision.get("best_failed_guess_switch_bonus", 0.0)
                )
                failed_guess_switch_signal_sum += float(
                    simulated_decision.get(
                        "best_failed_guess_switch_continuity_signal",
                        0.0,
                    )
                )
                post_hit_failure_recovery_bonus_sum += float(
                    simulated_decision.get("best_post_hit_failure_recovery_bonus", 0.0)
                )
                post_hit_failed_switch_bonus_sum += float(
                    simulated_decision.get("best_post_hit_failed_switch_bonus", 0.0)
                )
                target_attack_window_signal_sum += float(
                    simulated_decision.get("best_target_attack_window_signal", 0.0)
                )
                target_attack_window_bonus_sum += float(
                    simulated_decision.get("best_target_attack_window_bonus", 0.0)
                )
                target_attack_window_continuation_bonus_sum += float(
                    simulated_decision.get(
                        "best_target_attack_window_continuation_bonus",
                        0.0,
                    )
                )
                joint_collapse_signal_sum += float(
                    simulated_decision.get("best_joint_collapse_signal", 0.0)
                )
                joint_collapse_bonus_sum += float(
                    simulated_decision.get("best_joint_collapse_bonus", 0.0)
                )
                joint_collapse_continuation_bonus_sum += float(
                    simulated_decision.get(
                        "best_joint_collapse_continuation_bonus",
                        0.0,
                    )
                )
                public_reveal_bridge_signal_sum += float(
                    simulated_decision.get("best_public_reveal_bridge_signal", 0.0)
                )
                public_reveal_bridge_bonus_sum += float(
                    simulated_decision.get("best_public_reveal_bridge_bonus", 0.0)
                )
                public_reveal_bridge_continuation_bonus_sum += float(
                    simulated_decision.get(
                        "best_public_reveal_bridge_continuation_bonus",
                        0.0,
                    )
                )
                target_chain_signal_sum += float(
                    simulated_decision.get("best_target_chain_signal", 0.0)
                )
                target_chain_bonus_sum += float(
                    simulated_decision.get("best_target_chain_bonus", 0.0)
                )
                target_chain_continuation_bonus_sum += float(
                    simulated_decision.get(
                        "best_target_chain_continuation_bonus",
                        0.0,
                    )
                )
                target_finish_chain_signal_sum += float(
                    simulated_decision.get("best_target_finish_chain_signal", 0.0)
                )
                target_finish_chain_bonus_sum += float(
                    simulated_decision.get("best_target_finish_chain_bonus", 0.0)
                )
                target_finish_chain_continuation_bonus_sum += float(
                    simulated_decision.get(
                        "best_target_finish_chain_continuation_bonus",
                        0.0,
                    )
                )
                win_probability_sum += sampled_win_probability
                attackability_sum += float(
                    simulated_decision.get("best_attackability_after_hit", 0.0)
                )
                strategy_objective_sum += sampled_strategy_objective
                best_gap_sum += float(simulated_decision.get("best_gap", 0.0))
                sampled_best_values.append(sampled_best_value)
                sampled_win_probabilities.append(sampled_win_probability)
                sampled_information_gains.append(sampled_information_gain)
                if isinstance(simulated_best_move, dict):
                    active_opening_count += 1.0
                    simulated_target_player_id = simulated_best_move.get(
                        "target_player_id"
                    )
                    simulated_target_slot_index = simulated_best_move.get(
                        "target_slot_index"
                    )
                    if (
                        isinstance(simulated_target_player_id, str)
                        and isinstance(simulated_target_slot_index, int)
                    ):
                        target_key = (
                            simulated_target_player_id,
                            simulated_target_slot_index,
                        )
                        target_stats = opening_plan_stats.setdefault(
                            target_key,
                            {
                                "count": 0.0,
                                "expected_value_sum": 0.0,
                                "strategy_objective_sum": 0.0,
                                "win_probability_sum": 0.0,
                                "information_gain_sum": 0.0,
                                "continuation_likelihood_sum": 0.0,
                                "guess_counts": defaultdict(float),
                            },
                        )
                        target_stats["count"] += 1.0
                        target_stats["expected_value_sum"] += sampled_best_value
                        target_stats["strategy_objective_sum"] += sampled_strategy_objective
                        target_stats["win_probability_sum"] += sampled_win_probability
                        target_stats["information_gain_sum"] += sampled_information_gain
                        target_stats["continuation_likelihood_sum"] += float(
                            simulated_decision.get("best_continuation_likelihood", 0.0)
                        )
                    guessed_card = simulated_best_move.get("guess_card")
                    normalized_guessed_card = (
                        (guessed_card[0], guessed_card[1])
                        if isinstance(guessed_card, (list, tuple))
                        and len(guessed_card) == 2
                        else None
                    )
                    if (
                        isinstance(simulated_target_player_id, str)
                        and isinstance(simulated_target_slot_index, int)
                        and normalized_guessed_card is not None
                    ):
                        opening_plan_stats[
                            (
                                simulated_target_player_id,
                                simulated_target_slot_index,
                            )
                        ]["guess_counts"][normalized_guessed_card] += 1.0
                    if (
                        target_player_id is not None
                        and simulated_best_move.get("target_player_id") == target_player_id
                    ):
                        target_retention_count += 1.0
                    if (
                        normalized_guessed_card is not None
                        and normalized_guessed_card[0] == color
                    ):
                        color_alignment_count += 1.0

            summary["expected_best_value"][color] = best_value_sum / len(sample_cards)
            summary["expected_immediate_value"][color] = (
                immediate_value_sum / len(sample_cards)
            )
            summary["expected_continuation_value"][color] = (
                continuation_value_sum / len(sample_cards)
            )
            summary["expected_continuation_likelihood"][color] = (
                continuation_likelihood_sum / len(sample_cards)
            )
            summary["expected_self_public_exposure"][color] = (
                self_public_exposure_sum / len(sample_cards)
            )
            summary["expected_self_newly_drawn_exposure"][color] = (
                self_newly_drawn_exposure_sum / len(sample_cards)
            )
            summary["expected_self_finish_fragility"][color] = (
                self_finish_fragility_sum / len(sample_cards)
            )
            summary["expected_stop_threshold"][color] = (
                stop_threshold_sum / len(sample_cards)
            )
            summary["expected_continue_margin"][color] = (
                continue_margin_sum / len(sample_cards)
            )
            summary["expected_failure_collapse_bonus"][color] = (
                failure_collapse_bonus_sum / len(sample_cards)
            )
            summary["expected_failed_guess_switch_bonus"][color] = (
                failed_guess_switch_bonus_sum / len(sample_cards)
            )
            summary["expected_failed_guess_switch_signal"][color] = (
                failed_guess_switch_signal_sum / len(sample_cards)
            )
            summary["expected_post_hit_failure_recovery_bonus"][color] = (
                post_hit_failure_recovery_bonus_sum / len(sample_cards)
            )
            summary["expected_post_hit_failed_switch_bonus"][color] = (
                post_hit_failed_switch_bonus_sum / len(sample_cards)
            )
            summary["expected_target_attack_window_signal"][color] = (
                target_attack_window_signal_sum / len(sample_cards)
            )
            summary["expected_target_attack_window_bonus"][color] = (
                target_attack_window_bonus_sum / len(sample_cards)
            )
            summary["expected_target_attack_window_continuation_bonus"][color] = (
                target_attack_window_continuation_bonus_sum / len(sample_cards)
            )
            summary["expected_joint_collapse_signal"][color] = (
                joint_collapse_signal_sum / len(sample_cards)
            )
            summary["expected_joint_collapse_bonus"][color] = (
                joint_collapse_bonus_sum / len(sample_cards)
            )
            summary["expected_joint_collapse_continuation_bonus"][color] = (
                joint_collapse_continuation_bonus_sum / len(sample_cards)
            )
            summary["expected_public_reveal_bridge_signal"][color] = (
                public_reveal_bridge_signal_sum / len(sample_cards)
            )
            summary["expected_public_reveal_bridge_bonus"][color] = (
                public_reveal_bridge_bonus_sum / len(sample_cards)
            )
            summary["expected_public_reveal_bridge_continuation_bonus"][color] = (
                public_reveal_bridge_continuation_bonus_sum / len(sample_cards)
            )
            summary["expected_target_chain_signal"][color] = (
                target_chain_signal_sum / len(sample_cards)
            )
            summary["expected_target_chain_bonus"][color] = (
                target_chain_bonus_sum / len(sample_cards)
            )
            summary["expected_target_chain_continuation_bonus"][color] = (
                target_chain_continuation_bonus_sum / len(sample_cards)
            )
            summary["expected_target_finish_chain_signal"][color] = (
                target_finish_chain_signal_sum / len(sample_cards)
            )
            summary["expected_target_finish_chain_bonus"][color] = (
                target_finish_chain_bonus_sum / len(sample_cards)
            )
            summary["expected_target_finish_chain_continuation_bonus"][color] = (
                target_finish_chain_continuation_bonus_sum / len(sample_cards)
            )
            summary["expected_win_probability"][color] = (
                win_probability_sum / len(sample_cards)
            )
            summary["expected_attackability_after_hit"][color] = (
                attackability_sum / len(sample_cards)
            )
            summary["target_retention_ratio"][color] = (
                target_retention_count / len(sample_cards)
            )
            summary["color_alignment_ratio"][color] = (
                color_alignment_count / len(sample_cards)
            )
            summary["expected_best_gap"][color] = best_gap_sum / len(sample_cards)
            summary["expected_strategy_objective"][color] = (
                strategy_objective_sum / len(sample_cards)
            )
            summary["active_opening_ratio"][color] = (
                active_opening_count / len(sample_cards)
            )
            best_value_mean = summary["expected_best_value"][color]
            win_probability_mean = summary["expected_win_probability"][color]
            summary["best_value_stddev"][color] = sqrt(
                sum(
                    (value - best_value_mean) ** 2
                    for value in sampled_best_values
                ) / len(sampled_best_values)
            )
            summary["win_probability_stddev"][color] = sqrt(
                sum(
                    (probability - win_probability_mean) ** 2
                    for probability in sampled_win_probabilities
                ) / len(sampled_win_probabilities)
            )
            floor_count = max(1, len(sampled_best_values) // 2)
            summary["best_value_floor"][color] = sum(
                sorted(sampled_best_values)[:floor_count]
            ) / floor_count
            summary["win_probability_floor"][color] = sum(
                sorted(sampled_win_probabilities)[:floor_count]
            ) / floor_count
            summary["expected_information_gain"][color] = sum(
                sampled_information_gains
            ) / len(sampled_information_gains)
            summary["information_gain_floor"][color] = sum(
                sorted(sampled_information_gains)[:floor_count]
            ) / floor_count
            if opening_plan_stats:
                dominant_target_key, dominant_target_stats = max(
                    opening_plan_stats.items(),
                    key=lambda item: (
                        item[1]["count"],
                        (
                            item[1]["expected_value_sum"] / item[1]["count"]
                            if item[1]["count"] > 0.0
                            else 0.0
                        ),
                        (
                            item[1]["win_probability_sum"] / item[1]["count"]
                            if item[1]["count"] > 0.0
                            else 0.0
                        ),
                        -item[0][1],
                        item[0][0],
                    ),
                )
                dominant_guess_card = None
                dominant_guess_count = 0.0
                if dominant_target_stats["guess_counts"]:
                    dominant_guess_card, dominant_guess_count = max(
                        dominant_target_stats["guess_counts"].items(),
                        key=lambda item: (
                            item[1],
                            tuple(-part for part in card_sort_key(item[0])),
                        ),
                    )
                dominant_target_count = max(1.0, dominant_target_stats["count"])
                summary["opening_plan"][color] = {
                    "target_player_id": dominant_target_key[0],
                    "target_slot_index": dominant_target_key[1],
                    "guess_card": dominant_guess_card,
                    "support_ratio": dominant_target_stats["count"] / len(sample_cards),
                    "guess_support_ratio": dominant_guess_count / dominant_target_count,
                    "expected_value": (
                        dominant_target_stats["expected_value_sum"]
                        / dominant_target_count
                    ),
                    "strategy_objective": (
                        dominant_target_stats["strategy_objective_sum"]
                        / dominant_target_count
                    ),
                    "win_probability": (
                        dominant_target_stats["win_probability_sum"]
                        / dominant_target_count
                    ),
                    "information_gain": (
                        dominant_target_stats["information_gain_sum"]
                        / dominant_target_count
                    ),
                    "continuation_likelihood": (
                        dominant_target_stats["continuation_likelihood_sum"]
                        / dominant_target_count
                    ),
                }

        best_value_gap = (
            summary["expected_best_value"]["B"] - summary["expected_best_value"]["W"]
        )
        immediate_value_gap = (
            summary["expected_immediate_value"]["B"]
            - summary["expected_immediate_value"]["W"]
        )
        continuation_value_gap = (
            summary["expected_continuation_value"]["B"]
            - summary["expected_continuation_value"]["W"]
        )
        continuation_likelihood_gap = (
            summary["expected_continuation_likelihood"]["B"]
            - summary["expected_continuation_likelihood"]["W"]
        )
        win_probability_gap = (
            summary["expected_win_probability"]["B"]
            - summary["expected_win_probability"]["W"]
        )
        target_attack_window_signal_gap = (
            summary["expected_target_attack_window_signal"]["B"]
            - summary["expected_target_attack_window_signal"]["W"]
        )
        target_attack_window_bonus_gap = (
            summary["expected_target_attack_window_bonus"]["B"]
            - summary["expected_target_attack_window_bonus"]["W"]
        )
        target_attack_window_continuation_bonus_gap = (
            summary["expected_target_attack_window_continuation_bonus"]["B"]
            - summary["expected_target_attack_window_continuation_bonus"]["W"]
        )
        joint_collapse_signal_gap = (
            summary["expected_joint_collapse_signal"]["B"]
            - summary["expected_joint_collapse_signal"]["W"]
        )
        joint_collapse_bonus_gap = (
            summary["expected_joint_collapse_bonus"]["B"]
            - summary["expected_joint_collapse_bonus"]["W"]
        )
        joint_collapse_continuation_bonus_gap = (
            summary["expected_joint_collapse_continuation_bonus"]["B"]
            - summary["expected_joint_collapse_continuation_bonus"]["W"]
        )
        public_reveal_bridge_signal_gap = (
            summary["expected_public_reveal_bridge_signal"]["B"]
            - summary["expected_public_reveal_bridge_signal"]["W"]
        )
        public_reveal_bridge_bonus_gap = (
            summary["expected_public_reveal_bridge_bonus"]["B"]
            - summary["expected_public_reveal_bridge_bonus"]["W"]
        )
        public_reveal_bridge_continuation_bonus_gap = (
            summary["expected_public_reveal_bridge_continuation_bonus"]["B"]
            - summary["expected_public_reveal_bridge_continuation_bonus"]["W"]
        )
        target_chain_signal_gap = (
            summary["expected_target_chain_signal"]["B"]
            - summary["expected_target_chain_signal"]["W"]
        )
        target_chain_bonus_gap = (
            summary["expected_target_chain_bonus"]["B"]
            - summary["expected_target_chain_bonus"]["W"]
        )
        target_chain_continuation_bonus_gap = (
            summary["expected_target_chain_continuation_bonus"]["B"]
            - summary["expected_target_chain_continuation_bonus"]["W"]
        )
        target_finish_chain_signal_gap = (
            summary["expected_target_finish_chain_signal"]["B"]
            - summary["expected_target_finish_chain_signal"]["W"]
        )
        target_finish_chain_bonus_gap = (
            summary["expected_target_finish_chain_bonus"]["B"]
            - summary["expected_target_finish_chain_bonus"]["W"]
        )
        target_finish_chain_continuation_bonus_gap = (
            summary["expected_target_finish_chain_continuation_bonus"]["B"]
            - summary["expected_target_finish_chain_continuation_bonus"]["W"]
        )
        attackability_gap = (
            summary["expected_attackability_after_hit"]["B"]
            - summary["expected_attackability_after_hit"]["W"]
        )
        target_retention_gap = (
            summary["target_retention_ratio"]["B"]
            - summary["target_retention_ratio"]["W"]
        )
        color_alignment_gap = (
            summary["color_alignment_ratio"]["B"]
            - summary["color_alignment_ratio"]["W"]
        )
        best_gap_gap = (
            summary["expected_best_gap"]["B"]
            - summary["expected_best_gap"]["W"]
        )
        strategy_objective_gap = (
            summary["expected_strategy_objective"]["B"]
            - summary["expected_strategy_objective"]["W"]
        )
        active_opening_gap = (
            summary["active_opening_ratio"]["B"]
            - summary["active_opening_ratio"]["W"]
        )
        best_value_stddev_gap = (
            summary["best_value_stddev"]["W"]
            - summary["best_value_stddev"]["B"]
        )
        win_probability_stddev_gap = (
            summary["win_probability_stddev"]["W"]
            - summary["win_probability_stddev"]["B"]
        )
        best_value_floor_gap = (
            summary["best_value_floor"]["B"]
            - summary["best_value_floor"]["W"]
        )
        win_probability_floor_gap = (
            summary["win_probability_floor"]["B"]
            - summary["win_probability_floor"]["W"]
        )
        information_gain_gap = (
            summary["expected_information_gain"]["B"]
            - summary["expected_information_gain"]["W"]
        )
        information_gain_floor_gap = (
            summary["information_gain_floor"]["B"]
            - summary["information_gain_floor"]["W"]
        )
        summary["value_pressure"] = {
            "B": clamp(
                best_value_gap / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
            "W": clamp(
                (-best_value_gap) / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
        }
        summary["immediate_value_pressure"] = {
            "B": clamp(
                immediate_value_gap / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
            "W": clamp(
                (-immediate_value_gap) / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
        }
        summary["continuation_value_pressure"] = {
            "B": clamp(
                continuation_value_gap / self.DRAW_ROLLOUT_CONTINUATION_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
            "W": clamp(
                (-continuation_value_gap) / self.DRAW_ROLLOUT_CONTINUATION_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
        }
        summary["continuation_likelihood_pressure"] = {
            "B": clamp(continuation_likelihood_gap, -1.0, 1.0),
            "W": clamp(-continuation_likelihood_gap, -1.0, 1.0),
        }
        summary["win_probability_pressure"] = {
            "B": clamp(win_probability_gap, -1.0, 1.0),
            "W": clamp(-win_probability_gap, -1.0, 1.0),
        }
        summary["target_attack_window_signal_pressure"] = {
            "B": clamp(target_attack_window_signal_gap, -1.0, 1.0),
            "W": clamp(-target_attack_window_signal_gap, -1.0, 1.0),
        }
        summary["target_attack_window_bonus_pressure"] = {
            "B": clamp(
                target_attack_window_bonus_gap / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
            "W": clamp(
                (-target_attack_window_bonus_gap) / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
        }
        summary["target_attack_window_continuation_pressure"] = {
            "B": clamp(
                target_attack_window_continuation_bonus_gap
                / self.DRAW_ROLLOUT_CONTINUATION_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
            "W": clamp(
                (-target_attack_window_continuation_bonus_gap)
                / self.DRAW_ROLLOUT_CONTINUATION_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
        }
        summary["joint_collapse_signal_pressure"] = {
            "B": clamp(joint_collapse_signal_gap, -1.0, 1.0),
            "W": clamp(-joint_collapse_signal_gap, -1.0, 1.0),
        }
        summary["joint_collapse_bonus_pressure"] = {
            "B": clamp(
                joint_collapse_bonus_gap / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
            "W": clamp(
                (-joint_collapse_bonus_gap) / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
        }
        summary["joint_collapse_continuation_pressure"] = {
            "B": clamp(
                joint_collapse_continuation_bonus_gap
                / self.DRAW_ROLLOUT_CONTINUATION_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
            "W": clamp(
                (-joint_collapse_continuation_bonus_gap)
                / self.DRAW_ROLLOUT_CONTINUATION_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
        }
        summary["public_reveal_bridge_signal_pressure"] = {
            "B": clamp(public_reveal_bridge_signal_gap, -1.0, 1.0),
            "W": clamp(-public_reveal_bridge_signal_gap, -1.0, 1.0),
        }
        summary["public_reveal_bridge_bonus_pressure"] = {
            "B": clamp(
                public_reveal_bridge_bonus_gap / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
            "W": clamp(
                (-public_reveal_bridge_bonus_gap) / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
        }
        summary["public_reveal_bridge_continuation_pressure"] = {
            "B": clamp(
                public_reveal_bridge_continuation_bonus_gap
                / self.DRAW_ROLLOUT_CONTINUATION_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
            "W": clamp(
                (-public_reveal_bridge_continuation_bonus_gap)
                / self.DRAW_ROLLOUT_CONTINUATION_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
        }
        summary["target_finish_chain_signal_pressure"] = {
            "B": clamp(target_finish_chain_signal_gap, -1.0, 1.0),
            "W": clamp(-target_finish_chain_signal_gap, -1.0, 1.0),
        }
        summary["target_chain_signal_pressure"] = {
            "B": clamp(target_chain_signal_gap, -1.0, 1.0),
            "W": clamp(-target_chain_signal_gap, -1.0, 1.0),
        }
        summary["target_chain_bonus_pressure"] = {
            "B": clamp(
                target_chain_bonus_gap / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
            "W": clamp(
                (-target_chain_bonus_gap) / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
        }
        summary["target_chain_continuation_pressure"] = {
            "B": clamp(
                target_chain_continuation_bonus_gap
                / self.DRAW_ROLLOUT_CONTINUATION_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
            "W": clamp(
                (-target_chain_continuation_bonus_gap)
                / self.DRAW_ROLLOUT_CONTINUATION_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
        }
        summary["target_finish_chain_bonus_pressure"] = {
            "B": clamp(
                target_finish_chain_bonus_gap / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
            "W": clamp(
                (-target_finish_chain_bonus_gap) / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
        }
        summary["target_finish_chain_continuation_pressure"] = {
            "B": clamp(
                target_finish_chain_continuation_bonus_gap
                / self.DRAW_ROLLOUT_CONTINUATION_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
            "W": clamp(
                (-target_finish_chain_continuation_bonus_gap)
                / self.DRAW_ROLLOUT_CONTINUATION_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
        }
        summary["attackability_pressure"] = {
            "B": clamp(
                attackability_gap / self.DRAW_ROLLOUT_ATTACKABILITY_REFERENCE,
                -1.0,
                1.0,
            ),
            "W": clamp(
                (-attackability_gap) / self.DRAW_ROLLOUT_ATTACKABILITY_REFERENCE,
                -1.0,
                1.0,
            ),
        }
        summary["target_retention_pressure"] = {
            "B": clamp(target_retention_gap, -1.0, 1.0),
            "W": clamp(-target_retention_gap, -1.0, 1.0),
        }
        summary["color_alignment_pressure"] = {
            "B": clamp(color_alignment_gap, -1.0, 1.0),
            "W": clamp(-color_alignment_gap, -1.0, 1.0),
        }
        summary["best_gap_pressure"] = {
            "B": clamp(
                best_gap_gap / self.DRAW_ROLLOUT_BEST_GAP_REFERENCE,
                -1.0,
                1.0,
            ),
            "W": clamp(
                (-best_gap_gap) / self.DRAW_ROLLOUT_BEST_GAP_REFERENCE,
                -1.0,
                1.0,
            ),
        }
        summary["strategy_objective_pressure"] = {
            "B": clamp(
                strategy_objective_gap / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
            "W": clamp(
                (-strategy_objective_gap) / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
        }
        summary["active_opening_pressure"] = {
            "B": clamp(active_opening_gap, -1.0, 1.0),
            "W": clamp(-active_opening_gap, -1.0, 1.0),
        }
        summary["best_value_stability_pressure"] = {
            "B": clamp(
                best_value_stddev_gap / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
            "W": clamp(
                (-best_value_stddev_gap) / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
        }
        summary["win_probability_stability_pressure"] = {
            "B": clamp(win_probability_stddev_gap, -1.0, 1.0),
            "W": clamp(-win_probability_stddev_gap, -1.0, 1.0),
        }
        summary["best_value_floor_pressure"] = {
            "B": clamp(
                best_value_floor_gap / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
            "W": clamp(
                (-best_value_floor_gap) / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
        }
        summary["win_probability_floor_pressure"] = {
            "B": clamp(win_probability_floor_gap, -1.0, 1.0),
            "W": clamp(-win_probability_floor_gap, -1.0, 1.0),
        }
        summary["information_gain_pressure"] = {
            "B": clamp(
                information_gain_gap / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
            "W": clamp(
                (-information_gain_gap) / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
        }
        summary["information_gain_floor_pressure"] = {
            "B": clamp(
                information_gain_floor_gap / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
            "W": clamp(
                (-information_gain_floor_gap) / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            ),
        }
        return summary

    def _build_draw_opening_plan(
        self,
        draw_color_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        recommended_color = draw_color_summary.get("recommended_color")
        if recommended_color not in CARD_COLORS:
            return {}
        color_suffix = "black" if recommended_color == "B" else "white"
        guess_card = draw_color_summary.get(
            f"draw_rollout_opening_guess_card_{color_suffix}"
        )
        return {
            "recommended_color": recommended_color,
            "target_player_id": draw_color_summary.get(
                f"draw_rollout_opening_target_player_id_{color_suffix}"
            ),
            "target_slot_index": draw_color_summary.get(
                f"draw_rollout_opening_target_slot_{color_suffix}"
            ),
            "guess_card": [guess_card[0], guess_card[1]]
            if isinstance(guess_card, (list, tuple)) and len(guess_card) == 2
            else None,
            "support_ratio": draw_color_summary.get(
                f"draw_rollout_opening_support_{color_suffix}",
                0.0,
            ),
            "guess_support_ratio": draw_color_summary.get(
                f"draw_rollout_opening_guess_support_{color_suffix}",
                0.0,
            ),
            "expected_value": draw_color_summary.get(
                f"draw_rollout_opening_expected_value_{color_suffix}",
                0.0,
            ),
            "strategy_objective": draw_color_summary.get(
                f"draw_rollout_opening_strategy_objective_{color_suffix}",
                0.0,
            ),
            "win_probability": draw_color_summary.get(
                f"draw_rollout_opening_win_probability_{color_suffix}",
                0.0,
            ),
            "information_gain": draw_color_summary.get(
                f"draw_rollout_opening_information_gain_{color_suffix}",
                0.0,
            ),
            "continuation_likelihood": draw_color_summary.get(
                f"draw_rollout_opening_continuation_likelihood_{color_suffix}",
                0.0,
            ),
        }

    def _build_draw_color_summary(
        self,
        full_probability_matrix: Optional[FullProbabilityMatrix] = None,
    ) -> Dict[str, Any]:
        available_counts = {
            color: sum(1 for card in self.inference_engine.available_cards if card[0] == color)
            for color in CARD_COLORS
        }
        self_counts = {
            color: sum(
                1
                for slot in self.game_state.self_player().ordered_slots()
                if getattr(slot, "color", None) == color
            )
            for color in CARD_COLORS
        }
        total_self_cards = max(1, sum(self_counts.values()))
        total_available = max(1, sum(available_counts.values()))

        defense_balance = {
            "B": (self_counts["W"] - self_counts["B"]) / total_self_cards,
            "W": (self_counts["B"] - self_counts["W"]) / total_self_cards,
        }
        defense_guard_factor = clamp(
            1.0 - (0.75 * abs(defense_balance["B"] - defense_balance["W"])),
            0.25,
            1.0,
        )
        hidden_color_mass = {"B": 0.0, "W": 0.0}
        total_hidden_positions = 0.0
        for probability_matrix in (full_probability_matrix or {}).values():
            for slot_distribution in probability_matrix.values():
                total_hidden_positions += 1.0
                for card, probability in slot_distribution.items():
                    hidden_color_mass[card[0]] += float(probability)
        if total_hidden_positions > 0.0:
            offense_pressure = {
                color: hidden_color_mass[color] / total_hidden_positions
                for color in CARD_COLORS
            }
        else:
            offense_pressure = {"B": 0.0, "W": 0.0}
        entropy_pressure = self._color_entropy_pressure(full_probability_matrix)
        target_entropy_pressure = self._color_entropy_pressure(
            full_probability_matrix,
            target_player_only=True,
        )
        self_flexibility_pressure = self._self_flexibility_pressure()
        hidden_defense_pressure = self._hidden_defense_pressure()
        target_boundary_pressure = self._target_boundary_pressure(full_probability_matrix)
        target_finish_pressure = self._target_finish_pressure(full_probability_matrix)
        target_focus_pressure = self._target_focus_pressure(full_probability_matrix)
        target_recent_momentum_pressure = self._target_recent_momentum_pressure()
        recent_self_exposure_pressure = self._recent_self_exposure_pressure()
        target_attack_window_factor = self._target_attack_window_factor()
        draw_rollout = self._draw_rollout_summary()
        draw_rollout_expected_strategy_objective = draw_rollout.get(
            "expected_strategy_objective",
            {"B": 0.0, "W": 0.0},
        )
        draw_rollout_strategy_objective_pressure = draw_rollout.get(
            "strategy_objective_pressure",
            {"B": 0.0, "W": 0.0},
        )
        draw_rollout_self_exposure_pressure = {
            color: (
                draw_rollout["expected_self_public_exposure"]["W" if color == "B" else "B"]
                - draw_rollout["expected_self_public_exposure"][color]
            )
            for color in CARD_COLORS
        }
        draw_rollout_new_drawn_exposure_pressure = {
            color: (
                draw_rollout["expected_self_newly_drawn_exposure"]["W" if color == "B" else "B"]
                - draw_rollout["expected_self_newly_drawn_exposure"][color]
            )
            for color in CARD_COLORS
        }
        draw_rollout_finish_fragility_pressure = {
            color: (
                draw_rollout["expected_self_finish_fragility"]["W" if color == "B" else "B"]
                - draw_rollout["expected_self_finish_fragility"][color]
            )
            for color in CARD_COLORS
        }
        draw_rollout_stop_threshold_pressure = {
            color: clamp(
                (
                    draw_rollout["expected_stop_threshold"]["W" if color == "B" else "B"]
                    - draw_rollout["expected_stop_threshold"][color]
                ) / self.DRAW_ROLLOUT_STOP_THRESHOLD_REFERENCE,
                -1.0,
                1.0,
            )
            for color in CARD_COLORS
        }
        draw_rollout_continue_margin_pressure = {
            color: clamp(
                (
                    draw_rollout["expected_continue_margin"][color]
                    - draw_rollout["expected_continue_margin"]["W" if color == "B" else "B"]
                ) / self.DRAW_ROLLOUT_CONTINUE_MARGIN_REFERENCE,
                -1.0,
                1.0,
            )
            for color in CARD_COLORS
        }
        draw_rollout_failure_collapse_pressure = {
            color: clamp(
                (
                    draw_rollout["expected_failure_collapse_bonus"][color]
                    - draw_rollout["expected_failure_collapse_bonus"]["W" if color == "B" else "B"]
                ) / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            )
            for color in CARD_COLORS
        }
        draw_rollout_failed_switch_pressure = {
            color: clamp(
                (
                    draw_rollout["expected_failed_guess_switch_bonus"][color]
                    - draw_rollout["expected_failed_guess_switch_bonus"]["W" if color == "B" else "B"]
                ) / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            )
            for color in CARD_COLORS
        }
        draw_rollout_post_hit_failure_recovery_pressure = {
            color: clamp(
                (
                    draw_rollout["expected_post_hit_failure_recovery_bonus"][color]
                    - draw_rollout["expected_post_hit_failure_recovery_bonus"]["W" if color == "B" else "B"]
                ) / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            )
            for color in CARD_COLORS
        }
        draw_rollout_post_hit_failed_switch_pressure = {
            color: clamp(
                (
                    draw_rollout["expected_post_hit_failed_switch_bonus"][color]
                    - draw_rollout["expected_post_hit_failed_switch_bonus"]["W" if color == "B" else "B"]
                ) / self.DRAW_ROLLOUT_VALUE_REFERENCE,
                -1.0,
                1.0,
            )
            for color in CARD_COLORS
        }
        target_hidden_color_mass = {"B": 0.0, "W": 0.0}
        target_hidden_positions = 0.0
        target_player_id = getattr(self.game_state, "target_player_id", None)
        if target_player_id is not None:
            for slot_distribution in (full_probability_matrix or {}).get(
                target_player_id,
                {},
            ).values():
                target_hidden_positions += 1.0
                for card, probability in slot_distribution.items():
                    target_hidden_color_mass[card[0]] += float(probability)
        if target_hidden_positions > 0.0:
            target_attack_pressure = {
                color: target_hidden_color_mass[color] / target_hidden_positions
                for color in CARD_COLORS
            }
        else:
            target_attack_pressure = {"B": 0.0, "W": 0.0}
        availability_pressure = {
            color: (
                available_counts[color]
                - available_counts["W" if color == "B" else "B"]
            )
            / total_available
            for color in CARD_COLORS
        }
        base_color_scores = {
            color: defense_balance[color]
            + (0.28 * hidden_defense_pressure[color])
            + (0.26 * recent_self_exposure_pressure[color])
            + (
                defense_guard_factor
                * (
                    (0.35 * offense_pressure[color])
                    + (0.24 * entropy_pressure[color])
                    + (0.18 * availability_pressure[color])
                    + (0.22 * self_flexibility_pressure[color])
                    + (
                        target_attack_window_factor
                        * (
                            (0.24 * target_entropy_pressure[color])
                            + (0.34 * target_boundary_pressure[color])
                            + (0.26 * target_finish_pressure[color])
                            + (0.26 * target_focus_pressure[color])
                            + (0.24 * target_recent_momentum_pressure[color])
                            + (0.28 * target_attack_pressure[color])
                        )
                    )
                )
            )
            for color in CARD_COLORS
        }
        pre_rollout_margin = abs(
            base_color_scores["B"] - base_color_scores["W"]
        )
        draw_rollout_edge_scale = clamp(
            1.0 - (pre_rollout_margin / self.DRAW_ROLLOUT_EDGE_WINDOW),
            0.0,
            1.0,
        )
        target_lock_signal = max(
            abs(target_boundary_pressure["B"] - target_boundary_pressure["W"]),
            abs(target_finish_pressure["B"] - target_finish_pressure["W"]),
            abs(target_attack_pressure["B"] - target_attack_pressure["W"]),
        )
        draw_rollout_target_gate = clamp(
            1.0 - (0.75 * target_lock_signal),
            0.0,
            1.0,
        )
        draw_rollout_phase_gate = self._draw_rollout_phase_gate()
        draw_rollout_plan_gate = self._draw_rollout_plan_gate(draw_rollout)
        draw_rollout_activation_scale = (
            draw_rollout_edge_scale
            * draw_rollout_target_gate
            * draw_rollout_phase_gate
            * draw_rollout_plan_gate
        )
        color_scores = {
            color: base_color_scores[color]
            + (
                draw_rollout_activation_scale
                * (
                    (0.16 * draw_rollout["value_pressure"][color])
                    + (0.08 * draw_rollout["immediate_value_pressure"][color])
                    + (0.10 * draw_rollout["continuation_value_pressure"][color])
                    + (0.06 * draw_rollout["continuation_likelihood_pressure"][color])
                    + (0.08 * draw_rollout["win_probability_pressure"][color])
                    + (0.10 * draw_rollout["attackability_pressure"][color])
                    + (0.08 * draw_rollout["target_retention_pressure"][color])
                    + (0.06 * draw_rollout["color_alignment_pressure"][color])
                    + (0.10 * draw_rollout["best_gap_pressure"][color])
                    + (0.12 * draw_rollout_strategy_objective_pressure[color])
                    + (0.08 * draw_rollout["active_opening_pressure"][color])
                    + (0.08 * draw_rollout["best_value_stability_pressure"][color])
                    + (0.05 * draw_rollout["win_probability_stability_pressure"][color])
                    + (0.10 * draw_rollout["best_value_floor_pressure"][color])
                    + (0.06 * draw_rollout["win_probability_floor_pressure"][color])
                    + (0.08 * draw_rollout["information_gain_pressure"][color])
                    + (0.05 * draw_rollout["information_gain_floor_pressure"][color])
                    + (0.08 * draw_rollout_self_exposure_pressure[color])
                    + (0.06 * draw_rollout_new_drawn_exposure_pressure[color])
                    + (0.08 * draw_rollout_finish_fragility_pressure[color])
                    + (0.10 * draw_rollout_stop_threshold_pressure[color])
                    + (0.10 * draw_rollout_continue_margin_pressure[color])
                    + (0.08 * draw_rollout_failure_collapse_pressure[color])
                    + (0.08 * draw_rollout_failed_switch_pressure[color])
                    + (0.08 * draw_rollout_post_hit_failure_recovery_pressure[color])
                    + (0.06 * draw_rollout_post_hit_failed_switch_pressure[color])
                    + (0.08 * draw_rollout["target_attack_window_signal_pressure"][color])
                    + (0.10 * draw_rollout["target_attack_window_bonus_pressure"][color])
                    + (0.08 * draw_rollout["target_attack_window_continuation_pressure"][color])
                    + (0.08 * draw_rollout["joint_collapse_signal_pressure"][color])
                    + (0.10 * draw_rollout["joint_collapse_bonus_pressure"][color])
                    + (0.08 * draw_rollout["joint_collapse_continuation_pressure"][color])
                    + (0.08 * draw_rollout["public_reveal_bridge_signal_pressure"][color])
                    + (0.08 * draw_rollout["public_reveal_bridge_bonus_pressure"][color])
                    + (0.06 * draw_rollout["public_reveal_bridge_continuation_pressure"][color])
                    + (0.08 * draw_rollout["target_chain_signal_pressure"][color])
                    + (0.10 * draw_rollout["target_chain_bonus_pressure"][color])
                    + (0.08 * draw_rollout["target_chain_continuation_pressure"][color])
                    + (0.08 * draw_rollout["target_finish_chain_signal_pressure"][color])
                    + (0.10 * draw_rollout["target_finish_chain_bonus_pressure"][color])
                    + (0.08 * draw_rollout["target_finish_chain_continuation_pressure"][color])
                )
            )
            for color in CARD_COLORS
        }
        recommended_color = max(
            CARD_COLORS,
            key=lambda color: (color_scores[color], available_counts[color], color),
        )
        dominant_factor_margins = {
            "defense_balance": abs(defense_balance["B"] - defense_balance["W"]),
            "hidden_defense_pressure": abs(
                hidden_defense_pressure["B"] - hidden_defense_pressure["W"]
            ),
            "recent_self_exposure_pressure": abs(
                recent_self_exposure_pressure["B"]
                - recent_self_exposure_pressure["W"]
            ),
            "draw_rollout_self_exposure_pressure": (
                draw_rollout_activation_scale
                * 0.08
                * abs(
                    draw_rollout_self_exposure_pressure["B"]
                    - draw_rollout_self_exposure_pressure["W"]
                )
            ),
            "draw_rollout_new_drawn_exposure_pressure": (
                draw_rollout_activation_scale
                * 0.06
                * abs(
                    draw_rollout_new_drawn_exposure_pressure["B"]
                    - draw_rollout_new_drawn_exposure_pressure["W"]
                )
            ),
            "draw_rollout_finish_fragility_pressure": (
                draw_rollout_activation_scale
                * 0.08
                * abs(
                    draw_rollout_finish_fragility_pressure["B"]
                    - draw_rollout_finish_fragility_pressure["W"]
                )
            ),
            "draw_rollout_stop_threshold_pressure": (
                draw_rollout_activation_scale
                * 0.10
                * abs(
                    draw_rollout_stop_threshold_pressure["B"]
                    - draw_rollout_stop_threshold_pressure["W"]
                )
            ),
            "draw_rollout_continue_margin_pressure": (
                draw_rollout_activation_scale
                * 0.10
                * abs(
                    draw_rollout_continue_margin_pressure["B"]
                    - draw_rollout_continue_margin_pressure["W"]
                )
            ),
            "draw_rollout_failure_collapse_pressure": (
                draw_rollout_activation_scale
                * 0.08
                * abs(
                    draw_rollout_failure_collapse_pressure["B"]
                    - draw_rollout_failure_collapse_pressure["W"]
                )
            ),
            "draw_rollout_failed_switch_pressure": (
                draw_rollout_activation_scale
                * 0.08
                * abs(
                    draw_rollout_failed_switch_pressure["B"]
                    - draw_rollout_failed_switch_pressure["W"]
                )
            ),
            "draw_rollout_post_hit_failure_recovery_pressure": (
                draw_rollout_activation_scale
                * 0.08
                * abs(
                    draw_rollout_post_hit_failure_recovery_pressure["B"]
                    - draw_rollout_post_hit_failure_recovery_pressure["W"]
                )
            ),
            "draw_rollout_post_hit_failed_switch_pressure": (
                draw_rollout_activation_scale
                * 0.06
                * abs(
                    draw_rollout_post_hit_failed_switch_pressure["B"]
                    - draw_rollout_post_hit_failed_switch_pressure["W"]
                )
            ),
            "draw_rollout_target_attack_window_signal_pressure": (
                draw_rollout_activation_scale
                * 0.08
                * abs(
                    draw_rollout["target_attack_window_signal_pressure"]["B"]
                    - draw_rollout["target_attack_window_signal_pressure"]["W"]
                )
            ),
            "draw_rollout_target_attack_window_bonus_pressure": (
                draw_rollout_activation_scale
                * 0.10
                * abs(
                    draw_rollout["target_attack_window_bonus_pressure"]["B"]
                    - draw_rollout["target_attack_window_bonus_pressure"]["W"]
                )
            ),
            "draw_rollout_target_attack_window_continuation_pressure": (
                draw_rollout_activation_scale
                * 0.08
                * abs(
                    draw_rollout["target_attack_window_continuation_pressure"]["B"]
                    - draw_rollout["target_attack_window_continuation_pressure"]["W"]
                )
            ),
            "draw_rollout_joint_collapse_signal_pressure": (
                draw_rollout_activation_scale
                * 0.08
                * abs(
                    draw_rollout["joint_collapse_signal_pressure"]["B"]
                    - draw_rollout["joint_collapse_signal_pressure"]["W"]
                )
            ),
            "draw_rollout_joint_collapse_bonus_pressure": (
                draw_rollout_activation_scale
                * 0.10
                * abs(
                    draw_rollout["joint_collapse_bonus_pressure"]["B"]
                    - draw_rollout["joint_collapse_bonus_pressure"]["W"]
                )
            ),
            "draw_rollout_joint_collapse_continuation_pressure": (
                draw_rollout_activation_scale
                * 0.08
                * abs(
                    draw_rollout["joint_collapse_continuation_pressure"]["B"]
                    - draw_rollout["joint_collapse_continuation_pressure"]["W"]
                )
            ),
            "draw_rollout_public_reveal_bridge_signal_pressure": (
                draw_rollout_activation_scale
                * 0.08
                * abs(
                    draw_rollout["public_reveal_bridge_signal_pressure"]["B"]
                    - draw_rollout["public_reveal_bridge_signal_pressure"]["W"]
                )
            ),
            "draw_rollout_public_reveal_bridge_bonus_pressure": (
                draw_rollout_activation_scale
                * 0.08
                * abs(
                    draw_rollout["public_reveal_bridge_bonus_pressure"]["B"]
                    - draw_rollout["public_reveal_bridge_bonus_pressure"]["W"]
                )
            ),
            "draw_rollout_public_reveal_bridge_continuation_pressure": (
                draw_rollout_activation_scale
                * 0.06
                * abs(
                    draw_rollout["public_reveal_bridge_continuation_pressure"]["B"]
                    - draw_rollout["public_reveal_bridge_continuation_pressure"]["W"]
                )
            ),
            "draw_rollout_target_chain_signal_pressure": (
                draw_rollout_activation_scale
                * 0.08
                * abs(
                    draw_rollout["target_chain_signal_pressure"]["B"]
                    - draw_rollout["target_chain_signal_pressure"]["W"]
                )
            ),
            "draw_rollout_target_chain_bonus_pressure": (
                draw_rollout_activation_scale
                * 0.10
                * abs(
                    draw_rollout["target_chain_bonus_pressure"]["B"]
                    - draw_rollout["target_chain_bonus_pressure"]["W"]
                )
            ),
            "draw_rollout_target_chain_continuation_pressure": (
                draw_rollout_activation_scale
                * 0.08
                * abs(
                    draw_rollout["target_chain_continuation_pressure"]["B"]
                    - draw_rollout["target_chain_continuation_pressure"]["W"]
                )
            ),
            "draw_rollout_target_finish_chain_signal_pressure": (
                draw_rollout_activation_scale
                * 0.08
                * abs(
                    draw_rollout["target_finish_chain_signal_pressure"]["B"]
                    - draw_rollout["target_finish_chain_signal_pressure"]["W"]
                )
            ),
            "draw_rollout_target_finish_chain_bonus_pressure": (
                draw_rollout_activation_scale
                * 0.10
                * abs(
                    draw_rollout["target_finish_chain_bonus_pressure"]["B"]
                    - draw_rollout["target_finish_chain_bonus_pressure"]["W"]
                )
            ),
            "draw_rollout_target_finish_chain_continuation_pressure": (
                draw_rollout_activation_scale
                * 0.08
                * abs(
                    draw_rollout["target_finish_chain_continuation_pressure"]["B"]
                    - draw_rollout["target_finish_chain_continuation_pressure"]["W"]
                )
            ),
            "draw_rollout_strategy_objective_pressure": (
                draw_rollout_activation_scale
                * 0.12
                * abs(
                    draw_rollout_strategy_objective_pressure["B"]
                    - draw_rollout_strategy_objective_pressure["W"]
                )
            ),
            "offense_pressure": abs(offense_pressure["B"] - offense_pressure["W"]),
            "entropy_pressure": abs(entropy_pressure["B"] - entropy_pressure["W"]),
            "target_entropy_pressure": abs(
                target_entropy_pressure["B"] - target_entropy_pressure["W"]
            ),
            "target_boundary_pressure": abs(
                target_boundary_pressure["B"] - target_boundary_pressure["W"]
            ),
            "target_finish_pressure": abs(
                target_finish_pressure["B"] - target_finish_pressure["W"]
            ),
            "self_flexibility_pressure": abs(
                self_flexibility_pressure["B"] - self_flexibility_pressure["W"]
            ),
            "target_attack_pressure": abs(
                target_attack_pressure["B"] - target_attack_pressure["W"]
            ),
            "target_focus_pressure": abs(
                target_focus_pressure["B"] - target_focus_pressure["W"]
            ),
            "target_recent_momentum_pressure": abs(
                target_recent_momentum_pressure["B"]
                - target_recent_momentum_pressure["W"]
            ),
            "availability_pressure": abs(
                availability_pressure["B"] - availability_pressure["W"]
            ),
        }
        black_opening_plan = draw_rollout["opening_plan"]["B"]
        white_opening_plan = draw_rollout["opening_plan"]["W"]
        return {
            "recommended_color": recommended_color,
            "black_score": color_scores["B"],
            "white_score": color_scores["W"],
            "defense_balance_black": defense_balance["B"],
            "defense_balance_white": defense_balance["W"],
            "hidden_defense_pressure_black": hidden_defense_pressure["B"],
            "hidden_defense_pressure_white": hidden_defense_pressure["W"],
            "recent_self_exposure_pressure_black": recent_self_exposure_pressure["B"],
            "recent_self_exposure_pressure_white": recent_self_exposure_pressure["W"],
            "draw_rollout_expected_best_value_black": draw_rollout["expected_best_value"]["B"],
            "draw_rollout_expected_best_value_white": draw_rollout["expected_best_value"]["W"],
            "draw_rollout_expected_immediate_value_black": draw_rollout["expected_immediate_value"]["B"],
            "draw_rollout_expected_immediate_value_white": draw_rollout["expected_immediate_value"]["W"],
            "draw_rollout_value_pressure_black": draw_rollout["value_pressure"]["B"],
            "draw_rollout_value_pressure_white": draw_rollout["value_pressure"]["W"],
            "draw_rollout_immediate_value_pressure_black": draw_rollout["immediate_value_pressure"]["B"],
            "draw_rollout_immediate_value_pressure_white": draw_rollout["immediate_value_pressure"]["W"],
            "draw_rollout_expected_continuation_value_black": draw_rollout["expected_continuation_value"]["B"],
            "draw_rollout_expected_continuation_value_white": draw_rollout["expected_continuation_value"]["W"],
            "draw_rollout_expected_continuation_likelihood_black": draw_rollout["expected_continuation_likelihood"]["B"],
            "draw_rollout_expected_continuation_likelihood_white": draw_rollout["expected_continuation_likelihood"]["W"],
            "draw_rollout_expected_self_public_exposure_black": draw_rollout["expected_self_public_exposure"]["B"],
            "draw_rollout_expected_self_public_exposure_white": draw_rollout["expected_self_public_exposure"]["W"],
            "draw_rollout_expected_self_newly_drawn_exposure_black": draw_rollout["expected_self_newly_drawn_exposure"]["B"],
            "draw_rollout_expected_self_newly_drawn_exposure_white": draw_rollout["expected_self_newly_drawn_exposure"]["W"],
            "draw_rollout_expected_self_finish_fragility_black": draw_rollout["expected_self_finish_fragility"]["B"],
            "draw_rollout_expected_self_finish_fragility_white": draw_rollout["expected_self_finish_fragility"]["W"],
            "draw_rollout_expected_stop_threshold_black": draw_rollout["expected_stop_threshold"]["B"],
            "draw_rollout_expected_stop_threshold_white": draw_rollout["expected_stop_threshold"]["W"],
            "draw_rollout_expected_continue_margin_black": draw_rollout["expected_continue_margin"]["B"],
            "draw_rollout_expected_continue_margin_white": draw_rollout["expected_continue_margin"]["W"],
            "draw_rollout_expected_failure_collapse_bonus_black": draw_rollout["expected_failure_collapse_bonus"]["B"],
            "draw_rollout_expected_failure_collapse_bonus_white": draw_rollout["expected_failure_collapse_bonus"]["W"],
            "draw_rollout_expected_failed_guess_switch_bonus_black": draw_rollout["expected_failed_guess_switch_bonus"]["B"],
            "draw_rollout_expected_failed_guess_switch_bonus_white": draw_rollout["expected_failed_guess_switch_bonus"]["W"],
            "draw_rollout_expected_failed_guess_switch_signal_black": draw_rollout["expected_failed_guess_switch_signal"]["B"],
            "draw_rollout_expected_failed_guess_switch_signal_white": draw_rollout["expected_failed_guess_switch_signal"]["W"],
            "draw_rollout_expected_post_hit_failure_recovery_bonus_black": draw_rollout["expected_post_hit_failure_recovery_bonus"]["B"],
            "draw_rollout_expected_post_hit_failure_recovery_bonus_white": draw_rollout["expected_post_hit_failure_recovery_bonus"]["W"],
            "draw_rollout_expected_post_hit_failed_switch_bonus_black": draw_rollout["expected_post_hit_failed_switch_bonus"]["B"],
            "draw_rollout_expected_post_hit_failed_switch_bonus_white": draw_rollout["expected_post_hit_failed_switch_bonus"]["W"],
            "draw_rollout_expected_target_attack_window_signal_black": draw_rollout["expected_target_attack_window_signal"]["B"],
            "draw_rollout_expected_target_attack_window_signal_white": draw_rollout["expected_target_attack_window_signal"]["W"],
            "draw_rollout_expected_target_attack_window_bonus_black": draw_rollout["expected_target_attack_window_bonus"]["B"],
            "draw_rollout_expected_target_attack_window_bonus_white": draw_rollout["expected_target_attack_window_bonus"]["W"],
            "draw_rollout_expected_target_attack_window_continuation_bonus_black": draw_rollout["expected_target_attack_window_continuation_bonus"]["B"],
            "draw_rollout_expected_target_attack_window_continuation_bonus_white": draw_rollout["expected_target_attack_window_continuation_bonus"]["W"],
            "draw_rollout_expected_target_chain_signal_black": draw_rollout["expected_target_chain_signal"]["B"],
            "draw_rollout_expected_target_chain_signal_white": draw_rollout["expected_target_chain_signal"]["W"],
            "draw_rollout_expected_target_chain_bonus_black": draw_rollout["expected_target_chain_bonus"]["B"],
            "draw_rollout_expected_target_chain_bonus_white": draw_rollout["expected_target_chain_bonus"]["W"],
            "draw_rollout_expected_target_chain_continuation_bonus_black": draw_rollout["expected_target_chain_continuation_bonus"]["B"],
            "draw_rollout_expected_target_chain_continuation_bonus_white": draw_rollout["expected_target_chain_continuation_bonus"]["W"],
            "draw_rollout_expected_target_finish_chain_signal_black": draw_rollout["expected_target_finish_chain_signal"]["B"],
            "draw_rollout_expected_target_finish_chain_signal_white": draw_rollout["expected_target_finish_chain_signal"]["W"],
            "draw_rollout_expected_target_finish_chain_bonus_black": draw_rollout["expected_target_finish_chain_bonus"]["B"],
            "draw_rollout_expected_target_finish_chain_bonus_white": draw_rollout["expected_target_finish_chain_bonus"]["W"],
            "draw_rollout_expected_target_finish_chain_continuation_bonus_black": draw_rollout["expected_target_finish_chain_continuation_bonus"]["B"],
            "draw_rollout_expected_target_finish_chain_continuation_bonus_white": draw_rollout["expected_target_finish_chain_continuation_bonus"]["W"],
            "draw_rollout_expected_joint_collapse_signal_black": draw_rollout["expected_joint_collapse_signal"]["B"],
            "draw_rollout_expected_joint_collapse_signal_white": draw_rollout["expected_joint_collapse_signal"]["W"],
            "draw_rollout_expected_joint_collapse_bonus_black": draw_rollout["expected_joint_collapse_bonus"]["B"],
            "draw_rollout_expected_joint_collapse_bonus_white": draw_rollout["expected_joint_collapse_bonus"]["W"],
            "draw_rollout_expected_joint_collapse_continuation_bonus_black": draw_rollout["expected_joint_collapse_continuation_bonus"]["B"],
            "draw_rollout_expected_joint_collapse_continuation_bonus_white": draw_rollout["expected_joint_collapse_continuation_bonus"]["W"],
            "draw_rollout_expected_public_reveal_bridge_signal_black": draw_rollout["expected_public_reveal_bridge_signal"]["B"],
            "draw_rollout_expected_public_reveal_bridge_signal_white": draw_rollout["expected_public_reveal_bridge_signal"]["W"],
            "draw_rollout_expected_public_reveal_bridge_bonus_black": draw_rollout["expected_public_reveal_bridge_bonus"]["B"],
            "draw_rollout_expected_public_reveal_bridge_bonus_white": draw_rollout["expected_public_reveal_bridge_bonus"]["W"],
            "draw_rollout_expected_public_reveal_bridge_continuation_bonus_black": draw_rollout["expected_public_reveal_bridge_continuation_bonus"]["B"],
            "draw_rollout_expected_public_reveal_bridge_continuation_bonus_white": draw_rollout["expected_public_reveal_bridge_continuation_bonus"]["W"],
            "draw_rollout_continuation_value_pressure_black": draw_rollout["continuation_value_pressure"]["B"],
            "draw_rollout_continuation_value_pressure_white": draw_rollout["continuation_value_pressure"]["W"],
            "draw_rollout_continuation_likelihood_pressure_black": draw_rollout["continuation_likelihood_pressure"]["B"],
            "draw_rollout_continuation_likelihood_pressure_white": draw_rollout["continuation_likelihood_pressure"]["W"],
            "draw_rollout_self_exposure_pressure_black": draw_rollout_self_exposure_pressure["B"],
            "draw_rollout_self_exposure_pressure_white": draw_rollout_self_exposure_pressure["W"],
            "draw_rollout_new_drawn_exposure_pressure_black": draw_rollout_new_drawn_exposure_pressure["B"],
            "draw_rollout_new_drawn_exposure_pressure_white": draw_rollout_new_drawn_exposure_pressure["W"],
            "draw_rollout_finish_fragility_pressure_black": draw_rollout_finish_fragility_pressure["B"],
            "draw_rollout_finish_fragility_pressure_white": draw_rollout_finish_fragility_pressure["W"],
            "draw_rollout_stop_threshold_pressure_black": draw_rollout_stop_threshold_pressure["B"],
            "draw_rollout_stop_threshold_pressure_white": draw_rollout_stop_threshold_pressure["W"],
            "draw_rollout_continue_margin_pressure_black": draw_rollout_continue_margin_pressure["B"],
            "draw_rollout_continue_margin_pressure_white": draw_rollout_continue_margin_pressure["W"],
            "draw_rollout_failure_collapse_pressure_black": draw_rollout_failure_collapse_pressure["B"],
            "draw_rollout_failure_collapse_pressure_white": draw_rollout_failure_collapse_pressure["W"],
            "draw_rollout_failed_switch_pressure_black": draw_rollout_failed_switch_pressure["B"],
            "draw_rollout_failed_switch_pressure_white": draw_rollout_failed_switch_pressure["W"],
            "draw_rollout_post_hit_failure_recovery_pressure_black": draw_rollout_post_hit_failure_recovery_pressure["B"],
            "draw_rollout_post_hit_failure_recovery_pressure_white": draw_rollout_post_hit_failure_recovery_pressure["W"],
            "draw_rollout_post_hit_failed_switch_pressure_black": draw_rollout_post_hit_failed_switch_pressure["B"],
            "draw_rollout_post_hit_failed_switch_pressure_white": draw_rollout_post_hit_failed_switch_pressure["W"],
            "draw_rollout_target_attack_window_signal_pressure_black": draw_rollout["target_attack_window_signal_pressure"]["B"],
            "draw_rollout_target_attack_window_signal_pressure_white": draw_rollout["target_attack_window_signal_pressure"]["W"],
            "draw_rollout_target_attack_window_bonus_pressure_black": draw_rollout["target_attack_window_bonus_pressure"]["B"],
            "draw_rollout_target_attack_window_bonus_pressure_white": draw_rollout["target_attack_window_bonus_pressure"]["W"],
            "draw_rollout_target_attack_window_continuation_pressure_black": draw_rollout["target_attack_window_continuation_pressure"]["B"],
            "draw_rollout_target_attack_window_continuation_pressure_white": draw_rollout["target_attack_window_continuation_pressure"]["W"],
            "draw_rollout_target_chain_signal_pressure_black": draw_rollout["target_chain_signal_pressure"]["B"],
            "draw_rollout_target_chain_signal_pressure_white": draw_rollout["target_chain_signal_pressure"]["W"],
            "draw_rollout_target_chain_bonus_pressure_black": draw_rollout["target_chain_bonus_pressure"]["B"],
            "draw_rollout_target_chain_bonus_pressure_white": draw_rollout["target_chain_bonus_pressure"]["W"],
            "draw_rollout_target_chain_continuation_pressure_black": draw_rollout["target_chain_continuation_pressure"]["B"],
            "draw_rollout_target_chain_continuation_pressure_white": draw_rollout["target_chain_continuation_pressure"]["W"],
            "draw_rollout_target_finish_chain_signal_pressure_black": draw_rollout["target_finish_chain_signal_pressure"]["B"],
            "draw_rollout_target_finish_chain_signal_pressure_white": draw_rollout["target_finish_chain_signal_pressure"]["W"],
            "draw_rollout_target_finish_chain_bonus_pressure_black": draw_rollout["target_finish_chain_bonus_pressure"]["B"],
            "draw_rollout_target_finish_chain_bonus_pressure_white": draw_rollout["target_finish_chain_bonus_pressure"]["W"],
            "draw_rollout_target_finish_chain_continuation_pressure_black": draw_rollout["target_finish_chain_continuation_pressure"]["B"],
            "draw_rollout_target_finish_chain_continuation_pressure_white": draw_rollout["target_finish_chain_continuation_pressure"]["W"],
            "draw_rollout_joint_collapse_signal_pressure_black": draw_rollout["joint_collapse_signal_pressure"]["B"],
            "draw_rollout_joint_collapse_signal_pressure_white": draw_rollout["joint_collapse_signal_pressure"]["W"],
            "draw_rollout_joint_collapse_bonus_pressure_black": draw_rollout["joint_collapse_bonus_pressure"]["B"],
            "draw_rollout_joint_collapse_bonus_pressure_white": draw_rollout["joint_collapse_bonus_pressure"]["W"],
            "draw_rollout_joint_collapse_continuation_pressure_black": draw_rollout["joint_collapse_continuation_pressure"]["B"],
            "draw_rollout_joint_collapse_continuation_pressure_white": draw_rollout["joint_collapse_continuation_pressure"]["W"],
            "draw_rollout_public_reveal_bridge_signal_pressure_black": draw_rollout["public_reveal_bridge_signal_pressure"]["B"],
            "draw_rollout_public_reveal_bridge_signal_pressure_white": draw_rollout["public_reveal_bridge_signal_pressure"]["W"],
            "draw_rollout_public_reveal_bridge_bonus_pressure_black": draw_rollout["public_reveal_bridge_bonus_pressure"]["B"],
            "draw_rollout_public_reveal_bridge_bonus_pressure_white": draw_rollout["public_reveal_bridge_bonus_pressure"]["W"],
            "draw_rollout_public_reveal_bridge_continuation_pressure_black": draw_rollout["public_reveal_bridge_continuation_pressure"]["B"],
            "draw_rollout_public_reveal_bridge_continuation_pressure_white": draw_rollout["public_reveal_bridge_continuation_pressure"]["W"],
            "draw_rollout_expected_win_probability_black": draw_rollout["expected_win_probability"]["B"],
            "draw_rollout_expected_win_probability_white": draw_rollout["expected_win_probability"]["W"],
            "draw_rollout_expected_attackability_after_hit_black": draw_rollout["expected_attackability_after_hit"]["B"],
            "draw_rollout_expected_attackability_after_hit_white": draw_rollout["expected_attackability_after_hit"]["W"],
            "draw_rollout_win_probability_pressure_black": draw_rollout["win_probability_pressure"]["B"],
            "draw_rollout_win_probability_pressure_white": draw_rollout["win_probability_pressure"]["W"],
            "draw_rollout_attackability_pressure_black": draw_rollout["attackability_pressure"]["B"],
            "draw_rollout_attackability_pressure_white": draw_rollout["attackability_pressure"]["W"],
            "draw_rollout_target_retention_ratio_black": draw_rollout["target_retention_ratio"]["B"],
            "draw_rollout_target_retention_ratio_white": draw_rollout["target_retention_ratio"]["W"],
            "draw_rollout_color_alignment_ratio_black": draw_rollout["color_alignment_ratio"]["B"],
            "draw_rollout_color_alignment_ratio_white": draw_rollout["color_alignment_ratio"]["W"],
            "draw_rollout_target_retention_pressure_black": draw_rollout["target_retention_pressure"]["B"],
            "draw_rollout_target_retention_pressure_white": draw_rollout["target_retention_pressure"]["W"],
            "draw_rollout_color_alignment_pressure_black": draw_rollout["color_alignment_pressure"]["B"],
            "draw_rollout_color_alignment_pressure_white": draw_rollout["color_alignment_pressure"]["W"],
            "draw_rollout_expected_best_gap_black": draw_rollout["expected_best_gap"]["B"],
            "draw_rollout_expected_best_gap_white": draw_rollout["expected_best_gap"]["W"],
            "draw_rollout_expected_strategy_objective_black": draw_rollout_expected_strategy_objective["B"],
            "draw_rollout_expected_strategy_objective_white": draw_rollout_expected_strategy_objective["W"],
            "draw_rollout_active_opening_ratio_black": draw_rollout["active_opening_ratio"]["B"],
            "draw_rollout_active_opening_ratio_white": draw_rollout["active_opening_ratio"]["W"],
            "draw_rollout_best_gap_pressure_black": draw_rollout["best_gap_pressure"]["B"],
            "draw_rollout_best_gap_pressure_white": draw_rollout["best_gap_pressure"]["W"],
            "draw_rollout_strategy_objective_pressure_black": draw_rollout_strategy_objective_pressure["B"],
            "draw_rollout_strategy_objective_pressure_white": draw_rollout_strategy_objective_pressure["W"],
            "draw_rollout_active_opening_pressure_black": draw_rollout["active_opening_pressure"]["B"],
            "draw_rollout_active_opening_pressure_white": draw_rollout["active_opening_pressure"]["W"],
            "draw_rollout_best_value_stddev_black": draw_rollout["best_value_stddev"]["B"],
            "draw_rollout_best_value_stddev_white": draw_rollout["best_value_stddev"]["W"],
            "draw_rollout_win_probability_stddev_black": draw_rollout["win_probability_stddev"]["B"],
            "draw_rollout_win_probability_stddev_white": draw_rollout["win_probability_stddev"]["W"],
            "draw_rollout_best_value_floor_black": draw_rollout["best_value_floor"]["B"],
            "draw_rollout_best_value_floor_white": draw_rollout["best_value_floor"]["W"],
            "draw_rollout_win_probability_floor_black": draw_rollout["win_probability_floor"]["B"],
            "draw_rollout_win_probability_floor_white": draw_rollout["win_probability_floor"]["W"],
            "draw_rollout_expected_information_gain_black": draw_rollout["expected_information_gain"]["B"],
            "draw_rollout_expected_information_gain_white": draw_rollout["expected_information_gain"]["W"],
            "draw_rollout_information_gain_floor_black": draw_rollout["information_gain_floor"]["B"],
            "draw_rollout_information_gain_floor_white": draw_rollout["information_gain_floor"]["W"],
            "draw_rollout_best_value_stability_pressure_black": draw_rollout["best_value_stability_pressure"]["B"],
            "draw_rollout_best_value_stability_pressure_white": draw_rollout["best_value_stability_pressure"]["W"],
            "draw_rollout_win_probability_stability_pressure_black": draw_rollout["win_probability_stability_pressure"]["B"],
            "draw_rollout_win_probability_stability_pressure_white": draw_rollout["win_probability_stability_pressure"]["W"],
            "draw_rollout_best_value_floor_pressure_black": draw_rollout["best_value_floor_pressure"]["B"],
            "draw_rollout_best_value_floor_pressure_white": draw_rollout["best_value_floor_pressure"]["W"],
            "draw_rollout_win_probability_floor_pressure_black": draw_rollout["win_probability_floor_pressure"]["B"],
            "draw_rollout_win_probability_floor_pressure_white": draw_rollout["win_probability_floor_pressure"]["W"],
            "draw_rollout_information_gain_pressure_black": draw_rollout["information_gain_pressure"]["B"],
            "draw_rollout_information_gain_pressure_white": draw_rollout["information_gain_pressure"]["W"],
            "draw_rollout_information_gain_floor_pressure_black": draw_rollout["information_gain_floor_pressure"]["B"],
            "draw_rollout_information_gain_floor_pressure_white": draw_rollout["information_gain_floor_pressure"]["W"],
            "draw_rollout_opening_target_player_id_black": black_opening_plan["target_player_id"],
            "draw_rollout_opening_target_player_id_white": white_opening_plan["target_player_id"],
            "draw_rollout_opening_target_slot_black": black_opening_plan["target_slot_index"],
            "draw_rollout_opening_target_slot_white": white_opening_plan["target_slot_index"],
            "draw_rollout_opening_guess_card_black": serialize_card(black_opening_plan["guess_card"])
            if isinstance(black_opening_plan["guess_card"], tuple)
            and len(black_opening_plan["guess_card"]) == 2
            else None,
            "draw_rollout_opening_guess_card_white": serialize_card(white_opening_plan["guess_card"])
            if isinstance(white_opening_plan["guess_card"], tuple)
            and len(white_opening_plan["guess_card"]) == 2
            else None,
            "draw_rollout_opening_support_black": black_opening_plan["support_ratio"],
            "draw_rollout_opening_support_white": white_opening_plan["support_ratio"],
            "draw_rollout_opening_guess_support_black": black_opening_plan["guess_support_ratio"],
            "draw_rollout_opening_guess_support_white": white_opening_plan["guess_support_ratio"],
            "draw_rollout_opening_expected_value_black": black_opening_plan["expected_value"],
            "draw_rollout_opening_expected_value_white": white_opening_plan["expected_value"],
            "draw_rollout_opening_strategy_objective_black": float(
                black_opening_plan.get("strategy_objective", 0.0)
            ),
            "draw_rollout_opening_strategy_objective_white": float(
                white_opening_plan.get("strategy_objective", 0.0)
            ),
            "draw_rollout_opening_win_probability_black": black_opening_plan["win_probability"],
            "draw_rollout_opening_win_probability_white": white_opening_plan["win_probability"],
            "draw_rollout_opening_information_gain_black": black_opening_plan["information_gain"],
            "draw_rollout_opening_information_gain_white": white_opening_plan["information_gain"],
            "draw_rollout_opening_continuation_likelihood_black": black_opening_plan["continuation_likelihood"],
            "draw_rollout_opening_continuation_likelihood_white": white_opening_plan["continuation_likelihood"],
            "draw_rollout_sample_count_black": draw_rollout["sample_count"]["B"],
            "draw_rollout_sample_count_white": draw_rollout["sample_count"]["W"],
            "draw_rollout_edge_scale": draw_rollout_edge_scale,
            "draw_rollout_target_gate": draw_rollout_target_gate,
            "draw_rollout_phase_gate": draw_rollout_phase_gate,
            "draw_rollout_plan_gate": draw_rollout_plan_gate,
            "draw_rollout_activation_scale": draw_rollout_activation_scale,
            "offense_pressure_black": offense_pressure["B"],
            "offense_pressure_white": offense_pressure["W"],
            "entropy_pressure_black": entropy_pressure["B"],
            "entropy_pressure_white": entropy_pressure["W"],
            "target_entropy_pressure_black": target_entropy_pressure["B"],
            "target_entropy_pressure_white": target_entropy_pressure["W"],
            "target_boundary_pressure_black": target_boundary_pressure["B"],
            "target_boundary_pressure_white": target_boundary_pressure["W"],
            "target_finish_pressure_black": target_finish_pressure["B"],
            "target_finish_pressure_white": target_finish_pressure["W"],
            "target_focus_pressure_black": target_focus_pressure["B"],
            "target_focus_pressure_white": target_focus_pressure["W"],
            "target_recent_momentum_pressure_black": target_recent_momentum_pressure["B"],
            "target_recent_momentum_pressure_white": target_recent_momentum_pressure["W"],
            "self_flexibility_pressure_black": self_flexibility_pressure["B"],
            "self_flexibility_pressure_white": self_flexibility_pressure["W"],
            "target_attack_pressure_black": target_attack_pressure["B"],
            "target_attack_pressure_white": target_attack_pressure["W"],
            "availability_pressure_black": availability_pressure["B"],
            "availability_pressure_white": availability_pressure["W"],
            "available_black_count": available_counts["B"],
            "available_white_count": available_counts["W"],
            "self_black_count": self_counts["B"],
            "self_white_count": self_counts["W"],
            "defense_guard_factor": defense_guard_factor,
            "target_attack_window_factor": target_attack_window_factor,
            "dominant_factor": max(
                dominant_factor_margins,
                key=lambda factor: round(dominant_factor_margins[factor], 8),
            ),
        }

    def run_turn(
        self,
        *,
        include_draw_color_summary: bool = True,
    ) -> Dict[str, Any]:
        strategy_phase = self._strategy_phase()
        has_any_hidden_slots = bool(self.inference_engine.search_positions) or any(
            self.inference_engine.preassigned_hidden.values()
        )
        target_hidden_slots = self.game_state.target_hidden_slots()
        blocked_target_slots = {key for key in self.inference_engine.publicly_collapsed_slots if key[0] == getattr(self.game_state, "target_player_id", None)}
        my_hidden_count = self.game_state.my_hidden_count()
        default_risk = self.decision_engine.calculate_risk_factor(my_hidden_count)
        draw_color_summary = (
            self._build_draw_color_summary()
            if include_draw_color_summary
            else {}
        )
        draw_opening_plan = (
            self._build_draw_opening_plan(draw_color_summary)
            if include_draw_color_summary
            else {}
        )

        if not has_any_hidden_slots:
            return {
                "best_move": None,
                "top_moves": [],
                "probability_matrix": [],
                "full_probability_matrix": [],
                "hard_probability_matrix": [],
                "hard_full_probability_matrix": [],
                "soft_full_probability_matrix": [],
                "search_space_size": 0,
                "effective_weight_sum": 0.0,
                "opponent_hidden_count": 0,
                "risk_factor": default_risk,
                "behavior_debug": {
                    "hypothesis_source": "target_slot_top_k_posterior_with_map_context",
                    "aggregation_top_k": self.BEHAVIOR_DEBUG_TOP_K,
                    "signal_count": 0,
                    "map_signals": [],
                    "signals": [],
                },
                "behavior_guidance_profile": {
                    "signal_count": 0.0,
                    "average_posterior_support": 0.0,
                    "average_weighted_strength": 0.0,
                    "stable_signal_ratio": 0.0,
                    "guidance_multiplier": 1.0,
                    "source_support_progressive": 0.0,
                    "source_support_same_color_anchor": 0.0,
                    "source_support_local_boundary": 0.0,
                },
                "draw_color_summary": draw_color_summary,
                "draw_opening_plan": draw_opening_plan,
                "strategy_action_summary": self._build_strategy_action_summary(
                    decision_summary={},
                    draw_color_summary=draw_color_summary,
                    draw_opening_plan=draw_opening_plan,
                    phase=strategy_phase,
                ),
                "strategy_rollout_depth": 0,
                "recommended_action": self._build_strategy_action_summary(
                    decision_summary={},
                    draw_color_summary=draw_color_summary,
                    draw_opening_plan=draw_opening_plan,
                    phase=strategy_phase,
                )["recommended_action"],
                "recommended_draw_color": (
                    draw_color_summary.get("recommended_color")
                    if strategy_phase == "pre_draw"
                    else None
                ),
                "strategy_phase": strategy_phase,
                "should_stop": True,
            }

        guess_signals_by_player = self.behavior_model.build_guess_signals(self.game_state)
        hard_full_probability_matrix, soft_full_probability_matrix, full_probability_matrix, search_space_size, total_soft_weight = self.inference_engine.infer_hidden_probabilities(
            guess_signals_by_player,
            self.behavior_model,
        )
        behavior_map_hypothesis = self._map_hypothesis_from_matrix(full_probability_matrix)
        behavior_debug = self._build_behavior_debug(
            full_probability_matrix=full_probability_matrix,
            guess_signals_by_player=guess_signals_by_player,
            map_hypothesis=behavior_map_hypothesis,
        )
        behavior_guidance_profile = self._build_behavior_guidance_profile(
            behavior_debug_signals=behavior_debug["signals"],
            acting_player_id=self.game_state.self_player_id,
        )
        draw_color_summary = (
            self._build_draw_color_summary(full_probability_matrix)
            if include_draw_color_summary
            else {}
        )
        draw_opening_plan = (
            self._build_draw_opening_plan(draw_color_summary)
            if include_draw_color_summary
            else {}
        )

        hidden_index_by_player = {
            player_id: self.game_state.hidden_index_by_slot(player_id)
            for player_id in full_probability_matrix
        }
        rollout_depth = self._select_strategy_rollout_depth(
            search_space_size=float(search_space_size),
            my_hidden_count=my_hidden_count,
            target_hidden_count=max(
                0,
                len(target_hidden_slots) - len(blocked_target_slots),
            ),
        )
        all_moves, risk_factor = self.decision_engine.evaluate_all_moves(
            full_probability_matrix=full_probability_matrix,
            my_hidden_count=my_hidden_count,
            hidden_index_by_player=hidden_index_by_player,
            behavior_model=self.behavior_model,
            guess_signals_by_player=guess_signals_by_player,
            acting_player_id=self.game_state.self_player_id,
            behavior_guidance_profile=behavior_guidance_profile,
            game_state=self.game_state,
            behavior_map_hypothesis=behavior_map_hypothesis,
            blocked_slots=self.inference_engine.publicly_collapsed_slots,
            rollout_depth=rollout_depth,
        )
        best_move, decision_summary = self.decision_engine.choose_best_move(
            all_moves,
            risk_factor=risk_factor,
            my_hidden_count=my_hidden_count,
        )
        strategy_action_summary = self._build_strategy_action_summary(
            decision_summary=decision_summary,
            draw_color_summary=draw_color_summary,
            draw_opening_plan=draw_opening_plan,
            phase=strategy_phase,
        )

        target_player_id = getattr(self.game_state, "target_player_id", None)
        target_probability_matrix = full_probability_matrix.get(target_player_id, {})
        target_hard_matrix = hard_full_probability_matrix.get(target_player_id, {})

        return {
            "best_move": best_move,
            "top_moves": all_moves[:10],
            "probability_matrix": self._serialize_probability_matrix(
                target_probability_matrix,
                self.game_state.hidden_index_by_slot(target_player_id) if target_player_id is not None else {},
            ),
            "full_probability_matrix": self._serialize_full_probability_matrix(
                full_probability_matrix,
            ),
            "hard_probability_matrix": self._serialize_probability_matrix(
                target_hard_matrix,
                self.game_state.hidden_index_by_slot(target_player_id) if target_player_id is not None else {},
            ),
            "hard_full_probability_matrix": self._serialize_full_probability_matrix(
                hard_full_probability_matrix,
            ),
            "soft_full_probability_matrix": self._serialize_full_probability_matrix(
                soft_full_probability_matrix,
            ),
            "search_space_size": search_space_size,
            "opponent_hidden_count": max(0, len(target_hidden_slots) - len(blocked_target_slots)),
            "risk_factor": risk_factor,
            "effective_weight_sum": total_soft_weight,
            "behavior_blend": SOFT_BEHAVIOR_BLEND,
            "behavior_debug": behavior_debug,
            "behavior_guidance_profile": behavior_guidance_profile,
            "draw_color_summary": draw_color_summary,
            "draw_opening_plan": draw_opening_plan,
            "decision_summary": decision_summary,
            "strategy_action_summary": strategy_action_summary,
            "strategy_rollout_depth": rollout_depth,
            "strategy_phase": strategy_phase,
            "recommended_action": strategy_action_summary["recommended_action"],
            "recommended_draw_color": (
                "B"
                if strategy_action_summary["recommended_action"] == "draw_black"
                else "W"
                if strategy_action_summary["recommended_action"] == "draw_white"
                else None
            ),
            "should_stop": best_move is None,
        }

    def _strategy_phase(self) -> str:
        self_slots = self.game_state.resolved_ordered_slots(self.game_state.self_player_id)
        if any(getattr(slot, "is_newly_drawn", False) for slot in self_slots):
            latest_self_action = next(
                (
                    action
                    for action in reversed(self.game_state.actions)
                    if getattr(action, "guesser_id", None) == self.game_state.self_player_id
                ),
                None,
            )
            if latest_self_action is not None and getattr(latest_self_action, "result", False):
                return "post_hit_chain"
            return "post_draw_opening"
        return "pre_draw"

    def _select_strategy_rollout_depth(
        self,
        *,
        search_space_size: float,
        my_hidden_count: int,
        target_hidden_count: int,
    ) -> int:
        if (
            target_hidden_count
            <= self.decision_engine.DEEP_ROLLOUT_TARGET_HIDDEN_THRESHOLD
            and my_hidden_count
            <= self.decision_engine.DEEP_ROLLOUT_SELF_HIDDEN_THRESHOLD
            and search_space_size
            <= self.decision_engine.DEEP_ROLLOUT_SEARCH_SPACE_THRESHOLD
        ):
            return self.decision_engine.DEEP_ROLLOUT_DEPTH
        return 1

    def _build_strategy_action_summary(
        self,
        *,
        decision_summary: Optional[Dict[str, Any]],
        draw_color_summary: Optional[Dict[str, Any]],
        draw_opening_plan: Optional[Dict[str, Any]] = None,
        phase: str,
    ) -> Dict[str, Any]:
        decision_summary = decision_summary or {}
        draw_color_summary = draw_color_summary or {}
        draw_opening_plan = draw_opening_plan or {}
        action_scores = {
            "guess": float(
                decision_summary.get(
                    "strategy_objective_guess",
                    decision_summary.get("best_strategy_objective", 0.0),
                )
            ),
            "draw_black": float(
                draw_color_summary.get(
                    "draw_rollout_expected_strategy_objective_black",
                    0.0,
                )
            ),
            "draw_white": float(
                draw_color_summary.get(
                    "draw_rollout_expected_strategy_objective_white",
                    0.0,
                )
            ),
            "stop": float(
                decision_summary.get(
                    "strategy_objective_stop",
                    decision_summary.get("stop_score", 0.0),
                )
            ),
        }
        tree_action_scores = dict(action_scores)
        tree_action_scores["guess"] += (
            0.10
            * max(
                float(decision_summary.get("best_post_hit_expectimax_signal", 0.0)),
                float(decision_summary.get("best_post_hit_mcts_signal", 0.0)),
                float(decision_summary.get("best_post_hit_branch_search_signal", 0.0)),
            )
        )
        for action, color_suffix in (("draw_black", "black"), ("draw_white", "white")):
            tree_action_scores[action] += (
                0.10
                * float(
                    draw_color_summary.get(
                        f"draw_rollout_opening_strategy_objective_{color_suffix}",
                        0.0,
                    )
                )
            )
            tree_action_scores[action] += (
                0.06
                * float(
                    draw_color_summary.get(
                        f"draw_rollout_opening_support_{color_suffix}",
                        0.0,
                    )
                )
            )
            tree_action_scores[action] += (
                0.05
                * float(
                    draw_color_summary.get(
                        f"draw_rollout_opening_guess_support_{color_suffix}",
                        0.0,
                    )
                )
            )
        if draw_opening_plan:
            recommended_color = draw_opening_plan.get("recommended_color")
            if recommended_color == "B":
                tree_action_scores["draw_black"] += (
                    0.05 * float(draw_opening_plan.get("strategy_objective", 0.0))
                )
            elif recommended_color == "W":
                tree_action_scores["draw_white"] += (
                    0.05 * float(draw_opening_plan.get("strategy_objective", 0.0))
                )
        if phase == "pre_draw":
            allowed_actions = ["draw_black", "draw_white"]
        elif phase == "post_draw_opening":
            allowed_actions = ["guess"]
        else:
            allowed_actions = ["guess", "stop"]
        recommended_action = max(
            allowed_actions,
            key=lambda action: (
                round(tree_action_scores[action], 8),
                round(action_scores[action], 8),
                1 if action == "guess" else 0,
            ),
        )
        return {
            **action_scores,
            "guess_tree": tree_action_scores["guess"],
            "draw_black_tree": tree_action_scores["draw_black"],
            "draw_white_tree": tree_action_scores["draw_white"],
            "stop_tree": tree_action_scores["stop"],
            "phase": phase,
            "allowed_actions": allowed_actions,
            "recommended_action": recommended_action,
        }

    def _serialize_probability_matrix(
        self,
        probability_matrix: ProbabilityMatrix,
        hidden_index_by_slot: Dict[int, int],
    ) -> List[Dict[str, Any]]:
        serialized_positions: List[Dict[str, Any]] = []
        for target_slot_index in sorted(probability_matrix):
            sorted_candidates = sorted(
                probability_matrix[target_slot_index].items(),
                key=lambda item: (-item[1], card_sort_key(item[0])),
            )
            serialized_positions.append(
                {
                    "target_index": hidden_index_by_slot.get(target_slot_index, target_slot_index),
                    "target_slot_index": target_slot_index,
                    "target_scope": "player_slots",
                    "candidates": [
                        {
                            "card": serialize_card(card),
                            "probability": probability,
                        }
                        for card, probability in sorted_candidates
                    ],
                }
            )
        return serialized_positions

    def _serialize_full_probability_matrix(
        self,
        full_probability_matrix: FullProbabilityMatrix,
    ) -> List[Dict[str, Any]]:
        serialized_players: List[Dict[str, Any]] = []
        for player_id in full_probability_matrix:
            hidden_index_by_slot = self.game_state.hidden_index_by_slot(player_id)
            serialized_players.append(
                {
                    "player_id": player_id,
                    "positions": self._serialize_probability_matrix(
                        full_probability_matrix[player_id],
                        hidden_index_by_slot,
                    ),
                }
            )
        return serialized_players

    def _map_hypothesis_from_matrix(
        self,
        full_probability_matrix: FullProbabilityMatrix,
    ) -> Dict[str, Dict[int, Card]]:
        hypothesis_by_player: Dict[str, Dict[int, Card]] = {}
        for player_id, probability_matrix in full_probability_matrix.items():
            player_hypothesis: Dict[int, Card] = {}
            for slot_index, slot_distribution in probability_matrix.items():
                if not slot_distribution:
                    continue
                best_card = max(
                    slot_distribution.items(),
                    key=lambda item: (item[1], -card_sort_key(item[0])[0], -card_sort_key(item[0])[1]),
                )[0]
                player_hypothesis[slot_index] = best_card
            if player_hypothesis:
                hypothesis_by_player[player_id] = player_hypothesis
        return hypothesis_by_player

    def _build_behavior_debug(
        self,
        *,
        full_probability_matrix: FullProbabilityMatrix,
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        map_hypothesis: Optional[Dict[str, Dict[int, Card]]] = None,
    ) -> Dict[str, Any]:
        map_hypothesis = map_hypothesis or self._map_hypothesis_from_matrix(full_probability_matrix)
        map_signals = self.behavior_model.explain_guess_signals(
            map_hypothesis,
            guess_signals_by_player,
            self.game_state,
        )
        aggregated_signals = [
            self._aggregate_behavior_signal_debug(
                signal=signal,
                full_probability_matrix=full_probability_matrix,
                map_hypothesis=map_hypothesis,
            )
            for player_id in sorted(guess_signals_by_player)
            for signal in guess_signals_by_player[player_id]
        ]
        return {
            "hypothesis_source": "target_slot_top_k_posterior_with_map_context",
            "aggregation_top_k": self.BEHAVIOR_DEBUG_TOP_K,
            "signal_count": sum(len(signals) for signals in guess_signals_by_player.values()),
            "map_signals": map_signals,
            "signals": aggregated_signals,
        }

    def _aggregate_behavior_signal_debug(
        self,
        *,
        signal: GuessSignal,
        full_probability_matrix: FullProbabilityMatrix,
        map_hypothesis: Dict[str, Dict[int, Card]],
    ) -> Dict[str, Any]:
        map_explanation = self.behavior_model.explain_signal(
            map_hypothesis,
            self.game_state,
            signal,
        )
        slot_distribution = (
            full_probability_matrix
            .get(signal.target_player_id, {})
            .get(signal.target_slot_index, {})
        )
        ranked_candidates = sorted(
            slot_distribution.items(),
            key=lambda item: (-item[1], card_sort_key(item[0])),
        )[: self.BEHAVIOR_DEBUG_TOP_K]
        covered_probability = sum(probability for _, probability in ranked_candidates)
        if not ranked_candidates or covered_probability <= 0.0:
            fallback_value_selection = dict(map_explanation["value_selection"])
            fallback_value_selection.update(
                {
                    "mode": "map_only_fallback",
                    "covered_probability": 0.0,
                    "candidate_count": 0,
                    "reason_support": [],
                    "source_support": [],
                }
            )
            return {
                **map_explanation,
                "aggregation_mode": "map_only_fallback",
                "candidate_count": 0,
                "covered_probability": 0.0,
                "candidate_explanations": [],
                "map_explanation": map_explanation,
                "value_selection": fallback_value_selection,
            }

        normalized_total = max(self.behavior_model.EPSILON, covered_probability)
        weighted_component_sums: DefaultDict[str, float] = defaultdict(float)
        weighted_value_component_sums: DefaultDict[str, float] = defaultdict(float)
        reason_support: DefaultDict[str, float] = defaultdict(float)
        source_support: DefaultDict[str, float] = defaultdict(float)
        source_strength: DefaultDict[str, float] = defaultdict(float)
        candidate_explanations: List[Dict[str, Any]] = []

        for card, probability in ranked_candidates:
            normalized_weight = probability / normalized_total
            candidate_hypothesis = self._replace_hypothesis_card(
                map_hypothesis,
                signal.target_player_id,
                signal.target_slot_index,
                card,
            )
            explanation = self.behavior_model.explain_signal(
                candidate_hypothesis,
                self.game_state,
                signal,
            )
            candidate_explanations.append(
                {
                    "hypothesis_target_card": serialize_card(card),
                    "posterior_probability": probability,
                    "normalized_weight": normalized_weight,
                    "dominant_signal": explanation["value_selection"]["dominant_signal"],
                    "signal_tags": explanation["value_selection"]["signal_tags"],
                    "value_selection_total_weight": explanation["value_selection"]["total_weight"],
                }
            )
            for component_name, weight in explanation["component_weights"].items():
                weighted_component_sums[component_name] += normalized_weight * weight

            value_selection = explanation["value_selection"]
            weighted_value_component_sums["total_weight"] += normalized_weight * value_selection["total_weight"]
            weighted_value_component_sums["progressive"] += normalized_weight * value_selection["progressive"]["weight"]
            weighted_value_component_sums["anchor"] += normalized_weight * value_selection["anchor"]["weight"]
            weighted_value_component_sums["boundary"] += normalized_weight * value_selection["boundary"]["weight"]
            for reason in value_selection["signal_tags"]:
                reason_support[reason] += normalized_weight

            dominant_signal = value_selection["dominant_signal"]
            source = str(dominant_signal.get("source", "neutral"))
            source_support[source] += normalized_weight
            source_strength[source] += normalized_weight * abs(float(dominant_signal.get("weight", 1.0)) - 1.0)

        aggregated_value_selection = {
            "mode": "target_slot_top_k_posterior",
            "covered_probability": covered_probability,
            "candidate_count": len(candidate_explanations),
            "total_weight": weighted_value_component_sums["total_weight"],
            "expected_component_weights": {
                "progressive": weighted_value_component_sums["progressive"],
                "anchor": weighted_value_component_sums["anchor"],
                "boundary": weighted_value_component_sums["boundary"],
            },
            "reason_support": [
                {
                    "reason": reason,
                    "posterior_support": support,
                }
                for reason, support in sorted(
                    reason_support.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
            "source_support": [
                {
                    "source": source,
                    "posterior_support": source_support[source],
                    "weighted_strength": source_strength[source],
                }
                for source in sorted(
                    source_support,
                    key=lambda item: (-source_strength[item], -source_support[item], item),
                )
            ],
        }
        aggregated_value_selection["dominant_signal"] = self._dominant_posterior_signal(
            aggregated_value_selection["source_support"],
            aggregated_value_selection["reason_support"],
        )

        return {
            **map_explanation,
            "aggregation_mode": "target_slot_top_k_posterior",
            "candidate_count": len(candidate_explanations),
            "covered_probability": covered_probability,
            "candidate_explanations": candidate_explanations,
            "component_weights": dict(weighted_component_sums),
            "map_explanation": map_explanation,
            "value_selection": aggregated_value_selection,
        }

    def _replace_hypothesis_card(
        self,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        player_id: str,
        slot_index: int,
        card: Card,
    ) -> Dict[str, Dict[int, Card]]:
        cloned = {
            hypothesis_player_id: dict(cards_by_slot)
            for hypothesis_player_id, cards_by_slot in hypothesis_by_player.items()
        }
        player_hypothesis = cloned.setdefault(player_id, {})
        player_hypothesis[slot_index] = card
        return cloned

    def _dominant_posterior_signal(
        self,
        source_support: Sequence[Dict[str, Any]],
        reason_support: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not source_support:
            return {
                "source": "neutral",
                "reason": "neutral",
                "posterior_support": 0.0,
                "weighted_strength": 0.0,
            }

        strongest_source = source_support[0]
        strongest_reason = reason_support[0]["reason"] if reason_support else "neutral"
        strongest_reason_support = reason_support[0]["posterior_support"] if reason_support else 0.0
        return {
            "source": strongest_source["source"],
            "reason": strongest_reason,
            "posterior_support": strongest_source["posterior_support"],
            "weighted_strength": strongest_source["weighted_strength"],
            "reason_support": strongest_reason_support,
        }

    def _build_behavior_guidance_profile(
        self,
        *,
        behavior_debug_signals: Sequence[Dict[str, Any]],
        acting_player_id: Optional[str],
    ) -> Dict[str, float]:
        if acting_player_id is None:
            return {
                "signal_count": 0.0,
                "average_posterior_support": 0.0,
                "average_weighted_strength": 0.0,
                "stable_signal_ratio": 0.0,
                "guidance_multiplier": 1.0,
                "source_support_progressive": 0.0,
                "source_support_same_color_anchor": 0.0,
                "source_support_local_boundary": 0.0,
            }

        relevant_signals = [
            signal
            for signal in behavior_debug_signals
            if signal.get("guesser_id") == acting_player_id
        ]
        if not relevant_signals:
            return {
                "signal_count": 0.0,
                "average_posterior_support": 0.0,
                "average_weighted_strength": 0.0,
                "stable_signal_ratio": 0.0,
                "guidance_multiplier": 1.0,
                "source_support_progressive": 0.0,
                "source_support_same_color_anchor": 0.0,
                "source_support_local_boundary": 0.0,
            }

        signal_count = float(len(relevant_signals))
        average_posterior_support = sum(
            float(signal["value_selection"]["dominant_signal"].get("posterior_support", 0.0))
            for signal in relevant_signals
        ) / signal_count
        average_weighted_strength = sum(
            float(signal["value_selection"]["dominant_signal"].get("weighted_strength", 0.0))
            for signal in relevant_signals
        ) / signal_count
        stable_signal_ratio = sum(
            1.0
            for signal in relevant_signals
            if float(signal["value_selection"]["dominant_signal"].get("posterior_support", 0.0)) >= 0.55
        ) / signal_count
        source_support_sums: DefaultDict[str, float] = defaultdict(float)
        for signal in relevant_signals:
            for source_support in signal["value_selection"].get("source_support", []):
                source_support_sums[str(source_support.get("source", "neutral"))] += float(
                    source_support.get("posterior_support", 0.0)
                )
        guidance_multiplier = clamp(
            0.95
            + (0.08 * average_posterior_support)
            + (0.07 * stable_signal_ratio)
            + (0.14 * average_weighted_strength),
            0.93,
            1.12,
        )
        return {
            "signal_count": signal_count,
            "average_posterior_support": average_posterior_support,
            "average_weighted_strength": average_weighted_strength,
            "stable_signal_ratio": stable_signal_ratio,
            "guidance_multiplier": guidance_multiplier,
            "source_support_progressive": source_support_sums["progressive"] / signal_count,
            "source_support_same_color_anchor": source_support_sums["same_color_anchor"] / signal_count,
            "source_support_local_boundary": source_support_sums["local_boundary"] / signal_count,
        }
