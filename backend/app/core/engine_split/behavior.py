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
from .models import (
    CARD_COLOR_RANK, SOFT_BEHAVIOR_BLEND,
    HiddenPosition, HardConstraintSet, GuessSignal,
    ProbabilityMatrix, FullProbabilityMatrix, SlotKey
)
from .utils import numeric_card_value, card_sort_key, serialize_card, normalize_card_distribution, slot_key, clamp
from .constraints import HardConstraintCompiler

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

