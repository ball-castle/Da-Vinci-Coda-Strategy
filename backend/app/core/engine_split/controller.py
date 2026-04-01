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
from .models import SearchTreeNode
from .behavior import BehavioralLikelihoodModel
from .inference import DaVinciInferenceEngine
from .decision import DaVinciDecisionEngine

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
