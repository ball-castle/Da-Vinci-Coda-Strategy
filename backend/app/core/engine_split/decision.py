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
from .models import SearchTreeNode, GuessSignal, FullProbabilityMatrix, ProbabilityMatrix, SlotKey, SOFT_BEHAVIOR_BLEND
from .utils import numeric_card_value, card_sort_key, clamp
from .constraints import HardConstraintCompiler
from .behavior import BehavioralLikelihoodModel
from .inference import DaVinciInferenceEngine

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


