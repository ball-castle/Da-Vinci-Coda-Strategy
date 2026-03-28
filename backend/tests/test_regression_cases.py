import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.engine import BehavioralLikelihoodModel, DaVinciDecisionEngine
from app.core.state import CardSlot, GameState, GuessAction, PlayerState


@dataclass(frozen=True)
class DecisionRegressionCase:
    name: str
    my_hidden_count: int
    moves: List[Dict[str, Any]]
    expect_continue: bool
    stop_reason_contains: Optional[str] = None
    positive_breakdown_key: Optional[str] = None
    min_continue_margin: Optional[float] = None
    max_continue_margin: Optional[float] = None


@dataclass(frozen=True)
class BehaviorRegressionCase:
    name: str
    preferred_state: GameState
    alternative_state: GameState
    preferred_hypothesis: Dict[str, Dict[int, tuple]]
    alternative_hypothesis: Dict[str, Dict[int, tuple]]


def build_move(**overrides: Any) -> Dict[str, Any]:
    move: Dict[str, Any] = {
        "expected_value": 0.0,
        "win_probability": 0.0,
        "continuation_value": 0.0,
        "continuation_likelihood": 0.0,
        "attackability_after_hit": 0.0,
        "behavior_match_bonus": 0.0,
        "behavior_match_support": 0.0,
        "behavior_guidance_stable_ratio": 0.0,
        "behavior_candidate_signal": None,
        "post_hit_continue_score": 0.0,
        "post_hit_stop_score": 0.0,
        "post_hit_continue_margin": 0.0,
        "post_hit_best_gap": 0.0,
        "post_hit_top_k_continue_margin": 0.0,
        "post_hit_top_k_support_ratio": 0.0,
    }
    move.update(overrides)
    return move


DECISION_REGRESSION_CASES = [
    DecisionRegressionCase(
        name="weak_endgame_stop",
        my_hidden_count=1,
        moves=[
            build_move(
                expected_value=0.62,
                win_probability=0.41,
                continuation_value=0.09,
                continuation_likelihood=0.44,
                attackability_after_hit=0.30,
            ),
            build_move(
                expected_value=0.55,
                win_probability=0.40,
                continuation_value=0.06,
                continuation_likelihood=0.42,
                attackability_after_hit=0.28,
            ),
        ],
        expect_continue=False,
        stop_reason_contains="建议停手",
        max_continue_margin=0.0,
    ),
    DecisionRegressionCase(
        name="strong_continuation_continue",
        my_hidden_count=2,
        moves=[
            build_move(
                expected_value=1.45,
                win_probability=0.58,
                continuation_value=0.72,
                continuation_likelihood=0.76,
                attackability_after_hit=0.68,
                post_hit_continue_score=0.92,
                post_hit_stop_score=0.54,
                post_hit_continue_margin=0.38,
                post_hit_best_gap=0.31,
                post_hit_top_k_continue_margin=0.29,
                post_hit_top_k_support_ratio=1.0,
            ),
            build_move(
                expected_value=0.71,
                win_probability=0.49,
                continuation_value=0.22,
                continuation_likelihood=0.55,
                attackability_after_hit=0.42,
                post_hit_continue_score=0.28,
                post_hit_stop_score=0.24,
                post_hit_continue_margin=0.04,
                post_hit_best_gap=0.09,
                post_hit_top_k_continue_margin=0.02,
                post_hit_top_k_support_ratio=0.67,
            ),
        ],
        expect_continue=True,
        min_continue_margin=0.0,
    ),
    DecisionRegressionCase(
        name="stable_behavior_match_continue",
        my_hidden_count=2,
        moves=[
            build_move(
                expected_value=0.66,
                win_probability=0.55,
                continuation_value=0.18,
                continuation_likelihood=0.58,
                attackability_after_hit=0.72,
                behavior_match_bonus=0.10,
                behavior_match_support=0.18,
                behavior_guidance_stable_ratio=1.0,
            ),
        ],
        expect_continue=True,
        positive_breakdown_key="behavior_match_decision_bonus",
        min_continue_margin=0.0,
    ),
    DecisionRegressionCase(
        name="fragile_candidate_confidence_stop",
        my_hidden_count=2,
        moves=[
            build_move(
                expected_value=0.66,
                win_probability=0.55,
                continuation_value=0.18,
                continuation_likelihood=0.58,
                attackability_after_hit=0.72,
                behavior_match_bonus=0.10,
                behavior_match_support=0.18,
                behavior_guidance_stable_ratio=1.0,
                behavior_candidate_signal={
                    "mode": "neighbor_top_k_posterior",
                    "context_covered_probability": 0.18,
                    "dominant_signal": {
                        "source": "local_boundary",
                        "reason": "narrow_boundary_probe",
                        "weight": 1.15,
                        "posterior_support": 0.12,
                    },
                },
            ),
        ],
        expect_continue=False,
        positive_breakdown_key="behavior_match_candidate_confidence",
        max_continue_margin=0.0,
    ),
    DecisionRegressionCase(
        name="sparse_component_support_stop",
        my_hidden_count=2,
        moves=[
            build_move(
                expected_value=0.624,
                win_probability=0.55,
                continuation_value=0.18,
                continuation_likelihood=0.58,
                attackability_after_hit=0.72,
                behavior_match_bonus=0.20,
                behavior_match_support=0.18,
                behavior_guidance_stable_ratio=1.0,
                behavior_candidate_signal={
                    "mode": "neighbor_top_k_posterior",
                    "context_covered_probability": 0.64,
                    "context_candidate_count": 4,
                    "dominant_signal": {
                        "source": "local_boundary",
                        "reason": "narrow_boundary_probe",
                        "weight": 1.15,
                        "posterior_support": 0.62,
                    },
                    "progressive": {
                        "weight": 1.0,
                        "reason": "neutral",
                        "posterior_support": 0.0,
                    },
                    "anchor": {
                        "weight": 1.0,
                        "reason": "neutral",
                        "posterior_support": 0.0,
                    },
                    "boundary": {
                        "weight": 1.15,
                        "reason": "narrow_boundary_probe",
                        "posterior_support": 0.62,
                    },
                },
            ),
        ],
        expect_continue=False,
        max_continue_margin=0.0,
    ),
    DecisionRegressionCase(
        name="weak_component_weight_strength_stop",
        my_hidden_count=2,
        moves=[
            build_move(
                expected_value=0.58,
                win_probability=0.55,
                continuation_value=0.18,
                continuation_likelihood=0.58,
                attackability_after_hit=0.72,
                behavior_match_bonus=0.30,
                behavior_match_support=0.18,
                behavior_guidance_stable_ratio=1.0,
                behavior_candidate_signal={
                    "mode": "neighbor_top_k_posterior",
                    "context_covered_probability": 0.68,
                    "context_candidate_count": 4,
                    "dominant_signal": {
                        "source": "local_boundary",
                        "reason": "narrow_boundary_probe",
                        "weight": 1.03,
                        "posterior_support": 0.70,
                    },
                    "progressive": {
                        "weight": 1.01,
                        "reason": "progressive_step",
                        "posterior_support": 0.70,
                    },
                    "anchor": {
                        "weight": 1.01,
                        "reason": "same_color_sandwich_exact",
                        "posterior_support": 0.70,
                    },
                    "boundary": {
                        "weight": 1.03,
                        "reason": "narrow_boundary_probe",
                        "posterior_support": 0.70,
                    },
                },
            ),
        ],
        expect_continue=False,
        positive_breakdown_key="behavior_match_component_strength",
        max_continue_margin=0.0,
    ),
    DecisionRegressionCase(
        name="negative_component_weight_penalty_stop",
        my_hidden_count=2,
        moves=[
            build_move(
                expected_value=0.57,
                win_probability=0.55,
                continuation_value=0.18,
                continuation_likelihood=0.58,
                attackability_after_hit=0.72,
                behavior_match_bonus=0.36,
                behavior_match_support=0.18,
                behavior_guidance_stable_ratio=1.0,
                behavior_candidate_signal={
                    "mode": "neighbor_top_k_posterior",
                    "context_covered_probability": 0.68,
                    "context_candidate_count": 4,
                    "dominant_signal": {
                        "source": "local_boundary",
                        "reason": "wide_gap_edge_hug",
                        "weight": 0.96,
                        "posterior_support": 0.70,
                    },
                    "progressive": {
                        "weight": 0.90,
                        "reason": "stalled_after_failure",
                        "posterior_support": 0.70,
                    },
                    "anchor": {
                        "weight": 0.92,
                        "reason": "wrong_direction",
                        "posterior_support": 0.70,
                    },
                    "boundary": {
                        "weight": 0.96,
                        "reason": "wide_gap_edge_hug",
                        "posterior_support": 0.70,
                    },
                },
            ),
        ],
        expect_continue=False,
        positive_breakdown_key="behavior_match_component_penalty",
        max_continue_margin=0.0,
    ),
    DecisionRegressionCase(
        name="diffuse_context_focus_stop",
        my_hidden_count=2,
        moves=[
            build_move(
                expected_value=0.648,
                win_probability=0.55,
                continuation_value=0.18,
                continuation_likelihood=0.58,
                attackability_after_hit=0.72,
                behavior_match_bonus=0.10,
                behavior_match_support=0.18,
                behavior_guidance_stable_ratio=1.0,
                behavior_candidate_signal={
                    "mode": "neighbor_top_k_posterior",
                    "context_covered_probability": 0.72,
                    "context_candidate_count": 16,
                    "dominant_signal": {
                        "source": "local_boundary",
                        "reason": "narrow_boundary_probe",
                        "weight": 1.15,
                        "posterior_support": 0.68,
                    },
                    "progressive": {
                        "weight": 1.02,
                        "reason": "retry_directional_probe",
                        "posterior_support": 0.68,
                    },
                    "anchor": {
                        "weight": 1.03,
                        "reason": "same_color_sandwich_exact",
                        "posterior_support": 0.68,
                    },
                    "boundary": {
                        "weight": 1.15,
                        "reason": "narrow_boundary_probe",
                        "posterior_support": 0.68,
                    },
                },
            ),
        ],
        expect_continue=False,
        max_continue_margin=0.0,
    ),
    DecisionRegressionCase(
        name="fragile_candidate_rollout_stop",
        my_hidden_count=2,
        moves=[
            build_move(
                expected_value=0.74,
                win_probability=0.57,
                continuation_value=0.23,
                continuation_likelihood=0.63,
                attackability_after_hit=0.76,
                behavior_match_bonus=0.06,
                behavior_match_support=0.18,
                behavior_guidance_stable_ratio=1.0,
                behavior_candidate_signal={
                    "mode": "neighbor_top_k_posterior",
                    "context_covered_probability": 0.18,
                    "dominant_signal": {
                        "source": "local_boundary",
                        "reason": "narrow_boundary_probe",
                        "weight": 1.15,
                        "posterior_support": 0.12,
                    },
                },
                post_hit_continue_score=0.38,
                post_hit_stop_score=0.24,
                post_hit_continue_margin=0.14,
                post_hit_best_gap=0.28,
                post_hit_top_k_continue_margin=0.14,
                post_hit_top_k_support_ratio=1.0,
            ),
        ],
        expect_continue=False,
        positive_breakdown_key="behavior_rollout_pressure",
        max_continue_margin=0.0,
    ),
    DecisionRegressionCase(
        name="negative_post_hit_margin_stop",
        my_hidden_count=2,
        moves=[
            build_move(
                expected_value=0.84,
                win_probability=0.56,
                continuation_value=0.19,
                continuation_likelihood=0.63,
                attackability_after_hit=0.72,
                post_hit_continue_score=0.22,
                post_hit_stop_score=0.58,
                post_hit_continue_margin=-0.36,
                post_hit_best_gap=0.02,
                post_hit_top_k_continue_margin=0.0,
                post_hit_top_k_support_ratio=0.0,
            ),
        ],
        expect_continue=False,
        positive_breakdown_key="rollout_pressure",
        max_continue_margin=0.0,
    ),
    DecisionRegressionCase(
        name="fragile_positive_margin_stop",
        my_hidden_count=2,
        moves=[
            build_move(
                expected_value=0.81,
                win_probability=0.57,
                continuation_value=0.16,
                continuation_likelihood=0.61,
                attackability_after_hit=0.74,
                post_hit_continue_score=0.35,
                post_hit_stop_score=0.24,
                post_hit_continue_margin=0.11,
                post_hit_best_gap=0.02,
                post_hit_top_k_continue_margin=0.05,
                post_hit_top_k_support_ratio=0.33,
            ),
        ],
        expect_continue=False,
        positive_breakdown_key="fragile_rollout_pressure",
        max_continue_margin=0.0,
    ),
    DecisionRegressionCase(
        name="weak_topk_followup_stop",
        my_hidden_count=2,
        moves=[
            build_move(
                expected_value=0.80,
                win_probability=0.57,
                continuation_value=0.18,
                continuation_likelihood=0.64,
                attackability_after_hit=0.76,
                post_hit_continue_score=0.41,
                post_hit_stop_score=0.24,
                post_hit_continue_margin=0.17,
                post_hit_best_gap=0.30,
                post_hit_top_k_continue_margin=0.03,
                post_hit_top_k_support_ratio=0.33,
            ),
        ],
        expect_continue=False,
        positive_breakdown_key="top_k_rollout_pressure",
        max_continue_margin=0.0,
    ),
    DecisionRegressionCase(
        name="low_attackability_stop",
        my_hidden_count=2,
        moves=[
            build_move(
                expected_value=0.79,
                win_probability=0.55,
                continuation_value=0.18,
                continuation_likelihood=0.51,
                attackability_after_hit=0.05,
            ),
        ],
        expect_continue=False,
        positive_breakdown_key="attackability_pressure",
        max_continue_margin=0.0,
    ),
]


BEHAVIOR_REGRESSION_CASES = [
    BehaviorRegressionCase(
        name="same_player_focus",
        preferred_state=GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[CardSlot(slot_index=0, color="B", value=0, is_revealed=True)],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=1, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=4, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=11, is_revealed=True),
                    ],
                ),
            },
            actions=[
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=2,
                    result=False,
                ),
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=3,
                    result=False,
                ),
            ],
        ),
        alternative_state=GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[CardSlot(slot_index=0, color="B", value=0, is_revealed=True)],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=1, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=4, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=11, is_revealed=True),
                    ],
                ),
            },
            actions=[
                GuessAction(
                    guesser_id="me",
                    target_player_id="side",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=8,
                    result=False,
                ),
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=3,
                    result=False,
                ),
            ],
        ),
        preferred_hypothesis={"opp": {1: ("W", 3)}, "side": {1: ("W", 10)}},
        alternative_hypothesis={"opp": {1: ("W", 3)}, "side": {1: ("W", 10)}},
    ),
    BehaviorRegressionCase(
        name="same_slot_retry",
        preferred_state=GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[CardSlot(slot_index=0, color="B", value=0, is_revealed=True)],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=1, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=3, color="B", value=10, is_revealed=True),
                    ],
                ),
            },
            actions=[
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=2,
                    result=False,
                ),
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=3,
                    result=False,
                ),
            ],
        ),
        alternative_state=GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[CardSlot(slot_index=0, color="B", value=0, is_revealed=True)],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=1, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=3, color="B", value=10, is_revealed=True),
                    ],
                ),
            },
            actions=[
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=2,
                    guessed_color="W",
                    guessed_value=8,
                    result=False,
                ),
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=3,
                    result=False,
                ),
            ],
        ),
        preferred_hypothesis={"opp": {1: ("W", 3), 2: ("W", 9)}},
        alternative_hypothesis={"opp": {1: ("W", 3), 2: ("W", 9)}},
    ),
    BehaviorRegressionCase(
        name="progressive_value_step",
        preferred_state=GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[CardSlot(slot_index=0, color="B", value=0, is_revealed=True)],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=1, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=9, is_revealed=True),
                    ],
                ),
            },
            actions=[
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=5,
                    result=False,
                ),
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=6,
                    result=False,
                ),
            ],
        ),
        alternative_state=GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[CardSlot(slot_index=0, color="B", value=0, is_revealed=True)],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=1, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=9, is_revealed=True),
                    ],
                ),
            },
            actions=[
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=5,
                    result=False,
                ),
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=8,
                    result=False,
                ),
            ],
        ),
        preferred_hypothesis={"opp": {1: ("W", 7)}},
        alternative_hypothesis={"opp": {1: ("W", 7)}},
    ),
    BehaviorRegressionCase(
        name="same_color_anchor_sandwich",
        preferred_state=GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color=None, value=None, is_revealed=False),
                        CardSlot(slot_index=1, color=None, value=None, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=1, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=9, is_revealed=True),
                    ],
                ),
            },
            actions=[
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=5,
                    result=False,
                ),
            ],
        ),
        alternative_state=GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color=None, value=None, is_revealed=False),
                        CardSlot(slot_index=1, color=None, value=None, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=1, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=9, is_revealed=True),
                    ],
                ),
            },
            actions=[
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=5,
                    result=False,
                ),
            ],
        ),
        preferred_hypothesis={
            "me": {0: ("W", 4), 1: ("W", 6)},
            "opp": {1: ("W", 5)},
        },
        alternative_hypothesis={
            "me": {0: ("W", 4), 1: ("W", 9)},
            "opp": {1: ("W", 5)},
        },
    ),
    BehaviorRegressionCase(
        name="wide_gap_center_probe",
        preferred_state=GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[CardSlot(slot_index=0, color="B", value=0, is_revealed=True)],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=1, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=11, is_revealed=True),
                    ],
                ),
            },
            actions=[
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=6,
                    result=False,
                ),
            ],
        ),
        alternative_state=GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[CardSlot(slot_index=0, color="B", value=0, is_revealed=True)],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=1, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=11, is_revealed=True),
                    ],
                ),
            },
            actions=[
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=4,
                    result=False,
                ),
            ],
        ),
        preferred_hypothesis={"opp": {1: ("W", 5)}},
        alternative_hypothesis={"opp": {1: ("W", 5)}},
    ),
]


class DecisionRegressionCaseTests(unittest.TestCase):
    def test_decision_regression_cases(self):
        engine = DaVinciDecisionEngine()

        for case in DECISION_REGRESSION_CASES:
            with self.subTest(case=case.name):
                best_move, summary = engine.choose_best_move(
                    case.moves,
                    risk_factor=engine.calculate_risk_factor(case.my_hidden_count),
                    my_hidden_count=case.my_hidden_count,
                )

                self.assertEqual(best_move is not None, case.expect_continue)
                self.assertEqual(summary["recommend_stop"], not case.expect_continue)

                if case.stop_reason_contains is not None:
                    self.assertIn(case.stop_reason_contains, summary["stop_reason"])
                if case.positive_breakdown_key is not None:
                    self.assertGreater(
                        summary["decision_score_breakdown"][case.positive_breakdown_key],
                        0.0,
                    )
                if case.min_continue_margin is not None:
                    self.assertGreater(summary["continue_margin"], case.min_continue_margin)
                if case.max_continue_margin is not None:
                    self.assertLess(summary["continue_margin"], case.max_continue_margin)

    def test_decision_benchmark_quantifies_accuracy_and_margin(self):
        engine = DaVinciDecisionEngine()

        benchmark = engine.benchmark_decision_cases(DECISION_REGRESSION_CASES)

        self.assertEqual(
            benchmark["base_case_count"],
            float(len(DECISION_REGRESSION_CASES)),
        )
        self.assertGreater(
            benchmark["case_count"],
            benchmark["base_case_count"],
        )
        self.assertEqual(benchmark["accuracy"], 1.0)
        self.assertGreater(benchmark["margin_separation"], 0.15)
        self.assertGreater(benchmark["min_correct_margin"], 0.0)


class BehaviorRegressionCaseTests(unittest.TestCase):
    def test_behavior_regression_cases(self):
        model = BehavioralLikelihoodModel()

        for case in BEHAVIOR_REGRESSION_CASES:
            with self.subTest(case=case.name):
                preferred_score = model.score_hypothesis(
                    case.preferred_hypothesis,
                    model.build_guess_signals(case.preferred_state),
                    case.preferred_state,
                )
                alternative_score = model.score_hypothesis(
                    case.alternative_hypothesis,
                    model.build_guess_signals(case.alternative_state),
                    case.alternative_state,
                )
                self.assertGreater(preferred_score, alternative_score)

    def test_behavior_benchmark_quantifies_preference_margin(self):
        engine = DaVinciDecisionEngine()

        benchmark = engine.benchmark_behavior_cases(BEHAVIOR_REGRESSION_CASES)

        self.assertEqual(
            benchmark["base_case_count"],
            float(len(BEHAVIOR_REGRESSION_CASES)),
        )
        self.assertGreater(
            benchmark["case_count"],
            benchmark["base_case_count"],
        )
        self.assertEqual(benchmark["accuracy"], 1.0)
        self.assertGreater(benchmark["average_log_margin"], 0.0)
        self.assertGreater(benchmark["min_log_margin"], 0.0)
        self.assertGreater(benchmark["average_score_ratio"], 1.0)

    def test_self_play_benchmark_quantifies_guess_draw_and_rollout_behavior(self):
        engine = DaVinciDecisionEngine()

        benchmark = engine.benchmark_self_play_worlds(world_count=4, seed=11)

        self.assertEqual(benchmark["world_count"], 4.0)
        self.assertGreaterEqual(benchmark["guess_rate"], 0.0)
        self.assertLessEqual(benchmark["guess_rate"], 1.0)
        self.assertGreaterEqual(benchmark["top1_guess_accuracy"], 0.0)
        self.assertLessEqual(benchmark["top1_guess_accuracy"], 1.0)
        self.assertGreaterEqual(
            benchmark["top3_guess_accuracy"],
            benchmark["top1_guess_accuracy"],
        )
        self.assertGreaterEqual(benchmark["draw_color_alignment"], 0.0)
        self.assertLessEqual(benchmark["draw_color_alignment"], 1.0)
        self.assertGreaterEqual(benchmark["recommended_guess_rate"], 0.0)
        self.assertLessEqual(benchmark["recommended_guess_rate"], 1.0)
        self.assertGreaterEqual(benchmark["recommended_draw_rate"], 0.0)
        self.assertLessEqual(benchmark["recommended_draw_rate"], 1.0)
        self.assertGreaterEqual(benchmark["recommended_stop_rate"], 0.0)
        self.assertLessEqual(benchmark["recommended_stop_rate"], 1.0)
        self.assertGreaterEqual(benchmark["average_realized_hits_per_turn"], 0.0)
        self.assertGreater(benchmark["average_realized_strategy_objective"], -10.0)
        self.assertGreaterEqual(benchmark["average_executed_steps"], 0.0)
        self.assertGreaterEqual(benchmark["deep_rollout_usage"], 0.0)
        self.assertLessEqual(benchmark["deep_rollout_usage"], 1.0)

    def test_long_horizon_self_play_benchmark_quantifies_win_rate_and_stop_rate(self):
        engine = DaVinciDecisionEngine()

        benchmark = engine.benchmark_long_horizon_self_play(game_count=1, seed=13)

        self.assertEqual(benchmark["game_count"], 1.0)
        self.assertGreaterEqual(benchmark["p0_win_rate"], 0.0)
        self.assertLessEqual(benchmark["p0_win_rate"], 1.0)
        self.assertGreaterEqual(benchmark["p1_win_rate"], 0.0)
        self.assertLessEqual(benchmark["p1_win_rate"], 1.0)
        self.assertGreaterEqual(benchmark["draw_rate"], 0.0)
        self.assertLessEqual(benchmark["draw_rate"], 1.0)
        self.assertAlmostEqual(
            benchmark["p0_win_rate"] + benchmark["p1_win_rate"] + benchmark["draw_rate"],
            1.0,
            places=6,
        )
        self.assertGreaterEqual(benchmark["average_turn_count"], 0.0)
        self.assertGreaterEqual(benchmark["average_guess_count"], 0.0)
        self.assertGreaterEqual(benchmark["average_successful_guesses"], 0.0)
        self.assertGreaterEqual(benchmark["post_draw_stop_rate"], 0.0)
        self.assertLessEqual(benchmark["post_draw_stop_rate"], 1.0)
        self.assertGreaterEqual(benchmark["starting_player_win_rate"], 0.0)
        self.assertLessEqual(benchmark["starting_player_win_rate"], 1.0)
        self.assertGreaterEqual(benchmark["non_starting_player_win_rate"], 0.0)
        self.assertLessEqual(benchmark["non_starting_player_win_rate"], 1.0)
        self.assertAlmostEqual(
            benchmark["starting_player_win_rate"]
            + benchmark["non_starting_player_win_rate"]
            + benchmark["draw_rate"],
            1.0,
            places=6,
        )
        self.assertGreaterEqual(benchmark["seat_bias"], 0.0)
        self.assertLessEqual(benchmark["seat_bias"], 1.0)

    def test_long_horizon_self_play_stays_active_after_draw(self):
        engine = DaVinciDecisionEngine()

        benchmark = engine.benchmark_long_horizon_self_play(game_count=2, seed=19)

        self.assertEqual(benchmark["game_count"], 2.0)
        self.assertGreaterEqual(benchmark["average_guess_count"], 1.0)
        self.assertGreaterEqual(benchmark["average_successful_guesses"], 0.5)
        self.assertLessEqual(benchmark["post_draw_stop_rate"], 0.25)
        self.assertLess(benchmark["draw_rate"], 1.0)

    def test_long_horizon_league_benchmark_aggregates_multiple_matches(self):
        engine = DaVinciDecisionEngine()

        benchmark = engine.benchmark_long_horizon_league(
            match_count=2,
            games_per_match=1,
            seed=31,
        )

        self.assertEqual(benchmark["match_count"], 2.0)
        self.assertEqual(benchmark["games_per_match"], 1.0)
        self.assertEqual(benchmark["total_game_count"], 2.0)
        self.assertGreaterEqual(benchmark["p0_win_rate"], 0.0)
        self.assertLessEqual(benchmark["p0_win_rate"], 1.0)
        self.assertGreaterEqual(benchmark["p1_win_rate"], 0.0)
        self.assertLessEqual(benchmark["p1_win_rate"], 1.0)
        self.assertGreaterEqual(benchmark["draw_rate"], 0.0)
        self.assertLessEqual(benchmark["draw_rate"], 1.0)
        self.assertAlmostEqual(
            benchmark["p0_win_rate"] + benchmark["p1_win_rate"] + benchmark["draw_rate"],
            1.0,
            places=6,
        )
        self.assertGreaterEqual(benchmark["average_guess_count"], 0.0)
        self.assertGreaterEqual(benchmark["average_successful_guesses"], 0.0)
        self.assertGreaterEqual(benchmark["average_post_draw_stop_rate"], 0.0)
        self.assertLessEqual(benchmark["average_post_draw_stop_rate"], 1.0)
        self.assertGreaterEqual(benchmark["starting_player_win_rate"], 0.0)
        self.assertLessEqual(benchmark["starting_player_win_rate"], 1.0)
        self.assertGreaterEqual(benchmark["non_starting_player_win_rate"], 0.0)
        self.assertLessEqual(benchmark["non_starting_player_win_rate"], 1.0)
        self.assertGreaterEqual(benchmark["average_starting_player_advantage"], -1.0)
        self.assertLessEqual(benchmark["average_starting_player_advantage"], 1.0)
        self.assertGreaterEqual(benchmark["seat_bias"], 0.0)
        self.assertGreaterEqual(benchmark["average_match_seat_bias"], 0.0)


if __name__ == "__main__":
    unittest.main()
