import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.fixed_decision_cases import (
    FixedDecisionCase,
    assert_continue_summary,
    assert_negative_breakdown,
    assert_positive_breakdown,
    assert_stop_summary,
    run_decision_case,
)
from app.core.engine import BehavioralLikelihoodModel, DaVinciDecisionEngine
from app.core.state import CardSlot, GameState, GuessAction, PlayerState


class ContinuationLikelihoodTests(unittest.TestCase):
    def test_continue_likelihood_prefers_tighter_success_world(self):
        model = BehavioralLikelihoodModel()
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[CardSlot(slot_index=0, color="B", value=1, is_revealed=True)],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[CardSlot(slot_index=0, color=None, value=None, is_revealed=False)],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[CardSlot(slot_index=0, color=None, value=None, is_revealed=False)],
                ),
            },
            actions=[
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=0,
                    guessed_color="W",
                    guessed_value=7,
                    result=True,
                    continued_turn=True,
                )
            ],
        )
        signals = model.build_guess_signals(game_state)

        tight_matrix = {
            "side": {
                0: {("W", 3): 0.82, ("W", 4): 0.18},
            }
        }
        loose_matrix = {
            "side": {
                0: {("W", 2): 0.26, ("W", 4): 0.24, ("W", 6): 0.25, ("W", 8): 0.25},
            }
        }

        tight = model.estimate_continue_likelihood(tight_matrix, signals, "me")
        loose = model.estimate_continue_likelihood(loose_matrix, signals, "me")

        self.assertGreater(tight["continue_likelihood"], loose["continue_likelihood"])
        self.assertGreater(tight["attackability"], loose["attackability"])

    def test_continue_likelihood_rewards_secondary_attackable_followup(self):
        model = BehavioralLikelihoodModel()
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[CardSlot(slot_index=0, color="B", value=1, is_revealed=True)],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[CardSlot(slot_index=0, color=None, value=None, is_revealed=False)],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color=None, value=None, is_revealed=False),
                        CardSlot(slot_index=1, color=None, value=None, is_revealed=False),
                    ],
                ),
            },
            actions=[
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=0,
                    guessed_color="W",
                    guessed_value=7,
                    result=True,
                    continued_turn=True,
                )
            ],
        )
        signals = model.build_guess_signals(game_state)

        clustered_matrix = {
            "side": {
                0: {("W", 3): 0.74, ("W", 4): 0.26},
                1: {("B", 6): 0.70, ("B", 7): 0.30},
            }
        }
        isolated_matrix = {
            "side": {
                0: {("W", 3): 0.74, ("W", 4): 0.26},
                1: {("B", 5): 0.26, ("B", 7): 0.24, ("B", 9): 0.25, ("B", 11): 0.25},
            }
        }

        clustered = model.estimate_continue_likelihood(clustered_matrix, signals, "me")
        isolated = model.estimate_continue_likelihood(isolated_matrix, signals, "me")

        self.assertGreater(clustered["continue_likelihood"], isolated["continue_likelihood"])
        self.assertGreater(clustered["attackability"], isolated["attackability"])

    def test_continue_likelihood_prefers_same_player_followup_cluster(self):
        model = BehavioralLikelihoodModel()
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[CardSlot(slot_index=0, color="B", value=1, is_revealed=True)],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color=None, value=None, is_revealed=False),
                        CardSlot(slot_index=1, color=None, value=None, is_revealed=False),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[CardSlot(slot_index=0, color=None, value=None, is_revealed=False)],
                ),
            },
            actions=[
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=0,
                    guessed_color="W",
                    guessed_value=7,
                    result=True,
                    continued_turn=True,
                )
            ],
        )
        signals = model.build_guess_signals(game_state)

        same_player_clustered = {
            "opp": {
                0: {("W", 3): 0.74, ("W", 4): 0.26},
                1: {("B", 6): 0.70, ("B", 7): 0.30},
            },
            "side": {
                0: {("W", 9): 0.52, ("W", 10): 0.48},
            },
        }
        split_across_players = {
            "opp": {
                0: {("W", 3): 0.74, ("W", 4): 0.26},
                1: {("B", 5): 0.26, ("B", 7): 0.24, ("B", 9): 0.25, ("B", 11): 0.25},
            },
            "side": {
                0: {("W", 9): 0.70, ("W", 10): 0.30},
            },
        }

        same_target = model.estimate_continue_likelihood(same_player_clustered, signals, "me")
        split_targets = model.estimate_continue_likelihood(split_across_players, signals, "me")

        self.assertGreater(same_target["continue_likelihood"], split_targets["continue_likelihood"])
        self.assertGreater(same_target["attackability"], split_targets["attackability"])

    def test_continuation_profile_prefers_recent_continue_signal(self):
        model = BehavioralLikelihoodModel()
        recent_continue_world = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[CardSlot(slot_index=0, color="B", value=1, is_revealed=True)],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[CardSlot(slot_index=0, color=None, value=None, is_revealed=False)],
                ),
            },
            actions=[
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=0,
                    guessed_color="W",
                    guessed_value=7,
                    result=True,
                    continued_turn=False,
                ),
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=0,
                    guessed_color="W",
                    guessed_value=8,
                    result=True,
                    continued_turn=True,
                ),
            ],
        )
        recent_stop_world = GameState(
            self_player_id="me",
            target_player_id="opp",
            players=recent_continue_world.players,
            actions=[
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=0,
                    guessed_color="W",
                    guessed_value=7,
                    result=True,
                    continued_turn=True,
                ),
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=0,
                    guessed_color="W",
                    guessed_value=8,
                    result=True,
                    continued_turn=False,
                ),
            ],
        )

        recent_continue_profile = model.continuation_profile(
            model.build_guess_signals(recent_continue_world),
            "me",
        )
        recent_stop_profile = model.continuation_profile(
            model.build_guess_signals(recent_stop_world),
            "me",
        )

        self.assertGreater(
            recent_continue_profile["continue_rate"],
            recent_stop_profile["continue_rate"],
        )
        self.assertEqual(recent_continue_profile["observations"], 2.0)
        self.assertEqual(recent_stop_profile["observations"], 2.0)

    def test_continuation_profile_rewards_confident_continue_streak(self):
        model = BehavioralLikelihoodModel()
        streak_world = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[CardSlot(slot_index=0, color="B", value=1, is_revealed=True)],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[CardSlot(slot_index=0, color=None, value=None, is_revealed=False)],
                ),
            },
            actions=[
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=0,
                    guessed_color="W",
                    guessed_value=7,
                    result=True,
                    continued_turn=True,
                ),
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=0,
                    guessed_color="W",
                    guessed_value=8,
                    result=True,
                    continued_turn=True,
                ),
            ],
        )
        mixed_world = GameState(
            self_player_id="me",
            target_player_id="opp",
            players=streak_world.players,
            actions=[
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=0,
                    guessed_color="W",
                    guessed_value=7,
                    result=True,
                    continued_turn=False,
                ),
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=0,
                    guessed_color="W",
                    guessed_value=8,
                    result=True,
                    continued_turn=True,
                ),
            ],
        )

        streak_profile = model.continuation_profile(
            model.build_guess_signals(streak_world),
            "me",
        )
        mixed_profile = model.continuation_profile(
            model.build_guess_signals(mixed_world),
            "me",
        )

        self.assertGreater(streak_profile["continue_rate"], mixed_profile["continue_rate"])
        self.assertEqual(streak_profile["observations"], 2.0)
        self.assertEqual(mixed_profile["observations"], 2.0)


class StopThresholdTests(unittest.TestCase):
    def test_fixed_decision_case_runner_covers_strong_continue_edge(self):
        case = FixedDecisionCase(
            name="strong_continue_edge",
            my_hidden_count=2,
            all_moves=[
                {
                    "expected_value": 1.45,
                    "win_probability": 0.58,
                    "continuation_value": 0.72,
                    "continuation_likelihood": 0.76,
                    "attackability_after_hit": 0.68,
                    "post_hit_continue_score": 0.92,
                    "post_hit_stop_score": 0.54,
                    "post_hit_continue_margin": 0.38,
                    "post_hit_best_gap": 0.31,
                    "post_hit_top_k_continue_margin": 0.29,
                    "post_hit_top_k_support_ratio": 1.0,
                },
                {
                    "expected_value": 0.71,
                    "win_probability": 0.49,
                    "continuation_value": 0.22,
                    "continuation_likelihood": 0.55,
                    "attackability_after_hit": 0.42,
                    "post_hit_continue_score": 0.28,
                    "post_hit_stop_score": 0.24,
                    "post_hit_continue_margin": 0.04,
                    "post_hit_best_gap": 0.09,
                    "post_hit_top_k_continue_margin": 0.02,
                    "post_hit_top_k_support_ratio": 0.67,
                },
            ],
            checks=(assert_continue_summary,),
        )

        best_move, summary = run_decision_case(case)

        self.assertIsNotNone(best_move)
        self.assertGreater(summary["best_continuation_likelihood"], 0.70)

    def test_evaluate_all_moves_applies_behavior_guidance_to_continuation(self):
        engine = DaVinciDecisionEngine()
        model = BehavioralLikelihoodModel()
        full_probability_matrix = {
            "opp": {
                0: {("W", 5): 0.62, ("W", 6): 0.38},
            },
            "side": {
                0: {("B", 3): 0.81, ("B", 4): 0.19},
            },
        }
        hidden_index_by_player = {
            "opp": {0: 0},
            "side": {0: 0},
        }

        boosted_moves, _ = engine.evaluate_all_moves(
            full_probability_matrix=full_probability_matrix,
            my_hidden_count=2,
            hidden_index_by_player=hidden_index_by_player,
            behavior_model=model,
            guess_signals_by_player={},
            acting_player_id="me",
            behavior_guidance_profile={
                "signal_count": 2.0,
                "average_posterior_support": 0.82,
                "average_weighted_strength": 0.16,
                "stable_signal_ratio": 1.0,
                "guidance_multiplier": 1.08,
            },
            blocked_slots=set(),
            rollout_depth=1,
        )
        damped_moves, _ = engine.evaluate_all_moves(
            full_probability_matrix=full_probability_matrix,
            my_hidden_count=2,
            hidden_index_by_player=hidden_index_by_player,
            behavior_model=model,
            guess_signals_by_player={},
            acting_player_id="me",
            behavior_guidance_profile={
                "signal_count": 2.0,
                "average_posterior_support": 0.24,
                "average_weighted_strength": 0.03,
                "stable_signal_ratio": 0.0,
                "guidance_multiplier": 0.95,
            },
            blocked_slots=set(),
            rollout_depth=1,
        )

        self.assertGreater(
            boosted_moves[0]["continuation_likelihood"],
            damped_moves[0]["continuation_likelihood"],
        )
        self.assertGreater(
            boosted_moves[0]["behavior_guidance_multiplier"],
            damped_moves[0]["behavior_guidance_multiplier"],
        )
        self.assertGreater(
            boosted_moves[0]["behavior_guidance_support"],
            damped_moves[0]["behavior_guidance_support"],
        )

    def test_evaluate_all_moves_uses_behavior_match_bonus_in_ranking(self):
        engine = DaVinciDecisionEngine()
        model = BehavioralLikelihoodModel()
        game_state = GameState(
            self_player_id="me",
            target_player_id="anchor_target",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="W", value=4, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=6, is_revealed=True),
                    ],
                ),
                "anchor_target": PlayerState(
                    player_id="anchor_target",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=1, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=9, is_revealed=True),
                    ],
                ),
                "neutral_target": PlayerState(
                    player_id="neutral_target",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=1, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=9, is_revealed=True),
                    ],
                ),
            },
            actions=[
                GuessAction(
                    guesser_id="me",
                    target_player_id="anchor_target",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=5,
                    result=False,
                ),
            ],
        )
        full_probability_matrix = {
            "anchor_target": {
                1: {("W", 5): 0.650, ("W", 7): 0.350},
            },
            "neutral_target": {
                1: {("B", 3): 0.651, ("B", 5): 0.349},
            },
        }
        hidden_index_by_player = {
            "anchor_target": {1: 0},
            "neutral_target": {1: 0},
        }
        behavior_map_hypothesis = {
            "anchor_target": {1: ("W", 5)},
            "neutral_target": {1: ("B", 3)},
        }

        ranked_moves, _ = engine.evaluate_all_moves(
            full_probability_matrix=full_probability_matrix,
            my_hidden_count=10,
            hidden_index_by_player=hidden_index_by_player,
            behavior_model=model,
            guess_signals_by_player={},
            acting_player_id="me",
            behavior_guidance_profile={
                "signal_count": 1.0,
                "average_posterior_support": 0.90,
                "average_weighted_strength": 0.18,
                "stable_signal_ratio": 1.0,
                "guidance_multiplier": 1.0,
                "source_support_progressive": 0.0,
                "source_support_same_color_anchor": 1.0,
                "source_support_local_boundary": 0.0,
            },
            game_state=game_state,
            behavior_map_hypothesis=behavior_map_hypothesis,
            blocked_slots=set(),
            rollout_depth=0,
        )

        self.assertEqual(ranked_moves[0]["target_player_id"], "anchor_target")
        self.assertGreater(ranked_moves[0]["behavior_match_bonus"], 0.0)
        self.assertGreater(
            ranked_moves[0]["ranking_score"],
            ranked_moves[0]["expected_value"],
        )
        self.assertGreater(
            ranked_moves[0]["behavior_match_multiplier"],
            ranked_moves[1]["behavior_match_multiplier"],
        )

    def test_evaluate_all_moves_uses_neighbor_posterior_for_candidate_signal(self):
        engine = DaVinciDecisionEngine()
        model = BehavioralLikelihoodModel()
        game_state = GameState(
            self_player_id="me",
            target_player_id="boundary_target",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[CardSlot(slot_index=0, color="B", value=0, is_revealed=True)],
                ),
                "boundary_target": PlayerState(
                    player_id="boundary_target",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=None, is_revealed=False),
                    ],
                ),
            },
            actions=[],
        )
        full_probability_matrix = {
            "boundary_target": {
                0: {("B", 1): 0.51, ("B", 2): 0.49},
                1: {("W", 3): 0.58, ("W", 4): 0.42},
                2: {("B", 6): 0.51, ("B", 4): 0.49},
            },
        }
        hidden_index_by_player = {
            "boundary_target": {0: 0, 1: 1, 2: 2},
        }
        behavior_map_hypothesis = {
            "boundary_target": {
                0: ("B", 1),
                1: ("W", 3),
                2: ("B", 6),
            },
        }
        map_signal = model.describe_candidate_value_signal(
            game_state=game_state,
            hypothesis_by_player=behavior_map_hypothesis,
            guesser_id="me",
            target_player_id="boundary_target",
            target_slot_index=1,
            guessed_card=("W", 3),
        )

        ranked_moves, _ = engine.evaluate_all_moves(
            full_probability_matrix=full_probability_matrix,
            my_hidden_count=10,
            hidden_index_by_player=hidden_index_by_player,
            behavior_model=model,
            guess_signals_by_player={},
            acting_player_id="me",
            behavior_guidance_profile={
                "signal_count": 1.0,
                "average_posterior_support": 0.80,
                "average_weighted_strength": 0.10,
                "stable_signal_ratio": 1.0,
                "guidance_multiplier": 1.0,
                "source_support_progressive": 0.0,
                "source_support_same_color_anchor": 0.0,
                "source_support_local_boundary": 1.0,
            },
            game_state=game_state,
            behavior_map_hypothesis=behavior_map_hypothesis,
            blocked_slots={("boundary_target", 0), ("boundary_target", 2)},
            rollout_depth=0,
        )

        target_move = ranked_moves[0]
        self.assertEqual(target_move["guess_card"], ["W", 3])
        self.assertEqual(
            target_move["behavior_candidate_signal"]["mode"],
            "neighbor_top_k_posterior",
        )
        self.assertEqual(target_move["behavior_candidate_signal"]["context_candidate_count"], 4)
        self.assertGreater(
            target_move["behavior_candidate_signal"]["context_covered_probability"],
            0.0,
        )
        self.assertGreater(
            target_move["behavior_candidate_signal"]["boundary"]["weight"],
            map_signal["boundary"]["weight"],
        )
        self.assertGreater(target_move["behavior_match_bonus"], 0.0)

    def test_evaluate_all_moves_scales_ranking_bonus_by_candidate_confidence(self):
        engine = DaVinciDecisionEngine()
        model = BehavioralLikelihoodModel()
        game_state = GameState(
            self_player_id="me",
            target_player_id="stable_anchor_target",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="W", value=4, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=6, is_revealed=True),
                    ],
                ),
                "stable_anchor_target": PlayerState(
                    player_id="stable_anchor_target",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=1, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=9, is_revealed=True),
                    ],
                ),
                "diffuse_anchor_target": PlayerState(
                    player_id="diffuse_anchor_target",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=None, is_revealed=False),
                    ],
                ),
            },
            actions=[],
        )
        full_probability_matrix = {
            "stable_anchor_target": {
                1: {("W", 5): 0.650, ("W", 7): 0.350},
            },
            "diffuse_anchor_target": {
                0: {("B", 1): 0.34, ("B", 2): 0.33, ("B", 3): 0.33},
                1: {("W", 5): 0.650, ("W", 7): 0.350},
                2: {("B", 9): 0.34, ("B", 8): 0.33, ("B", 7): 0.33},
            },
        }
        hidden_index_by_player = {
            "stable_anchor_target": {1: 0},
            "diffuse_anchor_target": {0: 0, 1: 1, 2: 2},
        }
        behavior_map_hypothesis = {
            "stable_anchor_target": {1: ("W", 5)},
            "diffuse_anchor_target": {
                0: ("B", 1),
                1: ("W", 5),
                2: ("B", 9),
            },
        }

        ranked_moves, _ = engine.evaluate_all_moves(
            full_probability_matrix=full_probability_matrix,
            my_hidden_count=10,
            hidden_index_by_player=hidden_index_by_player,
            behavior_model=model,
            guess_signals_by_player={},
            acting_player_id="me",
            behavior_guidance_profile={
                "signal_count": 1.0,
                "average_posterior_support": 0.90,
                "average_weighted_strength": 0.18,
                "stable_signal_ratio": 1.0,
                "guidance_multiplier": 1.0,
                "source_support_progressive": 0.0,
                "source_support_same_color_anchor": 1.0,
                "source_support_local_boundary": 0.0,
            },
            game_state=game_state,
            behavior_map_hypothesis=behavior_map_hypothesis,
            blocked_slots={("diffuse_anchor_target", 0), ("diffuse_anchor_target", 2)},
            rollout_depth=0,
        )

        stable_move = next(
            move
            for move in ranked_moves
            if move["target_player_id"] == "stable_anchor_target"
            and move["guess_card"] == ["W", 5]
        )
        diffuse_move = next(
            move
            for move in ranked_moves
            if move["target_player_id"] == "diffuse_anchor_target"
            and move["guess_card"] == ["W", 5]
        )

        self.assertAlmostEqual(
            stable_move["behavior_match_bonus"],
            diffuse_move["behavior_match_bonus"],
            places=6,
        )
        self.assertGreater(
            stable_move["behavior_match_candidate_confidence"],
            diffuse_move["behavior_match_candidate_confidence"],
        )
        self.assertGreater(
            stable_move["behavior_match_ranking_bonus"],
            diffuse_move["behavior_match_ranking_bonus"],
        )
        self.assertAlmostEqual(
            stable_move["ranking_score"],
            stable_move["expected_value"] + stable_move["behavior_match_ranking_bonus"],
            places=6,
        )
        self.assertEqual(ranked_moves[0]["target_player_id"], "stable_anchor_target")

    def test_behavior_match_ranking_breakdown_rewards_positive_net_structure(self):
        engine = DaVinciDecisionEngine()
        positive_breakdown = engine._behavior_match_ranking_breakdown(
            best_move={
                "behavior_match_bonus": 0.10,
                "behavior_match_candidate_confidence": 0.72,
                "behavior_match_component_support": 0.74,
                "behavior_match_component_strength": 0.62,
                "behavior_match_component_penalty": 0.08,
                "behavior_match_context_focus": 0.90,
            },
        )
        neutral_breakdown = engine._behavior_match_ranking_breakdown(
            best_move={
                "behavior_match_bonus": 0.10,
                "behavior_match_candidate_confidence": 0.72,
                "behavior_match_component_support": 0.74,
                "behavior_match_component_strength": 0.18,
                "behavior_match_component_penalty": 0.18,
                "behavior_match_context_focus": 0.90,
            },
        )

        self.assertGreater(positive_breakdown["net_structure"], 0.0)
        self.assertEqual(neutral_breakdown["net_structure"], 0.0)
        self.assertGreater(
            positive_breakdown["structure_adjustment"],
            neutral_breakdown["structure_adjustment"],
        )
        self.assertGreater(
            positive_breakdown["ranking_bonus"],
            neutral_breakdown["ranking_bonus"],
        )

    def test_behavior_match_ranking_breakdown_penalizes_negative_net_structure(self):
        engine = DaVinciDecisionEngine()
        negative_breakdown = engine._behavior_match_ranking_breakdown(
            best_move={
                "behavior_match_bonus": 0.10,
                "behavior_match_candidate_confidence": 0.12,
                "behavior_match_component_support": 0.42,
                "behavior_match_component_strength": 0.05,
                "behavior_match_component_penalty": 0.70,
                "behavior_match_context_focus": 0.80,
            },
        )
        neutral_breakdown = engine._behavior_match_ranking_breakdown(
            best_move={
                "behavior_match_bonus": 0.10,
                "behavior_match_candidate_confidence": 0.12,
                "behavior_match_component_support": 0.42,
                "behavior_match_component_strength": 0.20,
                "behavior_match_component_penalty": 0.20,
                "behavior_match_context_focus": 0.80,
            },
        )

        self.assertLess(negative_breakdown["net_structure"], 0.0)
        self.assertLess(
            negative_breakdown["structure_adjustment"],
            neutral_breakdown["structure_adjustment"],
        )
        self.assertLess(
            negative_breakdown["ranking_bonus"],
            neutral_breakdown["ranking_bonus"],
        )
        self.assertGreaterEqual(negative_breakdown["ranking_bonus"], 0.0)

    def test_choose_best_move_rewards_positive_net_structure_on_narrow_decision_edge(self):
        engine = DaVinciDecisionEngine()
        positive_move = {
            "expected_value": 0.625,
            "win_probability": 0.55,
            "continuation_value": 0.18,
            "continuation_likelihood": 0.58,
            "attackability_after_hit": 0.72,
            "behavior_match_bonus": 0.16,
            "behavior_match_support": 0.16,
            "behavior_guidance_stable_ratio": 1.0,
            "behavior_match_candidate_confidence": 0.76,
            "behavior_match_component_support": 0.74,
            "behavior_match_component_strength": 0.70,
            "behavior_match_component_penalty": 0.10,
            "behavior_match_context_focus": 0.90,
        }
        neutral_move = {
            "expected_value": 0.625,
            "win_probability": 0.55,
            "continuation_value": 0.18,
            "continuation_likelihood": 0.58,
            "attackability_after_hit": 0.72,
            "behavior_match_bonus": 0.16,
            "behavior_match_support": 0.16,
            "behavior_guidance_stable_ratio": 1.0,
            "behavior_match_candidate_confidence": 0.76,
            "behavior_match_component_support": 0.74,
            "behavior_match_component_strength": 0.40,
            "behavior_match_component_penalty": 0.40,
            "behavior_match_context_focus": 0.90,
        }

        positive_best_move, positive_summary = engine.choose_best_move(
            [positive_move],
            risk_factor=engine.calculate_risk_factor(2),
            my_hidden_count=2,
        )
        neutral_best_move, neutral_summary = engine.choose_best_move(
            [neutral_move],
            risk_factor=engine.calculate_risk_factor(2),
            my_hidden_count=2,
        )

        self.assertIsNotNone(positive_best_move)
        self.assertFalse(positive_summary["recommend_stop"])
        self.assertGreater(
            positive_summary["best_behavior_match_decision_structure_adjustment"],
            0.0,
        )
        self.assertEqual(
            neutral_summary["best_behavior_match_decision_structure_adjustment"],
            0.0,
        )
        self.assertGreater(
            positive_summary["continue_score"],
            positive_summary["stop_score"],
        )
        self.assertIsNone(neutral_best_move)
        self.assertTrue(neutral_summary["recommend_stop"])
        self.assertLess(
            neutral_summary["continue_score"],
            neutral_summary["stop_score"],
        )

    def test_choose_best_move_penalizes_negative_net_structure_on_narrow_decision_edge(self):
        engine = DaVinciDecisionEngine()
        neutral_move = {
            "expected_value": 0.635,
            "win_probability": 0.55,
            "continuation_value": 0.18,
            "continuation_likelihood": 0.58,
            "attackability_after_hit": 0.72,
            "behavior_match_bonus": 0.16,
            "behavior_match_support": 0.16,
            "behavior_guidance_stable_ratio": 1.0,
            "behavior_match_candidate_confidence": 0.76,
            "behavior_match_component_support": 0.74,
            "behavior_match_component_strength": 0.40,
            "behavior_match_component_penalty": 0.40,
            "behavior_match_context_focus": 0.90,
        }
        negative_move = {
            "expected_value": 0.635,
            "win_probability": 0.55,
            "continuation_value": 0.18,
            "continuation_likelihood": 0.58,
            "attackability_after_hit": 0.72,
            "behavior_match_bonus": 0.16,
            "behavior_match_support": 0.16,
            "behavior_guidance_stable_ratio": 1.0,
            "behavior_match_candidate_confidence": 0.76,
            "behavior_match_component_support": 0.74,
            "behavior_match_component_strength": 0.10,
            "behavior_match_component_penalty": 0.70,
            "behavior_match_context_focus": 0.90,
        }

        neutral_best_move, neutral_summary = engine.choose_best_move(
            [neutral_move],
            risk_factor=engine.calculate_risk_factor(2),
            my_hidden_count=2,
        )
        negative_best_move, negative_summary = engine.choose_best_move(
            [negative_move],
            risk_factor=engine.calculate_risk_factor(2),
            my_hidden_count=2,
        )

        self.assertIsNotNone(neutral_best_move)
        self.assertFalse(neutral_summary["recommend_stop"])
        self.assertEqual(
            neutral_summary["best_behavior_match_decision_structure_adjustment"],
            0.0,
        )
        self.assertGreater(
            neutral_summary["continue_score"],
            neutral_summary["stop_score"],
        )
        self.assertIsNone(negative_best_move)
        self.assertTrue(negative_summary["recommend_stop"])
        self.assertLess(
            negative_summary["best_behavior_match_decision_structure_adjustment"],
            0.0,
        )
        self.assertLess(
            negative_summary["continue_score"],
            negative_summary["stop_score"],
        )

    def test_choose_best_move_stops_on_weak_endgame_edge(self):
        engine = DaVinciDecisionEngine()
        all_moves = [
            {
                "expected_value": 0.62,
                "win_probability": 0.41,
                "continuation_value": 0.09,
                "continuation_likelihood": 0.44,
                "attackability_after_hit": 0.30,
            },
            {
                "expected_value": 0.55,
                "win_probability": 0.40,
                "continuation_value": 0.06,
                "continuation_likelihood": 0.42,
                "attackability_after_hit": 0.28,
            },
        ]

        best_move, summary = engine.choose_best_move(
            all_moves,
            risk_factor=engine.calculate_risk_factor(1),
            my_hidden_count=1,
        )

        self.assertIsNone(best_move)
        self.assertTrue(summary["recommend_stop"])
        self.assertIn("建议停手", summary["stop_reason"])
        self.assertGreater(summary["stop_threshold"], 0.62)
        self.assertGreater(summary["stop_score"], summary["continue_score"])
        self.assertLess(summary["continue_margin"], 0.0)

    def test_choose_best_move_uses_stable_behavior_match_in_continue_boundary(self):
        engine = DaVinciDecisionEngine()
        stable_move = {
            "expected_value": 0.66,
            "win_probability": 0.55,
            "continuation_value": 0.18,
            "continuation_likelihood": 0.58,
            "attackability_after_hit": 0.72,
            "behavior_match_bonus": 0.10,
            "behavior_match_support": 0.18,
            "behavior_guidance_stable_ratio": 1.0,
        }
        weak_move = {
            "expected_value": 0.66,
            "win_probability": 0.55,
            "continuation_value": 0.18,
            "continuation_likelihood": 0.58,
            "attackability_after_hit": 0.72,
            "behavior_match_bonus": 0.10,
            "behavior_match_support": 0.02,
            "behavior_guidance_stable_ratio": 0.20,
        }

        stable_best_move, stable_summary = engine.choose_best_move(
            [stable_move],
            risk_factor=engine.calculate_risk_factor(2),
            my_hidden_count=2,
        )
        weak_best_move, weak_summary = engine.choose_best_move(
            [weak_move],
            risk_factor=engine.calculate_risk_factor(2),
            my_hidden_count=2,
        )

        self.assertIsNotNone(stable_best_move)
        self.assertFalse(stable_summary["recommend_stop"])
        self.assertGreater(
            stable_summary["decision_score_breakdown"]["behavior_match_decision_bonus"],
            0.0,
        )
        self.assertGreater(
            stable_summary["continue_score"],
            stable_summary["stop_score"],
        )
        self.assertIsNone(weak_best_move)
        self.assertTrue(weak_summary["recommend_stop"])
        self.assertLess(
            weak_summary["decision_score_breakdown"]["behavior_match_decision_bonus"],
            stable_summary["decision_score_breakdown"]["behavior_match_decision_bonus"],
        )
        self.assertLess(
            weak_summary["continue_score"],
            weak_summary["stop_score"],
        )

    def test_choose_best_move_scales_behavior_match_decision_bonus_by_candidate_confidence(self):
        engine = DaVinciDecisionEngine()
        strong_candidate_move = {
            "expected_value": 0.66,
            "win_probability": 0.55,
            "continuation_value": 0.18,
            "continuation_likelihood": 0.58,
            "attackability_after_hit": 0.72,
            "behavior_match_bonus": 0.10,
            "behavior_match_support": 0.18,
            "behavior_guidance_stable_ratio": 1.0,
            "behavior_candidate_signal": {
                "mode": "neighbor_top_k_posterior",
                "context_covered_probability": 0.92,
                "dominant_signal": {
                    "source": "local_boundary",
                    "reason": "narrow_boundary_probe",
                    "weight": 1.15,
                    "posterior_support": 0.88,
                },
            },
        }
        weak_candidate_move = {
            "expected_value": 0.66,
            "win_probability": 0.55,
            "continuation_value": 0.18,
            "continuation_likelihood": 0.58,
            "attackability_after_hit": 0.72,
            "behavior_match_bonus": 0.10,
            "behavior_match_support": 0.18,
            "behavior_guidance_stable_ratio": 1.0,
            "behavior_candidate_signal": {
                "mode": "neighbor_top_k_posterior",
                "context_covered_probability": 0.18,
                "dominant_signal": {
                    "source": "local_boundary",
                    "reason": "narrow_boundary_probe",
                    "weight": 1.15,
                    "posterior_support": 0.12,
                },
            },
        }

        strong_best_move, strong_summary = engine.choose_best_move(
            [strong_candidate_move],
            risk_factor=engine.calculate_risk_factor(2),
            my_hidden_count=2,
        )
        weak_best_move, weak_summary = engine.choose_best_move(
            [weak_candidate_move],
            risk_factor=engine.calculate_risk_factor(2),
            my_hidden_count=2,
        )

        self.assertIsNotNone(strong_best_move)
        self.assertFalse(strong_summary["recommend_stop"])
        self.assertGreater(
            strong_summary["best_behavior_match_candidate_confidence"],
            weak_summary["best_behavior_match_candidate_confidence"],
        )
        self.assertGreater(
            strong_summary["decision_score_breakdown"]["behavior_match_decision_bonus"],
            weak_summary["decision_score_breakdown"]["behavior_match_decision_bonus"],
        )
        self.assertGreater(
            strong_summary["continue_score"],
            strong_summary["stop_score"],
        )
        self.assertIsNone(weak_best_move)
        self.assertTrue(weak_summary["recommend_stop"])
        self.assertLess(
            weak_summary["continue_score"],
            weak_summary["stop_score"],
        )

    def test_choose_best_move_scales_candidate_confidence_by_component_support(self):
        engine = DaVinciDecisionEngine()
        strong_component_move = {
            "expected_value": 0.624,
            "win_probability": 0.55,
            "continuation_value": 0.18,
            "continuation_likelihood": 0.58,
            "attackability_after_hit": 0.72,
            "behavior_match_bonus": 0.20,
            "behavior_match_support": 0.18,
            "behavior_guidance_stable_ratio": 1.0,
            "behavior_candidate_signal": {
                "mode": "neighbor_top_k_posterior",
                "context_covered_probability": 0.64,
                "context_candidate_count": 4,
                "dominant_signal": {
                    "source": "local_boundary",
                    "reason": "narrow_boundary_probe",
                    "weight": 1.15,
                    "posterior_support": 0.62,
                },
                "progressive": {"weight": 1.04, "reason": "retry_step_probe", "posterior_support": 0.70},
                "anchor": {"weight": 1.06, "reason": "same_color_sandwich_exact", "posterior_support": 0.68},
                "boundary": {"weight": 1.15, "reason": "narrow_boundary_probe", "posterior_support": 0.62},
            },
        }
        sparse_component_move = {
            "expected_value": 0.624,
            "win_probability": 0.55,
            "continuation_value": 0.18,
            "continuation_likelihood": 0.58,
            "attackability_after_hit": 0.72,
            "behavior_match_bonus": 0.20,
            "behavior_match_support": 0.18,
            "behavior_guidance_stable_ratio": 1.0,
            "behavior_candidate_signal": {
                "mode": "neighbor_top_k_posterior",
                "context_covered_probability": 0.64,
                "context_candidate_count": 4,
                "dominant_signal": {
                    "source": "local_boundary",
                    "reason": "narrow_boundary_probe",
                    "weight": 1.15,
                    "posterior_support": 0.62,
                },
                "progressive": {"weight": 1.0, "reason": "neutral", "posterior_support": 0.0},
                "anchor": {"weight": 1.0, "reason": "neutral", "posterior_support": 0.0},
                "boundary": {"weight": 1.15, "reason": "narrow_boundary_probe", "posterior_support": 0.62},
            },
        }

        strong_best_move, strong_summary = engine.choose_best_move(
            [strong_component_move],
            risk_factor=engine.calculate_risk_factor(2),
            my_hidden_count=2,
        )
        sparse_best_move, sparse_summary = engine.choose_best_move(
            [sparse_component_move],
            risk_factor=engine.calculate_risk_factor(2),
            my_hidden_count=2,
        )

        self.assertIsNotNone(strong_best_move)
        self.assertFalse(strong_summary["recommend_stop"])
        self.assertGreater(
            strong_summary["best_behavior_match_component_support"],
            sparse_summary["best_behavior_match_component_support"],
        )
        self.assertGreater(
            strong_summary["best_behavior_match_candidate_confidence"],
            sparse_summary["best_behavior_match_candidate_confidence"],
        )
        self.assertGreater(
            strong_summary["continue_score"],
            strong_summary["stop_score"],
        )
        self.assertIsNone(sparse_best_move)
        self.assertTrue(sparse_summary["recommend_stop"])
        self.assertLess(
            sparse_summary["continue_score"],
            sparse_summary["stop_score"],
        )

    def test_choose_best_move_scales_candidate_confidence_by_component_weight_strength(self):
        engine = DaVinciDecisionEngine()
        strong_weight_move = {
            "expected_value": 0.58,
            "win_probability": 0.55,
            "continuation_value": 0.18,
            "continuation_likelihood": 0.58,
            "attackability_after_hit": 0.72,
            "behavior_match_bonus": 0.30,
            "behavior_match_support": 0.18,
            "behavior_guidance_stable_ratio": 1.0,
            "behavior_candidate_signal": {
                "mode": "neighbor_top_k_posterior",
                "context_covered_probability": 0.68,
                "context_candidate_count": 4,
                "dominant_signal": {
                    "source": "local_boundary",
                    "reason": "narrow_boundary_probe",
                    "weight": 1.15,
                    "posterior_support": 0.70,
                },
                "progressive": {"weight": 1.15, "reason": "progressive_step", "posterior_support": 0.70},
                "anchor": {"weight": 1.15, "reason": "same_color_sandwich_exact", "posterior_support": 0.70},
                "boundary": {"weight": 1.15, "reason": "narrow_boundary_probe", "posterior_support": 0.70},
            },
        }
        weak_weight_move = {
            "expected_value": 0.58,
            "win_probability": 0.55,
            "continuation_value": 0.18,
            "continuation_likelihood": 0.58,
            "attackability_after_hit": 0.72,
            "behavior_match_bonus": 0.30,
            "behavior_match_support": 0.18,
            "behavior_guidance_stable_ratio": 1.0,
            "behavior_candidate_signal": {
                "mode": "neighbor_top_k_posterior",
                "context_covered_probability": 0.68,
                "context_candidate_count": 4,
                "dominant_signal": {
                    "source": "local_boundary",
                    "reason": "narrow_boundary_probe",
                    "weight": 1.03,
                    "posterior_support": 0.70,
                },
                "progressive": {"weight": 1.01, "reason": "progressive_step", "posterior_support": 0.70},
                "anchor": {"weight": 1.01, "reason": "same_color_sandwich_exact", "posterior_support": 0.70},
                "boundary": {"weight": 1.03, "reason": "narrow_boundary_probe", "posterior_support": 0.70},
            },
        }

        strong_best_move, strong_summary = engine.choose_best_move(
            [strong_weight_move],
            risk_factor=engine.calculate_risk_factor(2),
            my_hidden_count=2,
        )
        weak_best_move, weak_summary = engine.choose_best_move(
            [weak_weight_move],
            risk_factor=engine.calculate_risk_factor(2),
            my_hidden_count=2,
        )

        self.assertIsNotNone(strong_best_move)
        self.assertFalse(strong_summary["recommend_stop"])
        self.assertGreater(
            strong_summary["best_behavior_match_component_strength"],
            weak_summary["best_behavior_match_component_strength"],
        )
        self.assertGreater(
            strong_summary["best_behavior_match_component_support"],
            weak_summary["best_behavior_match_component_support"],
        )
        self.assertGreater(
            strong_summary["best_behavior_match_candidate_confidence"],
            weak_summary["best_behavior_match_candidate_confidence"],
        )
        self.assertGreater(
            strong_summary["continue_score"],
            strong_summary["stop_score"],
        )
        self.assertIsNone(weak_best_move)
        self.assertTrue(weak_summary["recommend_stop"])
        self.assertLess(
            weak_summary["continue_score"],
            weak_summary["stop_score"],
        )

    def test_choose_best_move_penalizes_negative_component_weight_strength(self):
        engine = DaVinciDecisionEngine()
        neutral_weight_move = {
            "expected_value": 0.57,
            "win_probability": 0.55,
            "continuation_value": 0.18,
            "continuation_likelihood": 0.58,
            "attackability_after_hit": 0.72,
            "behavior_match_bonus": 0.36,
            "behavior_match_support": 0.18,
            "behavior_guidance_stable_ratio": 1.0,
            "behavior_candidate_signal": {
                "mode": "neighbor_top_k_posterior",
                "context_covered_probability": 0.68,
                "context_candidate_count": 4,
                "dominant_signal": {
                    "source": "local_boundary",
                    "reason": "narrow_boundary_probe",
                    "weight": 1.00,
                    "posterior_support": 0.70,
                },
                "progressive": {"weight": 1.00, "reason": "neutral", "posterior_support": 0.70},
                "anchor": {"weight": 1.00, "reason": "neutral", "posterior_support": 0.70},
                "boundary": {"weight": 1.00, "reason": "neutral", "posterior_support": 0.70},
            },
        }
        negative_weight_move = {
            "expected_value": 0.57,
            "win_probability": 0.55,
            "continuation_value": 0.18,
            "continuation_likelihood": 0.58,
            "attackability_after_hit": 0.72,
            "behavior_match_bonus": 0.36,
            "behavior_match_support": 0.18,
            "behavior_guidance_stable_ratio": 1.0,
            "behavior_candidate_signal": {
                "mode": "neighbor_top_k_posterior",
                "context_covered_probability": 0.68,
                "context_candidate_count": 4,
                "dominant_signal": {
                    "source": "local_boundary",
                    "reason": "wide_gap_edge_hug",
                    "weight": 0.96,
                    "posterior_support": 0.70,
                },
                "progressive": {"weight": 0.90, "reason": "stalled_after_failure", "posterior_support": 0.70},
                "anchor": {"weight": 0.92, "reason": "wrong_direction", "posterior_support": 0.70},
                "boundary": {"weight": 0.96, "reason": "wide_gap_edge_hug", "posterior_support": 0.70},
            },
        }

        neutral_best_move, neutral_summary = engine.choose_best_move(
            [neutral_weight_move],
            risk_factor=engine.calculate_risk_factor(2),
            my_hidden_count=2,
        )
        negative_best_move, negative_summary = engine.choose_best_move(
            [negative_weight_move],
            risk_factor=engine.calculate_risk_factor(2),
            my_hidden_count=2,
        )

        self.assertIsNotNone(neutral_best_move)
        self.assertFalse(neutral_summary["recommend_stop"])
        self.assertGreater(
            negative_summary["best_behavior_match_component_penalty"],
            neutral_summary["best_behavior_match_component_penalty"],
        )
        self.assertLess(
            negative_summary["best_behavior_match_component_support"],
            neutral_summary["best_behavior_match_component_support"],
        )
        self.assertLess(
            negative_summary["best_behavior_match_candidate_confidence"],
            neutral_summary["best_behavior_match_candidate_confidence"],
        )
        self.assertGreater(
            neutral_summary["continue_score"],
            neutral_summary["stop_score"],
        )
        self.assertIsNone(negative_best_move)
        self.assertTrue(negative_summary["recommend_stop"])
        self.assertLess(
            negative_summary["continue_score"],
            negative_summary["stop_score"],
        )

    def test_choose_best_move_scales_candidate_confidence_by_context_focus(self):
        engine = DaVinciDecisionEngine()
        focused_context_move = {
            "expected_value": 0.648,
            "win_probability": 0.55,
            "continuation_value": 0.18,
            "continuation_likelihood": 0.58,
            "attackability_after_hit": 0.72,
            "behavior_match_bonus": 0.10,
            "behavior_match_support": 0.18,
            "behavior_guidance_stable_ratio": 1.0,
            "behavior_candidate_signal": {
                "mode": "neighbor_top_k_posterior",
                "context_covered_probability": 0.72,
                "context_candidate_count": 2,
                "dominant_signal": {
                    "source": "local_boundary",
                    "reason": "narrow_boundary_probe",
                    "weight": 1.15,
                    "posterior_support": 0.68,
                },
                "progressive": {"weight": 1.02, "reason": "retry_directional_probe", "posterior_support": 0.68},
                "anchor": {"weight": 1.03, "reason": "same_color_sandwich_exact", "posterior_support": 0.68},
                "boundary": {"weight": 1.15, "reason": "narrow_boundary_probe", "posterior_support": 0.68},
            },
        }
        diffuse_context_move = {
            "expected_value": 0.648,
            "win_probability": 0.55,
            "continuation_value": 0.18,
            "continuation_likelihood": 0.58,
            "attackability_after_hit": 0.72,
            "behavior_match_bonus": 0.10,
            "behavior_match_support": 0.18,
            "behavior_guidance_stable_ratio": 1.0,
            "behavior_candidate_signal": {
                "mode": "neighbor_top_k_posterior",
                "context_covered_probability": 0.72,
                "context_candidate_count": 16,
                "dominant_signal": {
                    "source": "local_boundary",
                    "reason": "narrow_boundary_probe",
                    "weight": 1.15,
                    "posterior_support": 0.68,
                },
                "progressive": {"weight": 1.02, "reason": "retry_directional_probe", "posterior_support": 0.68},
                "anchor": {"weight": 1.03, "reason": "same_color_sandwich_exact", "posterior_support": 0.68},
                "boundary": {"weight": 1.15, "reason": "narrow_boundary_probe", "posterior_support": 0.68},
            },
        }

        focused_best_move, focused_summary = engine.choose_best_move(
            [focused_context_move],
            risk_factor=engine.calculate_risk_factor(2),
            my_hidden_count=2,
        )
        diffuse_best_move, diffuse_summary = engine.choose_best_move(
            [diffuse_context_move],
            risk_factor=engine.calculate_risk_factor(2),
            my_hidden_count=2,
        )

        self.assertIsNotNone(focused_best_move)
        self.assertFalse(focused_summary["recommend_stop"])
        self.assertGreater(
            focused_summary["best_behavior_match_context_focus"],
            diffuse_summary["best_behavior_match_context_focus"],
        )
        self.assertGreater(
            focused_summary["best_behavior_match_candidate_confidence"],
            diffuse_summary["best_behavior_match_candidate_confidence"],
        )
        self.assertGreater(
            focused_summary["continue_score"],
            focused_summary["stop_score"],
        )
        self.assertIsNone(diffuse_best_move)
        self.assertTrue(diffuse_summary["recommend_stop"])
        self.assertLess(
            diffuse_summary["continue_score"],
            diffuse_summary["stop_score"],
        )

    def test_choose_best_move_penalizes_fragile_candidate_confidence_in_post_hit_rollout(self):
        engine = DaVinciDecisionEngine()
        stable_rollout_move = {
            "expected_value": 0.74,
            "win_probability": 0.57,
            "continuation_value": 0.23,
            "continuation_likelihood": 0.63,
            "attackability_after_hit": 0.76,
            "behavior_match_bonus": 0.06,
            "behavior_match_support": 0.18,
            "behavior_guidance_stable_ratio": 1.0,
            "behavior_candidate_signal": {
                "mode": "neighbor_top_k_posterior",
                "context_covered_probability": 0.92,
                "dominant_signal": {
                    "source": "local_boundary",
                    "reason": "narrow_boundary_probe",
                    "weight": 1.15,
                    "posterior_support": 0.88,
                },
            },
            "post_hit_continue_score": 0.38,
            "post_hit_stop_score": 0.24,
            "post_hit_continue_margin": 0.14,
            "post_hit_best_gap": 0.28,
            "post_hit_top_k_continue_margin": 0.14,
            "post_hit_top_k_support_ratio": 1.0,
        }
        fragile_rollout_move = {
            "expected_value": 0.74,
            "win_probability": 0.57,
            "continuation_value": 0.23,
            "continuation_likelihood": 0.63,
            "attackability_after_hit": 0.76,
            "behavior_match_bonus": 0.06,
            "behavior_match_support": 0.18,
            "behavior_guidance_stable_ratio": 1.0,
            "behavior_candidate_signal": {
                "mode": "neighbor_top_k_posterior",
                "context_covered_probability": 0.18,
                "dominant_signal": {
                    "source": "local_boundary",
                    "reason": "narrow_boundary_probe",
                    "weight": 1.15,
                    "posterior_support": 0.12,
                },
            },
            "post_hit_continue_score": 0.38,
            "post_hit_stop_score": 0.24,
            "post_hit_continue_margin": 0.14,
            "post_hit_best_gap": 0.28,
            "post_hit_top_k_continue_margin": 0.14,
            "post_hit_top_k_support_ratio": 1.0,
        }

        stable_best_move, stable_summary = engine.choose_best_move(
            [stable_rollout_move],
            risk_factor=engine.calculate_risk_factor(2),
            my_hidden_count=2,
        )
        fragile_best_move, fragile_summary = engine.choose_best_move(
            [fragile_rollout_move],
            risk_factor=engine.calculate_risk_factor(2),
            my_hidden_count=2,
        )

        self.assertIsNotNone(stable_best_move)
        self.assertFalse(stable_summary["recommend_stop"])
        self.assertGreater(
            fragile_summary["decision_score_breakdown"]["behavior_rollout_pressure"],
            stable_summary["decision_score_breakdown"]["behavior_rollout_pressure"],
        )
        self.assertGreater(
            stable_summary["continue_score"],
            stable_summary["stop_score"],
        )
        self.assertIsNone(fragile_best_move)
        self.assertTrue(fragile_summary["recommend_stop"])
        self.assertGreater(
            fragile_summary["decision_score_breakdown"]["behavior_rollout_pressure"],
            0.0,
        )
        self.assertLess(
            fragile_summary["continue_score"],
            fragile_summary["stop_score"],
        )

    def test_post_hit_rollout_uses_behavior_ranked_next_turn_edges_when_context_available(self):
        engine = DaVinciDecisionEngine()
        model = BehavioralLikelihoodModel()
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="W", value=4, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=6, is_revealed=True),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[CardSlot(slot_index=0, color=None, value=None, is_revealed=False)],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[CardSlot(slot_index=0, color=None, value=None, is_revealed=False)],
                ),
            },
            actions=[],
        )
        success_matrix = {
            "side": {
                0: {("W", 5): 0.85, ("W", 7): 0.15},
            },
        }
        behavior_guidance_profile = {
            "signal_count": 1.0,
            "average_posterior_support": 0.90,
            "average_weighted_strength": 0.20,
            "stable_signal_ratio": 1.0,
            "guidance_multiplier": 1.0,
            "source_support_progressive": 0.0,
            "source_support_same_color_anchor": 1.0,
            "source_support_local_boundary": 0.0,
        }
        guess_signals_by_player = model.build_guess_signals(game_state)

        fallback_rollout = engine._evaluate_post_hit_rollout(
            success_matrix=success_matrix,
            my_hidden_count=2,
            behavior_model=model,
            guess_signals_by_player=guess_signals_by_player,
            acting_player_id="me",
            behavior_guidance_profile=behavior_guidance_profile,
            game_state=None,
            target_player_id="opp",
            target_slot_index=0,
            guessed_card=("B", 2),
            rollout_depth=0,
        )
        contextual_rollout = engine._evaluate_post_hit_rollout(
            success_matrix=success_matrix,
            my_hidden_count=2,
            behavior_model=model,
            guess_signals_by_player=guess_signals_by_player,
            acting_player_id="me",
            behavior_guidance_profile=behavior_guidance_profile,
            game_state=game_state,
            target_player_id="opp",
            target_slot_index=0,
            guessed_card=("B", 2),
            rollout_depth=0,
        )

        self.assertAlmostEqual(
            fallback_rollout["top_k_continue_margin"],
            fallback_rollout["top_k_expected_continue_margin"],
            places=6,
        )
        self.assertGreater(
            contextual_rollout["top_k_continue_margin"],
            contextual_rollout["top_k_expected_continue_margin"],
        )
        self.assertGreater(
            contextual_rollout["top_k_continue_margin"],
            fallback_rollout["top_k_continue_margin"],
        )

    def test_post_hit_rollout_rebuilds_guidance_from_confirmed_signal(self):
        engine = DaVinciDecisionEngine()
        model = BehavioralLikelihoodModel()
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="W", value=4, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=6, is_revealed=True),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[CardSlot(slot_index=0, color=None, value=None, is_revealed=False)],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[CardSlot(slot_index=0, color=None, value=None, is_revealed=False)],
                ),
            },
            actions=[],
        )
        success_matrix = {
            "side": {
                0: {("W", 5): 0.85, ("W", 7): 0.15},
            },
        }
        neutral_guidance_profile = {
            "signal_count": 0.0,
            "average_posterior_support": 0.0,
            "average_weighted_strength": 0.0,
            "stable_signal_ratio": 0.0,
            "guidance_multiplier": 1.0,
            "source_support_progressive": 0.0,
            "source_support_same_color_anchor": 0.0,
            "source_support_local_boundary": 0.0,
        }
        guess_signals_by_player = model.build_guess_signals(game_state)

        rollout = engine._evaluate_post_hit_rollout(
            success_matrix=success_matrix,
            my_hidden_count=2,
            behavior_model=model,
            guess_signals_by_player=guess_signals_by_player,
            acting_player_id="me",
            behavior_guidance_profile=neutral_guidance_profile,
            game_state=game_state,
            target_player_id="opp",
            target_slot_index=0,
            guessed_card=("W", 5),
            rollout_depth=0,
        )

        self.assertGreater(rollout["behavior_guidance_multiplier"], 1.0)
        self.assertGreater(rollout["behavior_guidance_signal_count"], 0.0)
        self.assertGreater(rollout["behavior_guidance_support"], 0.0)
        self.assertGreater(
            rollout["top_k_continue_margin"],
            rollout["top_k_expected_continue_margin"],
        )

    def test_post_hit_rollout_rebuilds_guidance_from_full_post_hit_signal_history(self):
        engine = DaVinciDecisionEngine()
        model = BehavioralLikelihoodModel()
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="W", value=4, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=6, is_revealed=True),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color=None, value=None, is_revealed=False),
                        CardSlot(slot_index=1, color=None, value=None, is_revealed=False),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[CardSlot(slot_index=0, color=None, value=None, is_revealed=False)],
                ),
            },
            actions=[
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=7,
                    result=False,
                ),
            ],
        )
        success_matrix = {
            "opp": {
                1: {("W", 8): 0.60, ("W", 9): 0.40},
            },
            "side": {
                0: {("W", 5): 0.85, ("W", 7): 0.15},
            },
        }
        neutral_guidance_profile = {
            "signal_count": 0.0,
            "average_posterior_support": 0.0,
            "average_weighted_strength": 0.0,
            "stable_signal_ratio": 0.0,
            "guidance_multiplier": 1.0,
            "source_support_progressive": 0.0,
            "source_support_same_color_anchor": 0.0,
            "source_support_local_boundary": 0.0,
        }

        rollout = engine._evaluate_post_hit_rollout(
            success_matrix=success_matrix,
            my_hidden_count=2,
            behavior_model=model,
            guess_signals_by_player=model.build_guess_signals(game_state),
            acting_player_id="me",
            behavior_guidance_profile=neutral_guidance_profile,
            game_state=game_state,
            target_player_id="opp",
            target_slot_index=0,
            guessed_card=("W", 5),
            rollout_depth=0,
        )

        self.assertGreaterEqual(rollout["behavior_guidance_signal_count"], 2.0)
        self.assertGreater(rollout["behavior_guidance_multiplier"], 1.0)
        self.assertGreater(rollout["behavior_guidance_support"], 0.0)
        self.assertTrue(rollout["guidance_debug"]["rebuild_applied"])
        self.assertGreaterEqual(rollout["guidance_debug"]["acting_signal_count"], 2.0)
        self.assertGreaterEqual(rollout["guidance_debug"]["augmented_known_slot_count"], 1.0)
        self.assertGreater(
            rollout["guidance_debug"]["rebuilt_profile"]["signal_count"],
            0.0,
        )
        self.assertEqual(
            len(rollout["guidance_debug"]["signal_summaries"]),
            int(rollout["guidance_debug"]["acting_signal_count"]),
        )
        self.assertGreater(
            rollout["guidance_debug"]["rebuilt_delta_from_base"]["signal_count"],
            0.0,
        )
        self.assertGreater(
            rollout["guidance_debug"]["blended_delta_from_base"]["guidance_multiplier"],
            0.0,
        )
        self.assertIn(
            rollout["guidance_debug"]["blended_dominant_source_shift"],
            {"progressive", "same_color_anchor", "local_boundary", "neutral"},
        )
        self.assertGreaterEqual(
            rollout["guidance_debug"]["blended_dominant_source_shift_strength"],
            0.0,
        )

    def test_choose_best_move_uses_post_hit_behavior_support_adjustment_on_narrow_edge(self):
        engine = DaVinciDecisionEngine()
        strong_support_move = {
            "expected_value": 0.676,
            "win_probability": 0.56,
            "continuation_value": 0.18,
            "continuation_likelihood": 0.60,
            "attackability_after_hit": 0.74,
            "post_hit_continue_score": 0.36,
            "post_hit_stop_score": 0.24,
            "post_hit_continue_margin": 0.10,
            "post_hit_best_gap": 0.24,
            "post_hit_guidance_multiplier": 1.08,
            "post_hit_guidance_support": 0.82,
            "post_hit_guidance_stable_ratio": 1.0,
            "post_hit_guidance_signal_count": 2.0,
            "post_hit_top_k_expected_continue_margin": 0.03,
            "post_hit_top_k_continue_margin": 0.10,
            "post_hit_top_k_expected_support_ratio": 0.33,
            "post_hit_top_k_support_ratio": 1.0,
        }
        weak_support_move = {
            "expected_value": 0.676,
            "win_probability": 0.56,
            "continuation_value": 0.18,
            "continuation_likelihood": 0.60,
            "attackability_after_hit": 0.74,
            "post_hit_continue_score": 0.36,
            "post_hit_stop_score": 0.24,
            "post_hit_continue_margin": 0.10,
            "post_hit_best_gap": 0.24,
            "post_hit_guidance_multiplier": 0.96,
            "post_hit_guidance_support": 0.18,
            "post_hit_guidance_stable_ratio": 0.0,
            "post_hit_guidance_signal_count": 1.0,
            "post_hit_top_k_expected_continue_margin": 0.06,
            "post_hit_top_k_continue_margin": 0.02,
            "post_hit_top_k_expected_support_ratio": 0.67,
            "post_hit_top_k_support_ratio": 0.33,
        }

        strong_best_move, strong_summary = engine.choose_best_move(
            [strong_support_move],
            risk_factor=engine.calculate_risk_factor(2),
            my_hidden_count=2,
        )
        weak_best_move, weak_summary = engine.choose_best_move(
            [weak_support_move],
            risk_factor=engine.calculate_risk_factor(2),
            my_hidden_count=2,
        )

        self.assertIsNotNone(strong_best_move)
        self.assertFalse(strong_summary["recommend_stop"])
        self.assertGreater(
            strong_summary["decision_score_breakdown"]["post_hit_behavior_support_adjustment"],
            0.0,
        )
        self.assertGreater(
            strong_summary["decision_score_breakdown"]["post_hit_behavior_support_gain"],
            0.0,
        )
        self.assertLess(
            strong_summary["decision_score_breakdown"]["post_hit_behavior_fragility_drag"],
            strong_summary["decision_score_breakdown"]["post_hit_behavior_support_gain"],
        )
        self.assertGreater(
            strong_summary["continue_score"],
            strong_summary["stop_score"],
        )
        self.assertIsNone(weak_best_move)
        self.assertTrue(weak_summary["recommend_stop"])
        self.assertLess(
            weak_summary["decision_score_breakdown"]["post_hit_behavior_support_adjustment"],
            0.0,
        )
        self.assertGreater(
            weak_summary["decision_score_breakdown"]["post_hit_behavior_fragility_drag"],
            0.0,
        )
        self.assertLess(
            weak_summary["decision_score_breakdown"]["post_hit_behavior_support_gain"],
            weak_summary["decision_score_breakdown"]["post_hit_behavior_fragility_drag"],
        )
        self.assertLess(
            weak_summary["continue_score"],
            weak_summary["stop_score"],
        )

    def test_fixed_decision_case_runner_covers_post_hit_behavior_support_continue_edge(self):
        case = FixedDecisionCase(
            name="post_hit_behavior_support_continue_edge",
            my_hidden_count=2,
            all_moves=[
                {
                    "expected_value": 0.676,
                    "win_probability": 0.56,
                    "continuation_value": 0.18,
                    "continuation_likelihood": 0.60,
                    "attackability_after_hit": 0.74,
                    "post_hit_continue_score": 0.36,
                    "post_hit_stop_score": 0.24,
                    "post_hit_continue_margin": 0.10,
                    "post_hit_best_gap": 0.24,
                    "post_hit_guidance_multiplier": 1.08,
                    "post_hit_guidance_support": 0.82,
                    "post_hit_guidance_stable_ratio": 1.0,
                    "post_hit_guidance_signal_count": 2.0,
                    "post_hit_top_k_expected_continue_margin": 0.03,
                    "post_hit_top_k_continue_margin": 0.10,
                    "post_hit_top_k_expected_support_ratio": 0.33,
                    "post_hit_top_k_support_ratio": 1.0,
                },
            ],
            checks=(
                assert_continue_summary,
                assert_positive_breakdown("post_hit_behavior_support_adjustment"),
                assert_positive_breakdown("post_hit_behavior_support_gain"),
            ),
        )

        best_move, summary = run_decision_case(case)

        self.assertIsNotNone(best_move)
        self.assertEqual(summary["decision_score_breakdown"]["post_hit_behavior_fragility_drag"], 0.0)

    def test_fixed_decision_case_runner_covers_post_hit_behavior_fragility_stop_edge(self):
        case = FixedDecisionCase(
            name="post_hit_behavior_fragility_stop_edge",
            my_hidden_count=2,
            all_moves=[
                {
                    "expected_value": 0.676,
                    "win_probability": 0.56,
                    "continuation_value": 0.18,
                    "continuation_likelihood": 0.60,
                    "attackability_after_hit": 0.74,
                    "post_hit_continue_score": 0.36,
                    "post_hit_stop_score": 0.24,
                    "post_hit_continue_margin": 0.10,
                    "post_hit_best_gap": 0.24,
                    "post_hit_guidance_multiplier": 0.96,
                    "post_hit_guidance_support": 0.18,
                    "post_hit_guidance_stable_ratio": 0.0,
                    "post_hit_guidance_signal_count": 1.0,
                    "post_hit_top_k_expected_continue_margin": 0.06,
                    "post_hit_top_k_continue_margin": 0.02,
                    "post_hit_top_k_expected_support_ratio": 0.67,
                    "post_hit_top_k_support_ratio": 0.33,
                },
            ],
            checks=(
                assert_stop_summary,
                assert_negative_breakdown("post_hit_behavior_support_adjustment"),
                assert_positive_breakdown("post_hit_behavior_fragility_drag"),
            ),
        )

        best_move, summary = run_decision_case(case)

        self.assertIsNone(best_move)
        self.assertEqual(summary["decision_score_breakdown"]["post_hit_behavior_support_gain"], 0.0)

    def test_choose_best_move_continues_on_strong_continuation_edge(self):
        engine = DaVinciDecisionEngine()
        all_moves = [
            {
                "expected_value": 1.45,
                "win_probability": 0.58,
                "continuation_value": 0.72,
                "continuation_likelihood": 0.76,
                "attackability_after_hit": 0.68,
                "post_hit_continue_score": 0.92,
                "post_hit_stop_score": 0.54,
                "post_hit_continue_margin": 0.38,
                "post_hit_best_gap": 0.31,
                "post_hit_top_k_continue_margin": 0.29,
                "post_hit_top_k_support_ratio": 1.0,
            },
            {
                "expected_value": 0.71,
                "win_probability": 0.49,
                "continuation_value": 0.22,
                "continuation_likelihood": 0.55,
                "attackability_after_hit": 0.42,
                "post_hit_continue_score": 0.28,
                "post_hit_stop_score": 0.24,
                "post_hit_continue_margin": 0.04,
                "post_hit_best_gap": 0.09,
                "post_hit_top_k_continue_margin": 0.02,
                "post_hit_top_k_support_ratio": 0.67,
            },
        ]

        best_move, summary = engine.choose_best_move(
            all_moves,
            risk_factor=engine.calculate_risk_factor(2),
            my_hidden_count=2,
        )

        self.assertIsNotNone(best_move)
        self.assertFalse(summary["recommend_stop"])
        self.assertGreater(summary["best_continuation_likelihood"], 0.70)
        self.assertGreater(summary["continue_score"], summary["stop_score"])
        self.assertGreater(summary["continue_margin"], 0.0)

    def test_choose_best_move_uses_negative_post_hit_margin_as_rollout_pressure(self):
        engine = DaVinciDecisionEngine()
        all_moves = [
            {
                "expected_value": 0.84,
                "win_probability": 0.56,
                "continuation_value": 0.19,
                "continuation_likelihood": 0.63,
                "attackability_after_hit": 0.72,
                "post_hit_continue_score": 0.22,
                "post_hit_stop_score": 0.58,
                "post_hit_continue_margin": -0.36,
                "post_hit_best_gap": 0.02,
                "post_hit_top_k_continue_margin": 0.0,
                "post_hit_top_k_support_ratio": 0.0,
            },
        ]

        best_move, summary = engine.choose_best_move(
            all_moves,
            risk_factor=engine.calculate_risk_factor(2),
            my_hidden_count=2,
        )

        self.assertIsNone(best_move)
        self.assertTrue(summary["recommend_stop"])
        self.assertGreater(summary["decision_score_breakdown"]["rollout_pressure"], 0.0)
        self.assertEqual(summary["decision_score_breakdown"]["attackability_pressure"], 0.0)
        self.assertLess(summary["continue_margin"], 0.0)

    def test_fixed_decision_case_runner_covers_rollout_stop_edge(self):
        case = FixedDecisionCase(
            name="rollout_stop_edge",
            my_hidden_count=2,
            all_moves=[
                {
                    "expected_value": 0.84,
                    "win_probability": 0.56,
                    "continuation_value": 0.19,
                    "continuation_likelihood": 0.63,
                    "attackability_after_hit": 0.72,
                    "post_hit_continue_score": 0.22,
                    "post_hit_stop_score": 0.58,
                    "post_hit_continue_margin": -0.36,
                    "post_hit_best_gap": 0.02,
                    "post_hit_top_k_continue_margin": 0.0,
                    "post_hit_top_k_support_ratio": 0.0,
                },
            ],
            checks=(assert_stop_summary,),
        )

        best_move, summary = run_decision_case(case)

        self.assertIsNone(best_move)
        self.assertGreater(summary["decision_score_breakdown"]["rollout_pressure"], 0.0)

    def test_fixed_decision_case_runner_covers_fragile_positive_stop_edge(self):
        case = FixedDecisionCase(
            name="fragile_positive_stop_edge",
            my_hidden_count=2,
            all_moves=[
                {
                    "expected_value": 0.81,
                    "win_probability": 0.57,
                    "continuation_value": 0.16,
                    "continuation_likelihood": 0.61,
                    "attackability_after_hit": 0.74,
                    "post_hit_continue_score": 0.35,
                    "post_hit_stop_score": 0.24,
                    "post_hit_continue_margin": 0.11,
                    "post_hit_best_gap": 0.02,
                    "post_hit_top_k_continue_margin": 0.05,
                    "post_hit_top_k_support_ratio": 0.33,
                },
            ],
            checks=(
                assert_stop_summary,
                assert_positive_breakdown("fragile_rollout_pressure"),
            ),
        )

        best_move, summary = run_decision_case(case)

        self.assertIsNone(best_move)
        self.assertEqual(summary["decision_score_breakdown"]["rollout_pressure"], 0.0)

    def test_fixed_decision_case_runner_covers_weak_top_k_support_stop_edge(self):
        case = FixedDecisionCase(
            name="weak_top_k_support_stop_edge",
            my_hidden_count=2,
            all_moves=[
                {
                    "expected_value": 0.80,
                    "win_probability": 0.57,
                    "continuation_value": 0.18,
                    "continuation_likelihood": 0.64,
                    "attackability_after_hit": 0.76,
                    "post_hit_continue_score": 0.41,
                    "post_hit_stop_score": 0.24,
                    "post_hit_continue_margin": 0.17,
                    "post_hit_best_gap": 0.30,
                    "post_hit_top_k_continue_margin": 0.03,
                    "post_hit_top_k_support_ratio": 0.33,
                },
            ],
            checks=(
                assert_stop_summary,
                assert_positive_breakdown("top_k_rollout_pressure"),
            ),
        )

        best_move, summary = run_decision_case(case)

        self.assertIsNone(best_move)
        self.assertEqual(summary["decision_score_breakdown"]["fragile_rollout_pressure"], 0.0)

    def test_choose_best_move_penalizes_fragile_positive_post_hit_gap(self):
        engine = DaVinciDecisionEngine()
        all_moves = [
            {
                "expected_value": 0.81,
                "win_probability": 0.57,
                "continuation_value": 0.16,
                "continuation_likelihood": 0.61,
                "attackability_after_hit": 0.74,
                "post_hit_continue_score": 0.35,
                "post_hit_stop_score": 0.24,
                "post_hit_continue_margin": 0.11,
                "post_hit_best_gap": 0.02,
                "post_hit_top_k_continue_margin": 0.05,
                "post_hit_top_k_support_ratio": 0.33,
            },
        ]

        best_move, summary = engine.choose_best_move(
            all_moves,
            risk_factor=engine.calculate_risk_factor(2),
            my_hidden_count=2,
        )

        self.assertIsNone(best_move)
        self.assertTrue(summary["recommend_stop"])
        self.assertGreater(summary["decision_score_breakdown"]["fragile_rollout_pressure"], 0.0)
        self.assertEqual(summary["decision_score_breakdown"]["rollout_pressure"], 0.0)
        self.assertLess(summary["continue_margin"], 0.0)

    def test_choose_best_move_penalizes_weak_top_k_followup_support(self):
        engine = DaVinciDecisionEngine()
        all_moves = [
            {
                "expected_value": 0.80,
                "win_probability": 0.57,
                "continuation_value": 0.18,
                "continuation_likelihood": 0.64,
                "attackability_after_hit": 0.76,
                "post_hit_continue_score": 0.41,
                "post_hit_stop_score": 0.24,
                "post_hit_continue_margin": 0.17,
                "post_hit_best_gap": 0.30,
                "post_hit_top_k_continue_margin": 0.03,
                "post_hit_top_k_support_ratio": 0.33,
            },
        ]

        best_move, summary = engine.choose_best_move(
            all_moves,
            risk_factor=engine.calculate_risk_factor(2),
            my_hidden_count=2,
        )

        self.assertIsNone(best_move)
        self.assertTrue(summary["recommend_stop"])
        self.assertGreater(summary["decision_score_breakdown"]["top_k_rollout_pressure"], 0.0)
        self.assertEqual(summary["decision_score_breakdown"]["fragile_rollout_pressure"], 0.0)
        self.assertLess(summary["continue_margin"], 0.0)

    def test_choose_best_move_stops_when_attackability_cannot_support_pressing(self):
        engine = DaVinciDecisionEngine()
        all_moves = [
            {
                "expected_value": 0.79,
                "win_probability": 0.55,
                "continuation_value": 0.18,
                "continuation_likelihood": 0.51,
                "attackability_after_hit": 0.05,
            },
        ]

        best_move, summary = engine.choose_best_move(
            all_moves,
            risk_factor=engine.calculate_risk_factor(2),
            my_hidden_count=2,
        )

        self.assertIsNone(best_move)
        self.assertTrue(summary["recommend_stop"])
        self.assertGreater(summary["stop_score"], summary["stop_threshold"])
        self.assertLess(summary["continue_margin"], 0.0)

    def test_fixed_decision_case_runner_covers_low_attackability_stop_edge(self):
        case = FixedDecisionCase(
            name="low_attackability_stop_edge",
            my_hidden_count=2,
            all_moves=[
                {
                    "expected_value": 0.79,
                    "win_probability": 0.55,
                    "continuation_value": 0.18,
                    "continuation_likelihood": 0.51,
                    "attackability_after_hit": 0.05,
                },
            ],
            checks=(
                assert_stop_summary,
                assert_positive_breakdown("attackability_pressure"),
            ),
        )

        best_move, summary = run_decision_case(case)

        self.assertIsNone(best_move)
        self.assertEqual(summary["decision_score_breakdown"]["rollout_pressure"], 0.0)


if __name__ == "__main__":
    unittest.main()
