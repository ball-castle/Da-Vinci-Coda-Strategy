import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.engine import BehavioralLikelihoodModel, GameController
from app.core.state import CardSlot, GameState, GuessAction, PlayerState


class BehavioralLikelihoodModelTests(unittest.TestCase):
    def test_failed_guess_prefers_interval_consistent_world(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=1, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=8, is_revealed=True),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=3, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=6, is_revealed=True),
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
                )
            ],
        )
        model = BehavioralLikelihoodModel()
        signals = model.build_guess_signals(game_state)

        consistent = model.score_hypothesis({"opp": {1: ("W", 5)}}, signals, game_state)
        inconsistent = model.score_hypothesis({"opp": {1: ("W", 10)}}, signals, game_state)

        self.assertGreater(consistent, inconsistent)

    def test_continue_signal_prefers_attackable_world(self):
        model = BehavioralLikelihoodModel()

        tight_world = GameState(
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
                        CardSlot(slot_index=0, color="W", value=None, is_revealed=False),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=2, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=4, is_revealed=True),
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
        loose_world = GameState(
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
                        CardSlot(slot_index=0, color="W", value=None, is_revealed=False),
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
                    target_slot_index=0,
                    guessed_color="W",
                    guessed_value=7,
                    result=True,
                    continued_turn=True,
                )
            ],
        )

        tight_signals = model.build_guess_signals(tight_world)
        loose_signals = model.build_guess_signals(loose_world)
        tight_score = model.score_hypothesis(
            {"opp": {0: ("W", 7)}, "side": {1: ("W", 3)}},
            tight_signals,
            tight_world,
        )
        loose_score = model.score_hypothesis(
            {"opp": {0: ("W", 7)}, "side": {1: ("W", 6)}},
            loose_signals,
            loose_world,
        )

        self.assertGreater(tight_score, loose_score)

    def test_target_player_selection_prefers_more_attackable_target(self):
        model = BehavioralLikelihoodModel()

        focused_world = GameState(
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
                )
            ],
        )
        distracted_world = GameState(
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
                        CardSlot(slot_index=0, color="B", value=7, is_revealed=True),
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
                    guessed_value=2,
                    result=False,
                )
            ],
        )

        focused_signals = model.build_guess_signals(focused_world)
        distracted_signals = model.build_guess_signals(distracted_world)
        focused_score = model.score_hypothesis(
            {"opp": {1: ("W", 2)}, "side": {1: ("W", 10)}},
            focused_signals,
            focused_world,
        )
        distracted_score = model.score_hypothesis(
            {"opp": {1: ("W", 2)}, "side": {1: ("W", 8)}},
            distracted_signals,
            distracted_world,
        )

        self.assertGreater(focused_score, distracted_score)

    def test_target_slot_selection_prefers_tighter_slot_on_same_player(self):
        model = BehavioralLikelihoodModel()
        game_state = GameState(
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
                        CardSlot(slot_index=3, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=4, color="B", value=10, is_revealed=True),
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
                )
            ],
        )

        signals = model.build_guess_signals(game_state)
        tight_slot_score = model.score_hypothesis(
            {"opp": {1: ("W", 2), 2: ("W", 3), 3: ("W", 9)}},
            signals,
            game_state,
        )
        loose_slot_score = model.score_hypothesis(
            {"opp": {1: ("W", 2), 2: ("W", 8), 3: ("W", 9)}},
            signals,
            game_state,
        )

        self.assertGreater(tight_slot_score, loose_slot_score)

    def test_target_player_selection_rewards_same_player_focus(self):
        model = BehavioralLikelihoodModel()

        focused_world = GameState(
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
        )
        switched_world = GameState(
            self_player_id="me",
            target_player_id="opp",
            players=focused_world.players,
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
        )

        hypothesis = {"opp": {1: ("W", 3)}, "side": {1: ("W", 10)}}
        focused_signal = model.build_guess_signals(focused_world)["me"][-1]
        switched_signal = model.build_guess_signals(switched_world)["me"][-1]

        focused_weight = model._score_target_player_selection(
            focused_world,
            hypothesis,
            focused_signal,
        )
        switched_weight = model._score_target_player_selection(
            switched_world,
            hypothesis,
            switched_signal,
        )

        self.assertGreater(focused_weight, switched_weight)

    def test_target_slot_selection_rewards_retry_after_failed_same_slot(self):
        model = BehavioralLikelihoodModel()
        retry_world = GameState(
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
        )
        fresh_world = GameState(
            self_player_id="me",
            target_player_id="opp",
            players=retry_world.players,
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
        )

        hypothesis = {"opp": {1: ("W", 3), 2: ("W", 9)}}
        retry_signal = model.build_guess_signals(retry_world)["me"][-1]
        fresh_signal = model.build_guess_signals(fresh_world)["me"][-1]

        retry_weight = model._score_target_slot_selection(
            retry_world,
            hypothesis,
            retry_signal,
        )
        fresh_weight = model._score_target_slot_selection(
            fresh_world,
            hypothesis,
            fresh_signal,
        )

        self.assertGreater(retry_weight, fresh_weight)

    def test_target_value_selection_prefers_progressive_step_after_failed_guess(self):
        model = BehavioralLikelihoodModel()
        game_state = GameState(
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
        )
        jump_world = GameState(
            self_player_id="me",
            target_player_id="opp",
            players=game_state.players,
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
        )

        hypothesis = {"opp": {1: ("W", 7)}}
        progressive_signals = model.build_guess_signals(game_state)
        jump_signals = model.build_guess_signals(jump_world)
        progressive_score = model.score_hypothesis(
            hypothesis,
            progressive_signals,
            game_state,
        )
        jump_score = model.score_hypothesis(
            hypothesis,
            jump_signals,
            jump_world,
        )

        self.assertGreater(progressive_score, jump_score)

    def test_target_value_selection_prefers_same_color_sandwich_anchor(self):
        model = BehavioralLikelihoodModel()
        game_state = GameState(
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
        )

        preferred_hypothesis = {
            "me": {0: ("W", 4), 1: ("W", 6)},
            "opp": {1: ("W", 5)},
        }
        alternative_hypothesis = {
            "me": {0: ("W", 4), 1: ("W", 9)},
            "opp": {1: ("W", 5)},
        }
        guess_signals = model.build_guess_signals(game_state)

        preferred_score = model.score_hypothesis(
            preferred_hypothesis,
            guess_signals,
            game_state,
        )
        alternative_score = model.score_hypothesis(
            alternative_hypothesis,
            guess_signals,
            game_state,
        )

        self.assertGreater(preferred_score, alternative_score)

    def test_target_value_selection_prefers_center_probe_in_wide_gap(self):
        model = BehavioralLikelihoodModel()
        center_world = GameState(
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
        )
        edge_world = GameState(
            self_player_id="me",
            target_player_id="opp",
            players=center_world.players,
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
        )

        hypothesis = {"opp": {1: ("W", 5)}}
        center_score = model.score_hypothesis(
            hypothesis,
            model.build_guess_signals(center_world),
            center_world,
        )
        edge_score = model.score_hypothesis(
            hypothesis,
            model.build_guess_signals(edge_world),
            edge_world,
        )

        self.assertGreater(center_score, edge_score)

    def test_behavior_explanation_labels_same_color_anchor_sandwich(self):
        model = BehavioralLikelihoodModel()
        game_state = GameState(
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
        )
        hypothesis = {
            "me": {0: ("W", 4), 1: ("W", 6)},
            "opp": {1: ("W", 5)},
        }

        explanation = model.explain_guess_signals(
            hypothesis,
            model.build_guess_signals(game_state),
            game_state,
        )[0]

        self.assertEqual(
            explanation["value_selection"]["anchor"]["reason"],
            "same_color_sandwich_exact",
        )
        self.assertEqual(
            explanation["value_selection"]["dominant_signal"]["source"],
            "same_color_anchor",
        )

    def test_behavior_explanation_labels_wide_gap_center_probe(self):
        model = BehavioralLikelihoodModel()
        game_state = GameState(
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
        )
        hypothesis = {"opp": {1: ("W", 5)}}

        explanation = model.explain_guess_signals(
            hypothesis,
            model.build_guess_signals(game_state),
            game_state,
        )[0]

        self.assertEqual(
            explanation["value_selection"]["boundary"]["reason"],
            "wide_gap_center_probe",
        )
        self.assertEqual(
            explanation["value_selection"]["dominant_signal"]["source"],
            "local_boundary",
        )


class GameControllerOutputTests(unittest.TestCase):
    def test_controller_returns_decision_summary_and_reasoning(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=8, is_revealed=True),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=3, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=5, is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )

        result = GameController(game_state).run_turn()
        self.assertIn("decision_summary", result)
        self.assertIn("behavior_debug", result)
        self.assertEqual(
            result["behavior_debug"]["hypothesis_source"],
            "target_slot_top_k_posterior_with_map_context",
        )
        self.assertEqual(result["behavior_debug"]["aggregation_top_k"], 3)
        self.assertEqual(result["behavior_debug"]["signal_count"], 0)
        self.assertEqual(result["behavior_debug"]["map_signals"], [])
        self.assertIn("behavior_guidance_profile", result)
        self.assertEqual(result["behavior_guidance_profile"]["guidance_multiplier"], 1.0)
        self.assertEqual(result["behavior_guidance_profile"]["source_support_progressive"], 0.0)
        self.assertEqual(result["behavior_guidance_profile"]["source_support_same_color_anchor"], 0.0)
        self.assertEqual(result["behavior_guidance_profile"]["source_support_local_boundary"], 0.0)
        self.assertIn("evaluated_move_count", result["decision_summary"])
        self.assertIn("stop_score", result["decision_summary"])
        self.assertIn("continue_score", result["decision_summary"])
        self.assertIn("continue_margin", result["decision_summary"])
        self.assertIn("decision_score_breakdown", result["decision_summary"])
        self.assertIn("best_post_hit_guidance_multiplier", result["decision_summary"])
        self.assertIn("best_post_hit_guidance_support", result["decision_summary"])
        self.assertIn("best_post_hit_guidance_stable_ratio", result["decision_summary"])
        self.assertIn("best_post_hit_guidance_signal_count", result["decision_summary"])
        self.assertIn("best_post_hit_top_k_continue_margin", result["decision_summary"])
        self.assertIn("best_post_hit_top_k_expected_continue_margin", result["decision_summary"])
        self.assertIn("best_post_hit_top_k_support_ratio", result["decision_summary"])
        self.assertIn("best_post_hit_top_k_expected_support_ratio", result["decision_summary"])
        if result["top_moves"]:
            self.assertIn("recommendation_reason", result["top_moves"][0])
            self.assertIn("score_breakdown", result["top_moves"][0])
            self.assertIn("immediate_expected_value", result["top_moves"][0])
            self.assertIn("post_hit_continuation_value", result["top_moves"][0])
            self.assertIn("post_hit_continue_score", result["top_moves"][0])
            self.assertIn("post_hit_stop_score", result["top_moves"][0])
            self.assertIn("post_hit_continue_margin", result["top_moves"][0])
            self.assertIn("post_hit_best_gap", result["top_moves"][0])
            self.assertIn("post_hit_guidance_multiplier", result["top_moves"][0])
            self.assertIn("post_hit_guidance_support", result["top_moves"][0])
            self.assertIn("post_hit_guidance_stable_ratio", result["top_moves"][0])
            self.assertIn("post_hit_guidance_signal_count", result["top_moves"][0])
            self.assertIn("post_hit_guidance_debug", result["top_moves"][0])
            self.assertIn("post_hit_behavior_support_adjustment", result["top_moves"][0])
            self.assertIn("post_hit_behavior_support_gain", result["top_moves"][0])
            self.assertIn("post_hit_behavior_fragility_drag", result["top_moves"][0])
            self.assertIn(
                "rebuilt_delta_from_base",
                result["top_moves"][0]["post_hit_guidance_debug"],
            )
            self.assertIn(
                "blended_delta_from_base",
                result["top_moves"][0]["post_hit_guidance_debug"],
            )
            self.assertIn("post_hit_top_k_expected_continue_margin", result["top_moves"][0])
            self.assertIn("post_hit_top_k_continue_margin", result["top_moves"][0])
            self.assertIn("post_hit_top_k_expected_support_ratio", result["top_moves"][0])
            self.assertIn("post_hit_top_k_support_ratio", result["top_moves"][0])

    def test_controller_returns_behavior_debug_for_guess_actions(self):
        game_state = GameState(
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
        )

        result = GameController(game_state).run_turn()

        self.assertEqual(
            result["behavior_debug"]["hypothesis_source"],
            "target_slot_top_k_posterior_with_map_context",
        )
        self.assertEqual(result["behavior_debug"]["aggregation_top_k"], 3)
        self.assertEqual(result["behavior_debug"]["signal_count"], 1)
        self.assertEqual(len(result["behavior_debug"]["map_signals"]), 1)
        self.assertEqual(len(result["behavior_debug"]["signals"]), 1)
        self.assertIn("behavior_guidance_profile", result)
        self.assertGreater(result["behavior_guidance_profile"]["signal_count"], 0.0)
        self.assertIn("component_weights", result["behavior_debug"]["signals"][0])
        self.assertIn("value_selection", result["behavior_debug"]["signals"][0])
        self.assertIn("candidate_explanations", result["behavior_debug"]["signals"][0])
        self.assertIn("source_support_same_color_anchor", result["behavior_guidance_profile"])
        self.assertIn("source_support_local_boundary", result["behavior_guidance_profile"])
        self.assertEqual(
            result["behavior_debug"]["signals"][0]["aggregation_mode"],
            "target_slot_top_k_posterior",
        )
        self.assertIn(
            "dominant_signal",
            result["behavior_debug"]["signals"][0]["value_selection"],
        )
        self.assertEqual(
            result["behavior_debug"]["signals"][0]["value_selection"]["mode"],
            "target_slot_top_k_posterior",
        )
        self.assertGreaterEqual(
            result["behavior_debug"]["signals"][0]["candidate_count"],
            1,
        )
        self.assertIn("best_behavior_guidance_multiplier", result["decision_summary"])
        self.assertIn("best_behavior_guidance_signal_count", result["decision_summary"])
        self.assertIn("best_behavior_guidance_support", result["decision_summary"])
        self.assertIn("best_behavior_guidance_stable_ratio", result["decision_summary"])
        self.assertIn("best_behavior_match_multiplier", result["decision_summary"])
        self.assertIn("best_behavior_match_bonus", result["decision_summary"])
        self.assertIn("best_behavior_match_support", result["decision_summary"])
        self.assertIn("best_behavior_match_decision_bonus", result["decision_summary"])
        self.assertIn("best_behavior_match_decision_structure_adjustment", result["decision_summary"])
        self.assertIn("best_behavior_match_ranking_bonus", result["decision_summary"])
        self.assertIn("best_behavior_match_net_structure", result["decision_summary"])
        self.assertIn("best_behavior_match_structure_adjustment", result["decision_summary"])
        self.assertIn("best_behavior_match_candidate_confidence", result["decision_summary"])
        self.assertIn("best_behavior_match_component_support", result["decision_summary"])
        self.assertIn("best_behavior_match_component_strength", result["decision_summary"])
        self.assertIn("best_behavior_match_component_penalty", result["decision_summary"])
        self.assertIn("best_behavior_match_context_focus", result["decision_summary"])
        self.assertIn("best_behavior_rollout_pressure", result["decision_summary"])
        self.assertIn("best_post_hit_behavior_support_adjustment", result["decision_summary"])
        self.assertIn("best_post_hit_behavior_support_gain", result["decision_summary"])
        self.assertIn("best_post_hit_behavior_fragility_drag", result["decision_summary"])
        self.assertIn(
            "behavior_match_decision_structure_adjustment",
            result["decision_summary"]["decision_score_breakdown"],
        )
        self.assertIn(
            "post_hit_behavior_support_adjustment",
            result["decision_summary"]["decision_score_breakdown"],
        )
        self.assertIn(
            "post_hit_behavior_support_gain",
            result["decision_summary"]["decision_score_breakdown"],
        )
        self.assertIn(
            "post_hit_behavior_fragility_drag",
            result["decision_summary"]["decision_score_breakdown"],
        )

    def test_controller_aggregates_behavior_debug_across_top_k_posterior_candidates(self):
        game_state = GameState(
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
        )

        result = GameController(game_state).run_turn()
        signal_debug = result["behavior_debug"]["signals"][0]

        self.assertEqual(signal_debug["candidate_count"], 3)
        self.assertGreater(signal_debug["covered_probability"], 0.0)
        self.assertEqual(len(signal_debug["candidate_explanations"]), 3)
        self.assertIn("reason_support", signal_debug["value_selection"])
        self.assertIn("source_support", signal_debug["value_selection"])
        self.assertIn("map_explanation", signal_debug)
        self.assertIn("dominant_signal", signal_debug["value_selection"])
        self.assertGreater(
            signal_debug["value_selection"]["dominant_signal"]["posterior_support"],
            0.0,
        )
        self.assertGreater(
            result["behavior_guidance_profile"]["guidance_multiplier"],
            1.0,
        )
        self.assertGreaterEqual(
            result["behavior_guidance_profile"]["source_support_same_color_anchor"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
