import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


class StopThresholdTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
