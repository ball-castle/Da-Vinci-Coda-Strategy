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
        self.assertIn("evaluated_move_count", result["decision_summary"])
        self.assertIn("stop_score", result["decision_summary"])
        self.assertIn("continue_score", result["decision_summary"])
        self.assertIn("continue_margin", result["decision_summary"])
        self.assertIn("decision_score_breakdown", result["decision_summary"])
        self.assertIn("best_post_hit_top_k_continue_margin", result["decision_summary"])
        self.assertIn("best_post_hit_top_k_support_ratio", result["decision_summary"])
        if result["top_moves"]:
            self.assertIn("recommendation_reason", result["top_moves"][0])
            self.assertIn("score_breakdown", result["top_moves"][0])
            self.assertIn("immediate_expected_value", result["top_moves"][0])
            self.assertIn("post_hit_continuation_value", result["top_moves"][0])
            self.assertIn("post_hit_continue_score", result["top_moves"][0])
            self.assertIn("post_hit_stop_score", result["top_moves"][0])
            self.assertIn("post_hit_continue_margin", result["top_moves"][0])
            self.assertIn("post_hit_best_gap", result["top_moves"][0])
            self.assertIn("post_hit_top_k_continue_margin", result["top_moves"][0])
            self.assertIn("post_hit_top_k_support_ratio", result["top_moves"][0])


if __name__ == "__main__":
    unittest.main()
