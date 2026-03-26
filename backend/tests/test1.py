import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.fixed_controller_cases import (
    FixedControllerCase,
    all_serialized_candidate_cards,
    assert_card_absent_from_slot_candidates,
    assert_draw_color_dominant_factor,
    assert_draw_color_recommendation,
    assert_serialized_probability_mass,
    assert_slot_candidate_count,
    assert_slot_candidate_set,
    run_controller_case,
)
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

    def test_target_player_selection_prefers_finishing_weaker_target(self):
        model = BehavioralLikelihoodModel()

        finishing_world = GameState(
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
                        CardSlot(slot_index=2, color="B", value=10, is_revealed=True),
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
        spread_world = GameState(
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
                        CardSlot(slot_index=3, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=4, color="B", value=11, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=7, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=10, is_revealed=True),
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

        finishing_signal = model.build_guess_signals(finishing_world)["me"][-1]
        spread_signal = model.build_guess_signals(spread_world)["me"][-1]
        finishing_hypothesis = {"opp": {1: ("W", 2)}, "side": {1: ("W", 8)}}
        spread_hypothesis = {"opp": {1: ("W", 2), 3: ("W", 8)}, "side": {1: ("W", 8)}}

        finishing_score = model._score_target_player_selection(
            finishing_world,
            finishing_hypothesis,
            finishing_signal,
        )
        spread_score = model._score_target_player_selection(
            spread_world,
            spread_hypothesis,
            spread_signal,
        )

        self.assertGreater(finishing_score, spread_score)

    def test_target_player_selection_prefers_recently_collapsed_target(self):
        model = BehavioralLikelihoodModel()

        collapse_world = GameState(
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
                        CardSlot(slot_index=2, color="B", value=None, is_revealed=False),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=7, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=10, is_revealed=True),
                    ],
                ),
            },
            actions=[
                GuessAction(
                    guesser_id="side",
                    target_player_id="opp",
                    target_slot_index=2,
                    guessed_color="B",
                    guessed_value=4,
                    result=False,
                    revealed_player_id="opp",
                    revealed_slot_index=2,
                    revealed_color="B",
                    revealed_value=4,
                ),
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=2,
                    result=False,
                ),
            ],
        )
        settled_world = GameState(
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
                        CardSlot(slot_index=2, color="B", value=10, is_revealed=True),
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
            ],
        )

        collapse_signal = model.build_guess_signals(collapse_world)["me"][-1]
        settled_signal = model.build_guess_signals(settled_world)["me"][-1]
        hypothesis = {"opp": {1: ("W", 2)}, "side": {1: ("W", 8)}}

        collapse_score = model._score_target_player_selection(
            collapse_world,
            hypothesis,
            collapse_signal,
        )
        settled_score = model._score_target_player_selection(
            settled_world,
            hypothesis,
            settled_signal,
        )

        self.assertGreater(collapse_score, settled_score)

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

    def test_target_player_selection_rewards_switch_after_failed_target_when_new_target_is_tighter(self):
        model = BehavioralLikelihoodModel()

        stuck_world = GameState(
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
                ),
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=2,
                    guessed_color="W",
                    guessed_value=8,
                    result=False,
                ),
            ],
        )
        switched_world = GameState(
            self_player_id="me",
            target_player_id="opp",
            players=stuck_world.players,
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
                    target_player_id="side",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=8,
                    result=False,
                ),
            ],
        )

        hypothesis = {"opp": {1: ("W", 2), 2: ("W", 8)}, "side": {1: ("W", 8)}}
        stuck_signal = model.build_guess_signals(stuck_world)["me"][-1]
        switched_signal = model.build_guess_signals(switched_world)["me"][-1]

        stuck_weight = model._score_target_player_selection(
            stuck_world,
            hypothesis,
            stuck_signal,
        )
        switched_weight = model._score_target_player_selection(
            switched_world,
            hypothesis,
            switched_signal,
        )

        self.assertGreater(switched_weight, stuck_weight)

    def test_target_player_selection_prefers_staying_on_same_player_after_confident_hit(self):
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
                        CardSlot(slot_index=0, color="B", value=7, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=10, is_revealed=True),
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
                    result=True,
                    continued_turn=True,
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
                    target_player_id="opp",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=2,
                    result=True,
                    continued_turn=True,
                ),
                GuessAction(
                    guesser_id="me",
                    target_player_id="side",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=8,
                    result=False,
                ),
            ],
        )

        hypothesis = {"opp": {1: ("W", 3)}, "side": {1: ("W", 8)}}
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

    def test_target_player_selection_rewards_continuous_switch_after_failed_guess(self):
        model = BehavioralLikelihoodModel()

        continuous_world = GameState(
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
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=2,
                    result=False,
                ),
                GuessAction(
                    guesser_id="me",
                    target_player_id="side",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=3,
                    result=False,
                ),
            ],
        )
        disjoint_world = GameState(
            self_player_id="me",
            target_player_id="opp",
            players=continuous_world.players,
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
                    target_player_id="side",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=8,
                    result=False,
                ),
            ],
        )

        hypothesis = {"opp": {1: ("W", 2), 2: ("W", 8)}, "side": {1: ("W", 3)}}
        continuous_signal = model.build_guess_signals(continuous_world)["me"][-1]
        disjoint_signal = model.build_guess_signals(disjoint_world)["me"][-1]

        continuous_weight = model._score_target_player_selection(
            continuous_world,
            hypothesis,
            continuous_signal,
        )
        disjoint_weight = model._score_target_player_selection(
            disjoint_world,
            hypothesis,
            disjoint_signal,
        )

        self.assertGreater(continuous_weight, disjoint_weight)

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

    def test_target_slot_selection_prefers_adjacent_follow_after_confident_hit(self):
        model = BehavioralLikelihoodModel()
        adjacent_world = GameState(
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
                    result=True,
                    continued_turn=True,
                ),
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=2,
                    guessed_color="W",
                    guessed_value=5,
                    result=False,
                ),
            ],
        )
        far_world = GameState(
            self_player_id="me",
            target_player_id="opp",
            players=adjacent_world.players,
            actions=[
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=2,
                    result=True,
                    continued_turn=True,
                ),
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=3,
                    guessed_color="W",
                    guessed_value=8,
                    result=False,
                ),
            ],
        )

        hypothesis = {"opp": {1: ("W", 2), 2: ("W", 5), 3: ("W", 8)}}
        adjacent_signal = model.build_guess_signals(adjacent_world)["me"][-1]
        far_signal = model.build_guess_signals(far_world)["me"][-1]

        adjacent_weight = model._score_target_slot_selection(
            adjacent_world,
            hypothesis,
            adjacent_signal,
        )
        far_weight = model._score_target_slot_selection(
            far_world,
            hypothesis,
            far_signal,
        )

        self.assertGreater(adjacent_weight, far_weight)

    def test_target_slot_selection_prefers_adjacent_probe_after_failed_guess(self):
        model = BehavioralLikelihoodModel()
        adjacent_world = GameState(
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
                ),
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=2,
                    guessed_color="W",
                    guessed_value=5,
                    result=False,
                ),
            ],
        )
        far_world = GameState(
            self_player_id="me",
            target_player_id="opp",
            players=adjacent_world.players,
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
                    target_slot_index=3,
                    guessed_color="W",
                    guessed_value=8,
                    result=False,
                ),
            ],
        )

        hypothesis = {"opp": {1: ("W", 2), 2: ("W", 5), 3: ("W", 8)}}
        adjacent_signal = model.build_guess_signals(adjacent_world)["me"][-1]
        far_signal = model.build_guess_signals(far_world)["me"][-1]

        adjacent_weight = model._score_target_slot_selection(
            adjacent_world,
            hypothesis,
            adjacent_signal,
        )
        far_weight = model._score_target_slot_selection(
            far_world,
            hypothesis,
            far_signal,
        )

        self.assertGreater(adjacent_weight, far_weight)

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

    def test_target_value_selection_prefers_local_step_after_confident_hit(self):
        model = BehavioralLikelihoodModel()
        local_step_world = GameState(
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
                        CardSlot(slot_index=3, color="B", value=9, is_revealed=True),
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
                    result=True,
                    continued_turn=True,
                ),
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=2,
                    guessed_color="W",
                    guessed_value=3,
                    result=False,
                ),
            ],
        )
        jump_world = GameState(
            self_player_id="me",
            target_player_id="opp",
            players=local_step_world.players,
            actions=[
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=2,
                    result=True,
                    continued_turn=True,
                ),
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=2,
                    guessed_color="W",
                    guessed_value=8,
                    result=False,
                ),
            ],
        )

        local_step_score = model.score_hypothesis(
            {"opp": {1: ("W", 2), 2: ("W", 3)}},
            model.build_guess_signals(local_step_world),
            local_step_world,
        )
        jump_score = model.score_hypothesis(
            {"opp": {1: ("W", 2), 2: ("W", 8)}},
            model.build_guess_signals(jump_world),
            jump_world,
        )

        self.assertGreater(local_step_score, jump_score)

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
    def test_fixed_controller_case_runner_covers_public_reveal_collapse(self):
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
                        CardSlot(slot_index=2, color="B", value=4, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=7, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=10, is_revealed=True),
                    ],
                ),
            },
            actions=[
                GuessAction(
                    guesser_id="side",
                    target_player_id="opp",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=6,
                    result=False,
                    revealed_player_id="side",
                    revealed_slot_index=1,
                    revealed_color="W",
                    revealed_value=5,
                )
            ],
        )

        result = run_controller_case(
            FixedControllerCase(
                name="public_reveal_collapse",
                game_state=game_state,
                checks=(assert_serialized_probability_mass,),
            )
        )

        self.assertNotIn(["W", 5], all_serialized_candidate_cards(result))

    def test_fixed_controller_case_runner_covers_failed_guess_interval_collapse(self):
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
                        CardSlot(slot_index=2, color="B", value=4, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=7, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=10, is_revealed=True),
                    ],
                ),
            },
            actions=[
                GuessAction(
                    guesser_id="side",
                    target_player_id="opp",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=6,
                    result=False,
                    revealed_player_id="side",
                    revealed_slot_index=1,
                    revealed_color="W",
                    revealed_value=5,
                )
            ],
        )

        result = run_controller_case(
            FixedControllerCase(
                name="failed_guess_interval_collapse",
                game_state=game_state,
                checks=(
                    assert_serialized_probability_mass,
                    assert_slot_candidate_set(
                        player_id="opp",
                        slot_index=1,
                        expected_cards=(
                            ("W", "-"),
                            ("W", 1),
                            ("W", 2),
                            ("W", 3),
                        ),
                    ),
                ),
            )
        )

        self.assertNotIn(["W", 6], all_serialized_candidate_cards(result))

    def test_fixed_controller_case_runner_covers_successful_guess_exact_fixation(self):
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
                        CardSlot(slot_index=2, color="B", value=4, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=7, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=10, is_revealed=True),
                    ],
                ),
            },
            actions=[
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=3,
                    result=True,
                )
            ],
        )

        result = run_controller_case(
            FixedControllerCase(
                name="successful_guess_exact_fixation",
                game_state=game_state,
                checks=(
                    assert_serialized_probability_mass,
                    assert_slot_candidate_set(
                        player_id="opp",
                        slot_index=1,
                        expected_cards=(
                            ("W", 3),
                        ),
                    ),
                ),
            )
        )

        self.assertIn(["W", 3], all_serialized_candidate_cards(result))

    def test_fixed_controller_case_runner_propagates_public_reveal_to_non_target_players(self):
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
                "third": PlayerState(
                    player_id="third",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=2, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=10, is_revealed=True),
                    ],
                ),
            },
            actions=[
                GuessAction(
                    guesser_id="opp",
                    target_player_id="third",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=6,
                    result=False,
                    revealed_player_id="third",
                    revealed_slot_index=1,
                    revealed_color="W",
                    revealed_value=5,
                )
            ],
        )

        result = run_controller_case(
            FixedControllerCase(
                name="public_reveal_propagates_to_non_target",
                game_state=game_state,
                checks=(
                    assert_serialized_probability_mass,
                    assert_card_absent_from_slot_candidates(
                        player_id="side",
                        slot_index=1,
                        card=("W", 5),
                    ),
                ),
            )
        )

        self.assertNotIn(["W", 5], all_serialized_candidate_cards(result))

    def test_fixed_controller_case_runner_shrinks_non_target_candidate_width_after_public_reveal(self):
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
                "third": PlayerState(
                    player_id="third",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=2, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=10, is_revealed=True),
                    ],
                ),
            },
            actions=[
                GuessAction(
                    guesser_id="opp",
                    target_player_id="third",
                    target_slot_index=1,
                    guessed_color="W",
                    guessed_value=6,
                    result=False,
                    revealed_player_id="third",
                    revealed_slot_index=1,
                    revealed_color="W",
                    revealed_value=5,
                )
            ],
        )

        result = run_controller_case(
            FixedControllerCase(
                name="public_reveal_shrinks_non_target_width",
                game_state=game_state,
                checks=(
                    assert_serialized_probability_mass,
                    assert_slot_candidate_count(
                        player_id="side",
                        slot_index=1,
                        expected_count=3,
                    ),
                ),
            )
        )

        self.assertNotIn(["W", 5], all_serialized_candidate_cards(result))

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
        self.assertIn("draw_color_summary", result)
        self.assertIn("draw_opening_plan", result)
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
        self.assertIn("best_post_hit_guidance_rebuild_applied", result["decision_summary"])
        self.assertIn("best_post_hit_guidance_rebuild_signal_count", result["decision_summary"])
        self.assertIn("best_post_hit_guidance_augmented_slot_count", result["decision_summary"])
        self.assertIn("best_post_hit_guidance_multiplier_delta", result["decision_summary"])
        self.assertIn("best_post_hit_guidance_source_shift", result["decision_summary"])
        self.assertIn("best_post_hit_guidance_source_shift_strength", result["decision_summary"])
        self.assertIn("best_post_hit_guidance_rebuild_reason", result["decision_summary"])
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
            self.assertIn(
                "rebuilt_dominant_source_shift",
                result["top_moves"][0]["post_hit_guidance_debug"],
            )
            self.assertIn(
                "blended_dominant_source_shift",
                result["top_moves"][0]["post_hit_guidance_debug"],
            )
            self.assertIn(
                "blended_dominant_source_shift_strength",
                result["top_moves"][0]["post_hit_guidance_debug"],
            )
            self.assertIn("post_hit_top_k_expected_continue_margin", result["top_moves"][0])
            self.assertIn("post_hit_top_k_continue_margin", result["top_moves"][0])
            self.assertIn("post_hit_top_k_expected_support_ratio", result["top_moves"][0])
            self.assertIn("post_hit_top_k_support_ratio", result["top_moves"][0])

    def test_controller_draw_color_summary_prefers_balancing_color(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=2, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=4, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=7, is_revealed=False),
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
            actions=[],
        )

        result = GameController(game_state).run_turn()
        draw_summary = result["draw_color_summary"]

        self.assertEqual(draw_summary["recommended_color"], "W")
        self.assertGreater(
            draw_summary["defense_balance_white"],
            draw_summary["defense_balance_black"],
        )

    def test_fixed_controller_case_runner_prefers_black_when_self_hand_is_white_heavy(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="W", value=1, is_revealed=False),
                        CardSlot(slot_index=1, color="W", value=3, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=5, is_revealed=False),
                        CardSlot(slot_index=3, color="B", value=8, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=2, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=9, is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )

        result = run_controller_case(
            FixedControllerCase(
                name="draw_color_black_when_white_heavy",
                game_state=game_state,
                checks=(assert_draw_color_recommendation("B"),),
            )
        )

        self.assertGreater(
            result["draw_color_summary"]["defense_balance_black"],
            result["draw_color_summary"]["defense_balance_white"],
        )

    def test_controller_draw_color_summary_uses_hidden_color_pressure_for_offense(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=2, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=5, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=7, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=1, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=9, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="W", value=3, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=10, is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )

        result = GameController(game_state).run_turn()
        draw_summary = result["draw_color_summary"]

        self.assertEqual(draw_summary["recommended_color"], "B")
        self.assertGreater(
            draw_summary["offense_pressure_black"],
            draw_summary["offense_pressure_white"],
        )
        self.assertEqual(draw_summary["dominant_factor"], "offense_pressure")

    def test_fixed_controller_case_runner_prefers_black_under_black_offense_pressure(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=2, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=5, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=7, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=1, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=9, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="W", value=3, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=10, is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )

        result = run_controller_case(
            FixedControllerCase(
                name="draw_color_black_under_black_offense_pressure",
                game_state=game_state,
                checks=(
                    assert_draw_color_recommendation("B"),
                    assert_draw_color_dominant_factor("offense_pressure"),
                ),
            )
        )

        self.assertGreater(
            result["draw_color_summary"]["offense_pressure_black"],
            result["draw_color_summary"]["offense_pressure_white"],
        )

    def test_controller_draw_color_summary_uses_hidden_color_pressure_for_white_offense(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=2, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=5, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=7, is_revealed=False),
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
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="W", value=3, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=10, is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )

        result = GameController(game_state).run_turn()
        draw_summary = result["draw_color_summary"]

        self.assertEqual(draw_summary["recommended_color"], "W")
        self.assertGreater(
            draw_summary["offense_pressure_white"],
            draw_summary["offense_pressure_black"],
        )
        self.assertEqual(draw_summary["dominant_factor"], "offense_pressure")

    def test_fixed_controller_case_runner_prefers_white_under_white_offense_pressure(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=2, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=5, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=7, is_revealed=False),
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
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="W", value=3, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=10, is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )

        result = run_controller_case(
            FixedControllerCase(
                name="draw_color_white_under_white_offense_pressure",
                game_state=game_state,
                checks=(
                    assert_draw_color_recommendation("W"),
                    assert_draw_color_dominant_factor("offense_pressure"),
                ),
            )
        )

        self.assertGreater(
            result["draw_color_summary"]["offense_pressure_white"],
            result["draw_color_summary"]["offense_pressure_black"],
        )

    def test_controller_draw_color_summary_uses_availability_pressure(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=2, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=5, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=7, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="W", value=1, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=3, is_revealed=True),
                        CardSlot(slot_index=2, color="W", value=9, is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )

        result = GameController(game_state).run_turn()
        draw_summary = result["draw_color_summary"]

        self.assertEqual(draw_summary["recommended_color"], "B")
        self.assertGreater(
            draw_summary["availability_pressure_black"],
            draw_summary["availability_pressure_white"],
        )

    def test_fixed_controller_case_runner_prefers_black_under_availability_pressure(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=2, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=5, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=7, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="W", value=1, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=3, is_revealed=True),
                        CardSlot(slot_index=2, color="W", value=9, is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )

        result = run_controller_case(
            FixedControllerCase(
                name="draw_color_black_under_availability_pressure",
                game_state=game_state,
                checks=(assert_draw_color_recommendation("B"),),
            )
        )

        self.assertGreater(
            result["draw_color_summary"]["availability_pressure_black"],
            result["draw_color_summary"]["availability_pressure_white"],
        )

    def test_controller_draw_color_summary_uses_target_attack_pressure(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=2, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=5, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=7, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=1, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=3, color="B", value=9, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="W", value=3, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=10, is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )

        result = GameController(game_state).run_turn()
        draw_summary = result["draw_color_summary"]

        self.assertEqual(draw_summary["recommended_color"], "B")
        self.assertGreater(
            draw_summary["target_attack_pressure_black"],
            draw_summary["target_attack_pressure_white"],
        )
        self.assertEqual(draw_summary["dominant_factor"], "target_attack_pressure")

    def test_fixed_controller_case_runner_prefers_black_under_target_attack_pressure(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=2, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=5, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=7, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=1, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=3, color="B", value=9, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="W", value=3, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=10, is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )

        result = run_controller_case(
            FixedControllerCase(
                name="draw_color_black_under_target_attack_pressure",
                game_state=game_state,
                checks=(
                    assert_draw_color_recommendation("B"),
                    assert_draw_color_dominant_factor("target_attack_pressure"),
                ),
            )
        )

        self.assertGreater(
            result["draw_color_summary"]["target_attack_pressure_black"],
            result["draw_color_summary"]["target_attack_pressure_white"],
        )

    def test_controller_draw_color_summary_uses_global_entropy_pressure(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=2, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=8, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=1, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=10, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=11, is_revealed=True),
                        CardSlot(slot_index=3, color="W", value=5, is_revealed=True),
                        CardSlot(slot_index=4, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=5, color="W", value=7, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="W", value="-", is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )

        result = GameController(game_state).run_turn()
        draw_summary = result["draw_color_summary"]

        self.assertGreater(
            draw_summary["entropy_pressure_black"],
            draw_summary["entropy_pressure_white"],
        )

    def test_controller_draw_color_summary_uses_target_entropy_pressure(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=2, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=8, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=1, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=10, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=11, is_revealed=True),
                        CardSlot(slot_index=3, color="W", value=5, is_revealed=True),
                        CardSlot(slot_index=4, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=5, color="W", value=7, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=4, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=6, is_revealed=True),
                        CardSlot(slot_index=3, color="W", value=0, is_revealed=True),
                        CardSlot(slot_index=4, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=5, color="W", value=11, is_revealed=True),
                    ],
                ),
                "reveals": PlayerState(
                    player_id="reveals",
                    slots=[
                        CardSlot(slot_index=0, color="B", value="-", is_revealed=True),
                        CardSlot(slot_index=1, color="W", value="-", is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )

        result = GameController(game_state).run_turn()
        draw_summary = result["draw_color_summary"]

        self.assertGreater(
            draw_summary["target_entropy_pressure_black"],
            draw_summary["target_entropy_pressure_white"],
        )

    def test_controller_draw_color_summary_uses_self_flexibility_pressure(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=10, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=4, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=6, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=2, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=4, is_revealed=True),
                        CardSlot(slot_index=3, color="W", value=7, is_revealed=True),
                        CardSlot(slot_index=4, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=5, color="W", value=9, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="B", value="-", is_revealed=True),
                        CardSlot(slot_index=1, color="W", value="-", is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )

        result = GameController(game_state).run_turn()
        draw_summary = result["draw_color_summary"]

        self.assertEqual(draw_summary["recommended_color"], "B")
        self.assertGreater(
            draw_summary["self_flexibility_pressure_black"],
            draw_summary["self_flexibility_pressure_white"],
        )
        self.assertEqual(draw_summary["dominant_factor"], "self_flexibility_pressure")

    def test_controller_draw_color_summary_uses_hidden_defense_pressure(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=2, is_revealed=True),
                        CardSlot(slot_index=2, color="B", value=4, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=6, is_revealed=True),
                        CardSlot(slot_index=4, color="W", value=8, is_revealed=False),
                        CardSlot(slot_index=5, color="W", value=10, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=3, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=5, is_revealed=True),
                        CardSlot(slot_index=3, color="W", value=7, is_revealed=True),
                        CardSlot(slot_index=4, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=5, color="W", value=9, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="B", value="-", is_revealed=True),
                        CardSlot(slot_index=1, color="W", value="-", is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )

        result = GameController(game_state).run_turn()
        draw_summary = result["draw_color_summary"]

        self.assertEqual(draw_summary["recommended_color"], "B")
        self.assertGreater(
            draw_summary["hidden_defense_pressure_black"],
            draw_summary["hidden_defense_pressure_white"],
        )
        self.assertEqual(draw_summary["dominant_factor"], "hidden_defense_pressure")

    def test_controller_draw_color_summary_uses_target_boundary_pressure(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=10, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=4, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=6, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=3, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=5, is_revealed=True),
                        CardSlot(slot_index=3, color="W", value=0, is_revealed=True),
                        CardSlot(slot_index=4, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=5, color="W", value=8, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="B", value="-", is_revealed=True),
                        CardSlot(slot_index=1, color="W", value="-", is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )

        result = GameController(game_state).run_turn()
        draw_summary = result["draw_color_summary"]

        self.assertEqual(draw_summary["recommended_color"], "B")
        self.assertGreater(
            draw_summary["target_boundary_pressure_black"],
            draw_summary["target_boundary_pressure_white"],
        )
        self.assertEqual(draw_summary["dominant_factor"], "target_boundary_pressure")

    def test_controller_draw_color_summary_uses_target_finish_pressure(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=10, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=4, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=6, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=3, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=5, is_revealed=True),
                        CardSlot(slot_index=3, color="W", value=0, is_revealed=True),
                        CardSlot(slot_index=4, color="W", value=8, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="B", value="-", is_revealed=True),
                        CardSlot(slot_index=1, color="W", value="-", is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )

        result = GameController(game_state).run_turn()
        draw_summary = result["draw_color_summary"]

        self.assertEqual(draw_summary["recommended_color"], "B")
        self.assertGreater(
            draw_summary["target_finish_pressure_black"],
            draw_summary["target_finish_pressure_white"],
        )
        self.assertEqual(draw_summary["dominant_factor"], "target_finish_pressure")

    def test_controller_draw_color_summary_uses_target_focus_pressure(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=4, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=5, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=8, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=6, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=8, is_revealed=True),
                        CardSlot(slot_index=3, color="W", value=9, is_revealed=True),
                        CardSlot(slot_index=4, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=5, color="W", value=11, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=10, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=0, is_revealed=True),
                        CardSlot(slot_index=2, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=2, is_revealed=True),
                    ],
                ),
                "reveals": PlayerState(
                    player_id="reveals",
                    slots=[
                        CardSlot(slot_index=0, color="B", value="-", is_revealed=True),
                        CardSlot(slot_index=1, color="W", value="-", is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )

        result = GameController(game_state).run_turn()
        draw_summary = result["draw_color_summary"]

        self.assertEqual(draw_summary["recommended_color"], "B")
        self.assertGreater(
            draw_summary["target_focus_pressure_black"],
            draw_summary["target_focus_pressure_white"],
        )
        self.assertEqual(draw_summary["dominant_factor"], "target_focus_pressure")

    def test_controller_draw_color_summary_uses_target_recent_momentum_pressure(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=4, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=6, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=9, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=1, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=3, color="B", value=9, is_revealed=True),
                        CardSlot(slot_index=4, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=5, color="W", value=11, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=10, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=2, is_revealed=True),
                    ],
                ),
            },
            actions=[
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=4,
                    guessed_color="W",
                    guessed_value=5,
                    result=False,
                    continued_turn=False,
                ),
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=1,
                    guessed_color="B",
                    guessed_value=3,
                    result=True,
                    continued_turn=True,
                ),
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=2,
                    guessed_color="B",
                    guessed_value=5,
                    result=True,
                    continued_turn=True,
                ),
            ],
        )

        result = GameController(game_state).run_turn()
        draw_summary = result["draw_color_summary"]

        self.assertEqual(draw_summary["recommended_color"], "B")
        self.assertGreater(
            draw_summary["target_recent_momentum_pressure_black"],
            draw_summary["target_recent_momentum_pressure_white"],
        )
        self.assertEqual(
            draw_summary["dominant_factor"],
            "target_recent_momentum_pressure",
        )

    def test_controller_draw_color_summary_uses_recent_self_exposure_pressure(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=2, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=4, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=6, is_revealed=False),
                        CardSlot(slot_index=4, color="W", value=8, is_revealed=False),
                        CardSlot(slot_index=5, color="W", value=10, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=3, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=5, is_revealed=True),
                        CardSlot(slot_index=3, color="W", value=None, is_revealed=False),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=9, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=1, is_revealed=True),
                    ],
                ),
            },
            actions=[
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=1,
                    guessed_color="B",
                    guessed_value=7,
                    result=False,
                    revealed_player_id="me",
                    revealed_slot_index=2,
                    revealed_color="W",
                    revealed_value=4,
                ),
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=1,
                    guessed_color="B",
                    guessed_value=8,
                    result=False,
                    revealed_player_id="me",
                    revealed_slot_index=4,
                    revealed_color="W",
                    revealed_value=8,
                ),
            ],
        )

        result = GameController(game_state).run_turn()
        draw_summary = result["draw_color_summary"]

        self.assertEqual(draw_summary["recommended_color"], "B")
        self.assertGreater(
            draw_summary["recent_self_exposure_pressure_black"],
            draw_summary["recent_self_exposure_pressure_white"],
        )
        self.assertEqual(
            draw_summary["dominant_factor"],
            "recent_self_exposure_pressure",
        )

    def test_controller_draw_color_summary_scales_target_window_by_phase(self):
        aggressive_game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=10, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=4, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=6, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=3, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=5, is_revealed=True),
                        CardSlot(slot_index=3, color="W", value=0, is_revealed=True),
                        CardSlot(slot_index=4, color="W", value=8, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="B", value="-", is_revealed=True),
                        CardSlot(slot_index=1, color="W", value="-", is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )
        conservative_game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=10, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=4, is_revealed=True),
                        CardSlot(slot_index=3, color="W", value=6, is_revealed=True),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=3, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=5, is_revealed=True),
                        CardSlot(slot_index=3, color="W", value=0, is_revealed=True),
                        CardSlot(slot_index=4, color="W", value=8, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="B", value="-", is_revealed=True),
                        CardSlot(slot_index=1, color="W", value="-", is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )

        aggressive_result = GameController(aggressive_game_state).run_turn()
        conservative_result = GameController(conservative_game_state).run_turn()
        aggressive_summary = aggressive_result["draw_color_summary"]
        conservative_summary = conservative_result["draw_color_summary"]

        self.assertGreater(
            aggressive_summary["target_attack_window_factor"],
            conservative_summary["target_attack_window_factor"],
        )
        self.assertGreater(
            aggressive_summary["draw_rollout_phase_gate"],
            conservative_summary["draw_rollout_phase_gate"],
        )
        self.assertGreater(
            aggressive_summary["black_score"] - aggressive_summary["white_score"],
            conservative_summary["black_score"] - conservative_summary["white_score"],
        )

    def test_controller_draw_color_summary_uses_post_draw_expected_value_rollout(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=8, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=2, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=10, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=3, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=5, is_revealed=True),
                        CardSlot(slot_index=3, color="W", value=1, is_revealed=True),
                        CardSlot(slot_index=4, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=5, color="W", value=9, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=11, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=0, is_revealed=True),
                        CardSlot(slot_index=2, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=6, is_revealed=True),
                    ],
                ),
                "reveals": PlayerState(
                    player_id="reveals",
                    slots=[
                        CardSlot(slot_index=0, color="B", value="-", is_revealed=True),
                        CardSlot(slot_index=1, color="W", value="-", is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )

        result = GameController(game_state).run_turn()
        draw_summary = result["draw_color_summary"]

        self.assertGreater(
            abs(
                draw_summary["draw_rollout_expected_best_value_black"]
                - draw_summary["draw_rollout_expected_best_value_white"]
            ),
            0.1,
        )
        self.assertGreater(
            abs(
                draw_summary["draw_rollout_value_pressure_black"]
                - draw_summary["draw_rollout_value_pressure_white"]
            ),
            0.1,
        )
        self.assertGreater(draw_summary["draw_rollout_sample_count_black"], 0.0)
        self.assertGreater(draw_summary["draw_rollout_sample_count_white"], 0.0)
        self.assertGreater(draw_summary["draw_rollout_edge_scale"], 0.0)

    def test_controller_draw_color_summary_uses_post_draw_continuation_rollout(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=4, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=6, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=9, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=1, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=3, color="B", value=9, is_revealed=True),
                        CardSlot(slot_index=4, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=5, color="W", value=11, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=10, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=2, is_revealed=True),
                    ],
                ),
            },
            actions=[
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=4,
                    guessed_color="W",
                    guessed_value=5,
                    result=False,
                    continued_turn=False,
                ),
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=1,
                    guessed_color="B",
                    guessed_value=3,
                    result=True,
                    continued_turn=True,
                ),
                GuessAction(
                    guesser_id="me",
                    target_player_id="opp",
                    target_slot_index=2,
                    guessed_color="B",
                    guessed_value=5,
                    result=True,
                    continued_turn=True,
                ),
            ],
        )

        result = GameController(game_state).run_turn()
        draw_summary = result["draw_color_summary"]

        self.assertGreater(
            draw_summary["draw_rollout_expected_continuation_value_white"],
            draw_summary["draw_rollout_expected_continuation_value_black"],
        )
        self.assertGreater(
            draw_summary["draw_rollout_continuation_value_pressure_white"],
            draw_summary["draw_rollout_continuation_value_pressure_black"],
        )
        self.assertGreaterEqual(
            draw_summary["draw_rollout_expected_continuation_likelihood_white"],
            draw_summary["draw_rollout_expected_continuation_likelihood_black"],
        )

    def test_controller_draw_color_summary_uses_post_draw_attackability_rollout(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=2, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=5, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=7, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=1, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=3, color="B", value=9, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="W", value=3, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=10, is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )

        result = GameController(game_state).run_turn()
        draw_summary = result["draw_color_summary"]

        self.assertGreater(
            draw_summary["draw_rollout_expected_attackability_after_hit_white"],
            draw_summary["draw_rollout_expected_attackability_after_hit_black"],
        )
        self.assertGreater(
            draw_summary["draw_rollout_attackability_pressure_white"],
            draw_summary["draw_rollout_attackability_pressure_black"],
        )
        self.assertGreaterEqual(
            draw_summary["draw_rollout_expected_win_probability_white"],
            draw_summary["draw_rollout_expected_win_probability_black"],
        )

    def test_controller_draw_color_summary_uses_post_draw_target_retention_rollout(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=10, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=4, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=6, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=3, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=5, is_revealed=True),
                        CardSlot(slot_index=3, color="W", value=0, is_revealed=True),
                        CardSlot(slot_index=4, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=5, color="W", value=8, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=11, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=2, is_revealed=True),
                        CardSlot(slot_index=2, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=10, is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )

        result = GameController(game_state).run_turn()
        draw_summary = result["draw_color_summary"]

        self.assertGreater(
            draw_summary["draw_rollout_target_retention_ratio_black"],
            draw_summary["draw_rollout_target_retention_ratio_white"],
        )
        self.assertGreater(
            draw_summary["draw_rollout_target_retention_pressure_black"],
            draw_summary["draw_rollout_target_retention_pressure_white"],
        )
        self.assertGreaterEqual(
            draw_summary["draw_rollout_color_alignment_ratio_black"],
            draw_summary["draw_rollout_color_alignment_ratio_white"],
        )

    def test_controller_draw_color_summary_uses_post_draw_best_gap_rollout(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=10, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=4, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=6, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=3, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=5, is_revealed=True),
                        CardSlot(slot_index=3, color="W", value=0, is_revealed=True),
                        CardSlot(slot_index=4, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=5, color="W", value=8, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="B", value="-", is_revealed=True),
                        CardSlot(slot_index=1, color="W", value="-", is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )

        result = GameController(game_state).run_turn()
        draw_summary = result["draw_color_summary"]

        self.assertGreater(
            draw_summary["draw_rollout_expected_best_gap_white"],
            draw_summary["draw_rollout_expected_best_gap_black"],
        )
        self.assertGreater(
            draw_summary["draw_rollout_best_gap_pressure_white"],
            draw_summary["draw_rollout_best_gap_pressure_black"],
        )
        self.assertGreaterEqual(
            draw_summary["draw_rollout_active_opening_ratio_white"],
            draw_summary["draw_rollout_active_opening_ratio_black"],
        )

    def test_controller_draw_color_summary_uses_post_draw_stability_rollout(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=8, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=2, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=10, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=3, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=5, is_revealed=True),
                        CardSlot(slot_index=3, color="W", value=1, is_revealed=True),
                        CardSlot(slot_index=4, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=5, color="W", value=9, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=11, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=0, is_revealed=True),
                        CardSlot(slot_index=2, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=6, is_revealed=True),
                    ],
                ),
                "reveals": PlayerState(
                    player_id="reveals",
                    slots=[
                        CardSlot(slot_index=0, color="B", value="-", is_revealed=True),
                        CardSlot(slot_index=1, color="W", value="-", is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )

        result = GameController(game_state).run_turn()
        draw_summary = result["draw_color_summary"]

        self.assertGreater(
            draw_summary["draw_rollout_best_value_stddev_black"],
            0.0,
        )
        self.assertGreater(
            draw_summary["draw_rollout_best_value_stddev_black"],
            draw_summary["draw_rollout_best_value_stddev_white"],
        )
        self.assertGreater(
            draw_summary["draw_rollout_best_value_stability_pressure_white"],
            draw_summary["draw_rollout_best_value_stability_pressure_black"],
        )
        self.assertGreaterEqual(
            draw_summary["draw_rollout_win_probability_stability_pressure_white"],
            draw_summary["draw_rollout_win_probability_stability_pressure_black"],
        )

    def test_controller_draw_color_summary_uses_post_draw_floor_rollout(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=8, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=2, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=10, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=3, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=5, is_revealed=True),
                        CardSlot(slot_index=3, color="W", value=1, is_revealed=True),
                        CardSlot(slot_index=4, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=5, color="W", value=9, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=11, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=0, is_revealed=True),
                        CardSlot(slot_index=2, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=6, is_revealed=True),
                    ],
                ),
                "reveals": PlayerState(
                    player_id="reveals",
                    slots=[
                        CardSlot(slot_index=0, color="B", value="-", is_revealed=True),
                        CardSlot(slot_index=1, color="W", value="-", is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )

        result = GameController(game_state).run_turn()
        draw_summary = result["draw_color_summary"]

        self.assertGreater(
            draw_summary["draw_rollout_best_value_floor_white"],
            draw_summary["draw_rollout_best_value_floor_black"],
        )
        self.assertGreater(
            draw_summary["draw_rollout_best_value_floor_pressure_white"],
            draw_summary["draw_rollout_best_value_floor_pressure_black"],
        )
        self.assertGreaterEqual(
            draw_summary["draw_rollout_win_probability_floor_white"],
            draw_summary["draw_rollout_win_probability_floor_black"],
        )

    def test_controller_draw_color_summary_uses_post_draw_information_rollout(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=2, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=5, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=7, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=1, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=3, color="B", value=9, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="W", value=3, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=10, is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )

        result = GameController(game_state).run_turn()
        draw_summary = result["draw_color_summary"]

        self.assertGreater(
            abs(
                draw_summary["draw_rollout_expected_information_gain_black"]
                - draw_summary["draw_rollout_expected_information_gain_white"]
            ),
            0.01,
        )
        self.assertGreater(
            abs(
                draw_summary["draw_rollout_information_gain_floor_black"]
                - draw_summary["draw_rollout_information_gain_floor_white"]
            ),
            0.01,
        )
        self.assertNotEqual(
            draw_summary["draw_rollout_information_gain_pressure_black"],
            draw_summary["draw_rollout_information_gain_pressure_white"],
        )

    def test_controller_exposes_post_draw_opening_plan(self):
        game_state = GameState(
            self_player_id="me",
            target_player_id="opp",
            players={
                "me": PlayerState(
                    player_id="me",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=0, is_revealed=False),
                        CardSlot(slot_index=1, color="B", value=10, is_revealed=False),
                        CardSlot(slot_index=2, color="W", value=4, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=6, is_revealed=False),
                    ],
                ),
                "opp": PlayerState(
                    player_id="opp",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=3, is_revealed=True),
                        CardSlot(slot_index=1, color="B", value=None, is_revealed=False),
                        CardSlot(slot_index=2, color="B", value=5, is_revealed=True),
                        CardSlot(slot_index=3, color="W", value=0, is_revealed=True),
                        CardSlot(slot_index=4, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=5, color="W", value=8, is_revealed=True),
                    ],
                ),
                "side": PlayerState(
                    player_id="side",
                    slots=[
                        CardSlot(slot_index=0, color="B", value=11, is_revealed=True),
                        CardSlot(slot_index=1, color="W", value=2, is_revealed=True),
                        CardSlot(slot_index=2, color="W", value=None, is_revealed=False),
                        CardSlot(slot_index=3, color="W", value=10, is_revealed=True),
                    ],
                ),
            },
            actions=[],
        )

        result = GameController(game_state).run_turn()
        draw_summary = result["draw_color_summary"]
        opening_plan = result["draw_opening_plan"]

        self.assertEqual(opening_plan["recommended_color"], "B")
        self.assertEqual(opening_plan["target_player_id"], "opp")
        self.assertEqual(opening_plan["target_slot_index"], 1)
        self.assertEqual(opening_plan["guess_card"], ["B", 4])
        self.assertGreater(
            draw_summary["draw_rollout_opening_support_black"],
            draw_summary["draw_rollout_opening_support_white"],
        )
        self.assertGreater(
            draw_summary["draw_rollout_opening_expected_value_black"],
            draw_summary["draw_rollout_opening_expected_value_white"],
        )

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
