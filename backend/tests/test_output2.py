from fastapi.encoders import jsonable_encoder

from app.core.engine import GameController
from app.core.state import CardSlot, GameState, PlayerState


def test_game_controller_returns_serializable_best_move():
    state = GameState(
        self_player_id="p1",
        target_player_id="p2",
        players={
            "p1": PlayerState(
                player_id="p1",
                slots=[
                    CardSlot(slot_index=0, color="B", value=1, is_revealed=True),
                    CardSlot(slot_index=1, color="W", value=2, is_revealed=True),
                ],
            ),
            "p2": PlayerState(
                player_id="p2",
                slots=[
                    CardSlot(slot_index=0, color="B", value=None, is_revealed=False),
                    CardSlot(slot_index=1, color="W", value=4, is_revealed=True),
                ],
            ),
        },
        actions=[],
    )

    result = GameController(state).run_turn()

    assert "best_move" in result
    assert "top_moves" in result
    if result["best_move"] is not None:
        assert "_success_matrix" not in result["best_move"]
    jsonable_encoder(result)


def test_draw_rollout_sampling_is_capped():
    state = GameState(
        self_player_id="me",
        target_player_id="opp",
        players={
            "me": PlayerState(
                player_id="me",
                slots=[
                    CardSlot(slot_index=0, color="B", value=0, is_revealed=True),
                ],
            ),
            "opp": PlayerState(
                player_id="opp",
                slots=[
                    CardSlot(slot_index=0, color="W", value=1, is_revealed=True),
                    CardSlot(slot_index=1, color=None, value=None, is_revealed=False),
                    CardSlot(slot_index=2, color=None, value=None, is_revealed=False),
                ],
            ),
        },
        actions=[],
    )

    controller = GameController(state)
    black_samples = controller._representative_draw_cards("B")
    white_samples = controller._representative_draw_cards("W")

    assert len(black_samples) <= controller.DRAW_ROLLOUT_MAX_SAMPLES_PER_COLOR
    assert len(white_samples) <= controller.DRAW_ROLLOUT_MAX_SAMPLES_PER_COLOR
