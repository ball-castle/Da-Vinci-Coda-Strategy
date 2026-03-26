from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence

from app.core.engine import GameController
from app.core.state import GameState


@dataclass(frozen=True)
class FixedControllerCase:
    name: str
    game_state: GameState
    checks: Sequence[Callable[[Dict[str, Any]], None]] = field(default_factory=tuple)


def run_controller_case(case: FixedControllerCase) -> Dict[str, Any]:
    result = GameController(case.game_state).run_turn()
    for check in case.checks:
        check(result)
    return result


def assert_serialized_probability_mass(result: Dict[str, Any]) -> None:
    for matrix_key in (
        "probability_matrix",
        "full_probability_matrix",
        "hard_probability_matrix",
        "hard_full_probability_matrix",
        "soft_full_probability_matrix",
    ):
        matrix = result.get(matrix_key, [])
        if matrix_key.endswith("full_probability_matrix"):
            positions: List[Dict[str, Any]] = [
                position
                for player in matrix
                for position in player.get("positions", [])
            ]
        else:
            positions = list(matrix)
        for position in positions:
            candidates = position.get("candidates", [])
            if not candidates:
                continue
            total = sum(float(candidate.get("probability", 0.0)) for candidate in candidates)
            if abs(total - 1.0) > 1e-6:
                raise AssertionError(
                    f"Probability mass drifted in {matrix_key} at slot {position.get('target_slot_index')}: {total}"
                )


def all_serialized_candidate_cards(result: Dict[str, Any]) -> List[List[Any]]:
    cards: List[List[Any]] = []
    for player in result.get("full_probability_matrix", []):
        for position in player.get("positions", []):
            for candidate in position.get("candidates", []):
                cards.append(list(candidate.get("card", [])))
    return cards


def serialized_position_candidates(
    result: Dict[str, Any],
    *,
    player_id: str,
    slot_index: int,
) -> List[List[Any]]:
    for player in result.get("full_probability_matrix", []):
        if player.get("player_id") != player_id:
            continue
        for position in player.get("positions", []):
            if int(position.get("target_slot_index", -1)) == slot_index:
                return [
                    list(candidate.get("card", []))
                    for candidate in position.get("candidates", [])
                ]
    raise AssertionError(
        f"Could not find serialized candidates for player {player_id!r} slot {slot_index}."
    )


def assert_slot_candidate_set(
    *,
    player_id: str,
    slot_index: int,
    expected_cards: Sequence[Sequence[Any]],
) -> Callable[[Dict[str, Any]], None]:
    def _normalize(cards: Sequence[Sequence[Any]]) -> List[List[Any]]:
        return sorted(
            [list(card) for card in cards],
            key=lambda card: (str(card[0]), str(card[1])),
        )

    expected = _normalize(expected_cards)

    def _assert(result: Dict[str, Any]) -> None:
        actual = _normalize(
            serialized_position_candidates(
                result,
                player_id=player_id,
                slot_index=slot_index,
            )
        )
        if actual != expected:
            raise AssertionError(
                f"Expected candidate set {expected!r} for player {player_id!r} slot {slot_index}, got {actual!r}."
            )

    return _assert
