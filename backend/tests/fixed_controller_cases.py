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
