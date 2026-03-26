from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from app.core.engine import DaVinciDecisionEngine


@dataclass(frozen=True)
class FixedDecisionCase:
    name: str
    all_moves: Sequence[Dict[str, Any]]
    my_hidden_count: int
    risk_factor: Optional[float] = None
    checks: Sequence[
        Callable[[Optional[Dict[str, Any]], Dict[str, Any]], None]
    ] = field(default_factory=tuple)


def run_decision_case(
    case: FixedDecisionCase,
    *,
    engine: Optional[DaVinciDecisionEngine] = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    engine = engine or DaVinciDecisionEngine()
    risk_factor = (
        case.risk_factor
        if case.risk_factor is not None
        else engine.calculate_risk_factor(case.my_hidden_count)
    )
    best_move, summary = engine.choose_best_move(
        list(case.all_moves),
        risk_factor=risk_factor,
        my_hidden_count=case.my_hidden_count,
    )
    for check in case.checks:
        check(best_move, summary)
    return best_move, summary


def assert_continue_summary(best_move: Optional[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    if best_move is None:
        raise AssertionError("Expected a continuing decision, but best_move is None.")
    if summary.get("recommend_stop", True):
        raise AssertionError("Expected recommend_stop to be False for continue case.")
    if float(summary.get("continue_margin", 0.0)) <= 0.0:
        raise AssertionError("Expected continue_margin to be positive for continue case.")


def assert_stop_summary(best_move: Optional[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    if best_move is not None:
        raise AssertionError("Expected a stop decision, but best_move is present.")
    if not summary.get("recommend_stop", False):
        raise AssertionError("Expected recommend_stop to be True for stop case.")
    if float(summary.get("continue_margin", 0.0)) >= 0.0:
        raise AssertionError("Expected continue_margin to be negative for stop case.")


def assert_positive_breakdown(component: str) -> Callable[[Optional[Dict[str, Any]], Dict[str, Any]], None]:
    def _assert(best_move: Optional[Dict[str, Any]], summary: Dict[str, Any]) -> None:
        breakdown = summary.get("decision_score_breakdown", {})
        if float(breakdown.get(component, 0.0)) <= 0.0:
            raise AssertionError(
                f"Expected decision_score_breakdown[{component!r}] to be positive."
            )

    return _assert


def assert_negative_breakdown(component: str) -> Callable[[Optional[Dict[str, Any]], Dict[str, Any]], None]:
    def _assert(best_move: Optional[Dict[str, Any]], summary: Dict[str, Any]) -> None:
        breakdown = summary.get("decision_score_breakdown", {})
        if float(breakdown.get(component, 0.0)) >= 0.0:
            raise AssertionError(
                f"Expected decision_score_breakdown[{component!r}] to be negative."
            )

    return _assert
