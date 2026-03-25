import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.engine import DaVinciDecisionEngine


@dataclass(frozen=True)
class DecisionRegressionCase:
    name: str
    my_hidden_count: int
    moves: List[Dict[str, float]]
    expect_continue: bool
    stop_reason_contains: Optional[str] = None
    positive_breakdown_key: Optional[str] = None
    min_continue_margin: Optional[float] = None
    max_continue_margin: Optional[float] = None


def build_move(**overrides: float) -> Dict[str, float]:
    move: Dict[str, float] = {
        "expected_value": 0.0,
        "win_probability": 0.0,
        "continuation_value": 0.0,
        "continuation_likelihood": 0.0,
        "attackability_after_hit": 0.0,
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


if __name__ == "__main__":
    unittest.main()
