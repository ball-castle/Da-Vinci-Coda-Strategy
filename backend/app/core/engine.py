import itertools
from collections import defaultdict
from math import log2
from typing import Any, DefaultDict, Dict, Iterable, List, Sequence, Tuple, Union

CardValue = Union[int, str]
Card = Tuple[str, CardValue]
ProbabilityMatrix = Dict[int, Dict[Card, float]]

CARD_COLORS = ("B", "W")
CARD_COLOR_RANK = {"B": 0, "W": 1}
JOKER = "-"
MAX_CARD_VALUE = 11


def card_sort_key(card: Card) -> Tuple[int, int]:
    color, value = card
    if value == JOKER:
        return (MAX_CARD_VALUE + 1, CARD_COLOR_RANK[color])
    return (int(value), CARD_COLOR_RANK[color])


def serialize_card(card: Card) -> List[CardValue]:
    return [card[0], card[1]]


class DaVinciInferenceEngine:
    """推理引擎：枚举合法隐藏手牌，并输出按位置的后验概率分布。"""

    def __init__(self, my_cards: Sequence[Card], public_cards: Dict[str, Sequence[Card]]):
        self.my_cards = tuple(my_cards)
        self.public_cards = {player: tuple(cards) for player, cards in public_cards.items()}
        self.available_cards = tuple(self._get_available_cards())

    def _build_all_possible_cards(self) -> Tuple[Card, ...]:
        ordered_cards = [
            (color, value)
            for value in range(MAX_CARD_VALUE + 1)
            for color in CARD_COLORS
        ]
        ordered_cards.extend((color, JOKER) for color in CARD_COLORS)
        return tuple(ordered_cards)

    def _get_available_cards(self) -> Iterable[Card]:
        known_cards = set(self.my_cards)
        for cards in self.public_cards.values():
            known_cards.update(cards)

        for card in self._build_all_possible_cards():
            if card not in known_cards:
                yield card

    def infer_hidden_probabilities(
        self,
        hidden_card_count: int,
        guess_signals: Sequence[Dict[str, Any]],
        psy_filter: "PsychologicalFilter",
    ) -> Tuple[ProbabilityMatrix, int, float]:
        if hidden_card_count <= 0 or hidden_card_count > len(self.available_cards):
            return {}, 0, 0.0

        position_weights: DefaultDict[int, DefaultDict[Card, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        total_weight = 0.0
        search_space_size = 0

        for hypothesis in itertools.combinations(self.available_cards, hidden_card_count):
            search_space_size += 1
            weight = psy_filter.score_hypothesis(hypothesis, guess_signals)
            if weight <= 0:
                continue

            total_weight += weight
            for idx, card in enumerate(hypothesis):
                position_weights[idx][card] += weight

        if total_weight == 0:
            return {}, search_space_size, total_weight

        probability_matrix: ProbabilityMatrix = {}
        for idx, weighted_cards in position_weights.items():
            probability_matrix[idx] = {
                card: weight / total_weight for card, weight in weighted_cards.items()
            }

        return probability_matrix, search_space_size, total_weight


class PsychologicalFilter:
    """心理战过滤：用历史猜测对候选手牌做贝叶斯式加权。"""

    EXACT_GUESS_PENALTY = 0.10
    EXACT_GUESS_PENALTY_ON_HIT = 0.05
    ADJACENT_BONUS = 1.10
    ADJACENT_BONUS_ON_HIT = 1.20

    def build_guess_signals(self, opponent_history: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        signals: List[Dict[str, Any]] = []

        for action in opponent_history:
            if action.get("type") != "guess":
                continue

            color = action.get("target_color")
            number = action.get("target_num")
            if color not in CARD_COLORS or not isinstance(number, int):
                continue

            adjacent_cards: List[Card] = []
            if number > 0:
                adjacent_cards.append((color, number - 1))
            if number < MAX_CARD_VALUE:
                adjacent_cards.append((color, number + 1))

            signals.append(
                {
                    "exact_card": (color, number),
                    "adjacent_cards": tuple(adjacent_cards),
                    "result": bool(action.get("result")),
                }
            )

        return signals

    def score_hypothesis(
        self,
        hypothesis: Sequence[Card],
        guess_signals: Sequence[Dict[str, Any]],
    ) -> float:
        if not guess_signals:
            return 1.0

        numeric_cards = {card for card in hypothesis if card[1] != JOKER}
        weight = 1.0

        for signal in guess_signals:
            if signal["exact_card"] in numeric_cards:
                weight *= (
                    self.EXACT_GUESS_PENALTY_ON_HIT
                    if signal["result"]
                    else self.EXACT_GUESS_PENALTY
                )
                continue

            if any(adjacent in numeric_cards for adjacent in signal["adjacent_cards"]):
                weight *= (
                    self.ADJACENT_BONUS_ON_HIT
                    if signal["result"]
                    else self.ADJACENT_BONUS
                )

        return weight


class DaVinciDecisionEngine:
    """决策引擎：综合命中收益、失败风险和信息增益。"""

    BASE_REWARD = 10.0
    RISK_BASE = 20.0
    INFORMATION_GAIN_WEIGHT = 1.5

    def calculate_risk_factor(self, my_hidden_count: int) -> float:
        return self.RISK_BASE / max(1, my_hidden_count)

    def calculate_ev(self, probability: float, risk_factor: float) -> Tuple[float, float]:
        information_gain = self._binary_entropy(probability)
        reward = probability * self.BASE_REWARD
        penalty = (1.0 - probability) * risk_factor
        expected_value = reward - penalty + information_gain * self.INFORMATION_GAIN_WEIGHT
        return expected_value, information_gain

    def evaluate_moves(
        self,
        probability_matrix: ProbabilityMatrix,
        my_hidden_count: int,
    ) -> Tuple[List[Dict[str, Any]], float]:
        risk_factor = self.calculate_risk_factor(my_hidden_count)
        moves: List[Dict[str, Any]] = []

        for target_idx, card_probs in probability_matrix.items():
            for card, probability in card_probs.items():
                expected_value, information_gain = self.calculate_ev(probability, risk_factor)
                moves.append(
                    {
                        "target_index": target_idx,
                        "guess_card": serialize_card(card),
                        "win_probability": probability,
                        "expected_value": expected_value,
                        "information_gain": information_gain,
                        "target_scope": "hidden_cards",
                    }
                )

        moves.sort(
            key=lambda move: (
                -move["expected_value"],
                -move["win_probability"],
                -move["information_gain"],
                card_sort_key((move["guess_card"][0], move["guess_card"][1])),
            )
        )
        return moves, risk_factor

    def _binary_entropy(self, probability: float) -> float:
        if probability <= 0.0 or probability >= 1.0:
            return 0.0
        return -(probability * log2(probability) + (1.0 - probability) * log2(1.0 - probability))


class GameController:
    """主循环：把概率推理、心理过滤和行动评估串起来。"""

    def __init__(self, my_cards: Sequence[Card], public_cards: Dict[str, Sequence[Card]]):
        self.my_cards = list(my_cards)
        self.public_cards = {player: list(cards) for player, cards in public_cards.items()}
        self.inference_engine = DaVinciInferenceEngine(my_cards, public_cards)
        self.psy_filter = PsychologicalFilter()
        self.decision_engine = DaVinciDecisionEngine()
        self.opponent_history: List[Dict[str, Any]] = []

    def update_opponent_action(self, action_dict: Dict[str, Any]) -> None:
        self.opponent_history.append(action_dict)

    def run_turn(self, opponent_card_count: int) -> Dict[str, Any]:
        opponent_revealed_count = len(self.public_cards.get("opponent", []))
        opponent_hidden_count = max(0, opponent_card_count - opponent_revealed_count)

        if opponent_hidden_count == 0:
            return {
                "best_move": None,
                "top_moves": [],
                "probability_matrix": [],
                "search_space_size": 0,
                "opponent_hidden_count": 0,
                "risk_factor": self.decision_engine.calculate_risk_factor(self._my_hidden_count()),
                "should_stop": True,
            }

        guess_signals = self.psy_filter.build_guess_signals(self.opponent_history)
        probability_matrix, search_space_size, total_weight = self.inference_engine.infer_hidden_probabilities(
            opponent_hidden_count,
            guess_signals,
            self.psy_filter,
        )

        my_hidden_count = self._my_hidden_count()
        evaluated_moves, risk_factor = self.decision_engine.evaluate_moves(probability_matrix, my_hidden_count)
        best_move = evaluated_moves[0] if evaluated_moves and evaluated_moves[0]["expected_value"] > 0 else None

        return {
            "best_move": best_move,
            "top_moves": evaluated_moves[:5],
            "probability_matrix": self._serialize_probability_matrix(probability_matrix),
            "search_space_size": search_space_size,
            "opponent_hidden_count": opponent_hidden_count,
            "risk_factor": risk_factor,
            "effective_weight_sum": total_weight,
            "should_stop": best_move is None,
        }

    def _my_hidden_count(self) -> int:
        my_public_cards = set(self.public_cards.get("me", []))
        return sum(1 for card in self.my_cards if card not in my_public_cards)

    def _serialize_probability_matrix(self, probability_matrix: ProbabilityMatrix) -> List[Dict[str, Any]]:
        serialized_positions: List[Dict[str, Any]] = []

        for target_idx in sorted(probability_matrix):
            sorted_candidates = sorted(
                probability_matrix[target_idx].items(),
                key=lambda item: (-item[1], card_sort_key(item[0])),
            )
            serialized_positions.append(
                {
                    "target_index": target_idx,
                    "target_scope": "hidden_cards",
                    "candidates": [
                        {
                            "card": serialize_card(card),
                            "probability": probability,
                        }
                        for card, probability in sorted_candidates
                    ],
                }
            )

        return serialized_positions
