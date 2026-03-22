from collections import defaultdict
from math import log2
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

from app.core.state import (
    CARD_COLORS,
    JOKER,
    MAX_CARD_VALUE,
    Card,
    CardSlot,
    GameState,
)

ProbabilityMatrix = Dict[int, Dict[Card, float]]
CARD_COLOR_RANK = {"B": 0, "W": 1}


def card_sort_key(card: Card) -> Tuple[int, int]:
    color, value = card
    if value == JOKER:
        return (MAX_CARD_VALUE + 1, CARD_COLOR_RANK[color])
    return (int(value), CARD_COLOR_RANK[color])


def serialize_card(card: Card) -> List[Any]:
    return [card[0], card[1]]


class DaVinciInferenceEngine:
    """Infer posterior card probabilities for the target player's hidden slots."""

    def __init__(self, game_state: GameState):
        self.game_state = game_state
        self.target_slots = game_state.target_player().ordered_slots()
        self.hidden_slots = [slot for slot in self.target_slots if slot.value is None]
        self.available_cards = tuple(self._get_available_cards())

    def _build_all_possible_cards(self) -> Tuple[Card, ...]:
        ordered_cards = [
            (color, value)
            for value in range(MAX_CARD_VALUE + 1)
            for color in CARD_COLORS
        ]
        ordered_cards.extend((color, JOKER) for color in CARD_COLORS)
        return tuple(ordered_cards)

    def _get_available_cards(self) -> Tuple[Card, ...]:
        known_cards = set(self.game_state.known_cards())
        return tuple(
            card for card in self._build_all_possible_cards() if card not in known_cards
        )

    def infer_hidden_probabilities(
        self,
        guess_signals: Sequence[Dict[str, Any]],
        psy_filter: "PsychologicalFilter",
    ) -> Tuple[ProbabilityMatrix, int, float]:
        if not self.hidden_slots:
            return {}, 0, 0.0

        position_weights: DefaultDict[int, DefaultDict[Card, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        total_weight = 0.0
        search_space_size = 0
        hidden_assignment: Dict[int, Card] = {}

        def dfs(
            slot_cursor: int,
            available_cards: Tuple[Card, ...],
            last_numeric_card: Optional[Card],
        ) -> None:
            nonlocal search_space_size, total_weight

            if slot_cursor == len(self.target_slots):
                search_space_size += 1
                hypothesis = tuple(
                    hidden_assignment[slot.slot_index] for slot in self.hidden_slots
                )
                weight = psy_filter.score_hypothesis(hypothesis, guess_signals)
                if weight <= 0:
                    return

                total_weight += weight
                for slot in self.hidden_slots:
                    position_weights[slot.slot_index][hidden_assignment[slot.slot_index]] += weight
                return

            slot = self.target_slots[slot_cursor]
            fixed_card = slot.known_card()
            if fixed_card is not None:
                next_last_numeric = self._advance_sequence(last_numeric_card, fixed_card)
                if next_last_numeric is None:
                    return
                dfs(slot_cursor + 1, available_cards, next_last_numeric)
                return

            for idx, candidate in enumerate(available_cards):
                if slot.color is not None and candidate[0] != slot.color:
                    continue

                next_last_numeric = self._advance_sequence(last_numeric_card, candidate)
                if next_last_numeric is None:
                    continue

                hidden_assignment[slot.slot_index] = candidate
                dfs(
                    slot_cursor + 1,
                    available_cards[:idx] + available_cards[idx + 1 :],
                    next_last_numeric,
                )
                del hidden_assignment[slot.slot_index]

        dfs(0, self.available_cards, None)

        if total_weight == 0:
            return {}, search_space_size, total_weight

        probability_matrix: ProbabilityMatrix = {}
        for slot_index, weighted_cards in position_weights.items():
            probability_matrix[slot_index] = {
                card: weight / total_weight for card, weight in weighted_cards.items()
            }

        return probability_matrix, search_space_size, total_weight

    def _advance_sequence(
        self,
        last_numeric_card: Optional[Card],
        candidate: Card,
    ) -> Optional[Card]:
        if candidate[1] == JOKER:
            return last_numeric_card

        if last_numeric_card is None:
            return candidate

        if candidate[1] < last_numeric_card[1]:
            return None

        if candidate[1] == last_numeric_card[1]:
            if not (
                last_numeric_card[0] == "B"
                and candidate[0] == "W"
            ):
                return None

        return candidate


class PsychologicalFilter:
    """Apply lightweight Bayesian-style weighting based on observed actions."""

    EXACT_GUESS_PENALTY = 0.10
    EXACT_GUESS_PENALTY_ON_HIT = 0.05
    ADJACENT_BONUS = 1.10
    ADJACENT_BONUS_ON_HIT = 1.20
    CONTINUE_GUESS_BONUS = 1.15

    def build_guess_signals(
        self,
        game_state: GameState,
    ) -> List[Dict[str, Any]]:
        signals: List[Dict[str, Any]] = []

        for action in game_state.actions:
            if action.action_type != "guess":
                continue
            if action.guesser_id != game_state.target_player_id:
                continue

            guessed_card = action.guessed_card()
            if guessed_card is None or not isinstance(guessed_card[1], int):
                continue

            guessed_color, guessed_number = guessed_card
            adjacent_cards: List[Card] = []
            if guessed_number > 0:
                adjacent_cards.append((guessed_color, guessed_number - 1))
            if guessed_number < MAX_CARD_VALUE:
                adjacent_cards.append((guessed_color, guessed_number + 1))

            signals.append(
                {
                    "exact_card": guessed_card,
                    "adjacent_cards": tuple(adjacent_cards),
                    "result": action.result,
                    "continued_turn": action.continued_turn,
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

            if signal["continued_turn"]:
                weight *= self.CONTINUE_GUESS_BONUS

        return weight


class DaVinciDecisionEngine:
    """Score candidate actions using hit rate, risk, and information gain."""

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
        hidden_index_by_slot: Dict[int, int],
    ) -> Tuple[List[Dict[str, Any]], float]:
        risk_factor = self.calculate_risk_factor(my_hidden_count)
        moves: List[Dict[str, Any]] = []

        for target_slot_index, card_probs in probability_matrix.items():
            for card, probability in card_probs.items():
                expected_value, information_gain = self.calculate_ev(probability, risk_factor)
                moves.append(
                    {
                        "target_index": hidden_index_by_slot.get(target_slot_index, target_slot_index),
                        "target_slot_index": target_slot_index,
                        "guess_card": serialize_card(card),
                        "win_probability": probability,
                        "expected_value": expected_value,
                        "information_gain": information_gain,
                        "target_scope": "player_slots",
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
    """Coordinate inference, action weighting, and candidate move scoring."""

    def __init__(self, game_state: GameState):
        self.game_state = game_state
        self.inference_engine = DaVinciInferenceEngine(game_state)
        self.psy_filter = PsychologicalFilter()
        self.decision_engine = DaVinciDecisionEngine()

    def run_turn(self) -> Dict[str, Any]:
        hidden_slots = self.game_state.target_hidden_slots()
        if not hidden_slots:
            return {
                "best_move": None,
                "top_moves": [],
                "probability_matrix": [],
                "search_space_size": 0,
                "opponent_hidden_count": 0,
                "risk_factor": self.decision_engine.calculate_risk_factor(self.game_state.my_hidden_count()),
                "should_stop": True,
            }

        guess_signals = self.psy_filter.build_guess_signals(self.game_state)
        probability_matrix, search_space_size, total_weight = self.inference_engine.infer_hidden_probabilities(
            guess_signals,
            self.psy_filter,
        )

        hidden_index_by_slot = self.game_state.hidden_index_by_slot(self.game_state.target_player_id)
        evaluated_moves, risk_factor = self.decision_engine.evaluate_moves(
            probability_matrix,
            self.game_state.my_hidden_count(),
            hidden_index_by_slot,
        )
        best_move = evaluated_moves[0] if evaluated_moves and evaluated_moves[0]["expected_value"] > 0 else None

        return {
            "best_move": best_move,
            "top_moves": evaluated_moves[:5],
            "probability_matrix": self._serialize_probability_matrix(
                probability_matrix,
                hidden_index_by_slot,
            ),
            "search_space_size": search_space_size,
            "opponent_hidden_count": len(hidden_slots),
            "risk_factor": risk_factor,
            "effective_weight_sum": total_weight,
            "should_stop": best_move is None,
        }

    def _serialize_probability_matrix(
        self,
        probability_matrix: ProbabilityMatrix,
        hidden_index_by_slot: Dict[int, int],
    ) -> List[Dict[str, Any]]:
        serialized_positions: List[Dict[str, Any]] = []

        for target_slot_index in sorted(probability_matrix):
            sorted_candidates = sorted(
                probability_matrix[target_slot_index].items(),
                key=lambda item: (-item[1], card_sort_key(item[0])),
            )
            serialized_positions.append(
                {
                    "target_index": hidden_index_by_slot.get(target_slot_index, target_slot_index),
                    "target_slot_index": target_slot_index,
                    "target_scope": "player_slots",
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
