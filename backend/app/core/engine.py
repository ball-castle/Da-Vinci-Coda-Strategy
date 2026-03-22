from collections import defaultdict
from math import log2
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

from app.core.state import (
    CARD_COLORS,
    JOKER,
    MAX_CARD_VALUE,
    Card,
    GameState,
)

ProbabilityMatrix = Dict[int, Dict[Card, float]]
FullProbabilityMatrix = Dict[str, ProbabilityMatrix]
CARD_COLOR_RANK = {"B": 0, "W": 1}


def card_sort_key(card: Card) -> Tuple[int, int]:
    color, value = card
    if value == JOKER:
        return (MAX_CARD_VALUE + 1, CARD_COLOR_RANK[color])
    return (int(value), CARD_COLOR_RANK[color])


def serialize_card(card: Card) -> List[Any]:
    return [card[0], card[1]]


class DaVinciInferenceEngine:
    """Infer posterior card probabilities for all non-self hidden slots jointly."""

    def __init__(self, game_state: GameState):
        self.game_state = game_state
        self.inference_player_ids = [
            player_id
            for player_id in game_state.inference_player_ids()
            if game_state.resolved_ordered_slots(player_id)
        ]
        self.player_slots = {
            player_id: game_state.resolved_ordered_slots(player_id)
            for player_id in self.inference_player_ids
        }
        self.hidden_slots_by_player = {
            player_id: [slot for slot in slots if slot.value is None]
            for player_id, slots in self.player_slots.items()
        }
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
        guess_signals_by_player: Dict[str, Sequence[Dict[str, Any]]],
        psy_filter: "PsychologicalFilter",
    ) -> Tuple[FullProbabilityMatrix, int, float]:
        if not any(self.hidden_slots_by_player.values()):
            return {}, 0, 0.0

        position_weights: Dict[str, DefaultDict[int, DefaultDict[Card, float]]] = {
            player_id: defaultdict(lambda: defaultdict(float))
            for player_id in self.inference_player_ids
        }
        total_weight = 0.0
        search_space_size = 0
        hidden_assignment: Dict[str, Dict[int, Card]] = {
            player_id: {}
            for player_id in self.inference_player_ids
        }

        def dfs_player(player_cursor: int, available_cards: Tuple[Card, ...]) -> None:
            nonlocal search_space_size, total_weight

            if player_cursor == len(self.inference_player_ids):
                search_space_size += 1
                hypothesis_by_player = {
                    player_id: dict(hidden_assignment[player_id])
                    for player_id in self.inference_player_ids
                    if self.hidden_slots_by_player[player_id]
                }
                weight = psy_filter.score_global_hypothesis(
                    hypothesis_by_player,
                    guess_signals_by_player,
                    self.game_state,
                )
                if weight <= 0:
                    return

                total_weight += weight
                for player_id, slots in self.hidden_slots_by_player.items():
                    for slot in slots:
                        position_weights[player_id][slot.slot_index][
                            hidden_assignment[player_id][slot.slot_index]
                        ] += weight
                return

            player_id = self.inference_player_ids[player_cursor]
            player_slots = self.player_slots[player_id]

            def dfs_slot(
                slot_cursor: int,
                current_available_cards: Tuple[Card, ...],
                last_numeric_card: Optional[Card],
            ) -> None:
                if slot_cursor == len(player_slots):
                    dfs_player(player_cursor + 1, current_available_cards)
                    return

                slot = player_slots[slot_cursor]
                fixed_card = slot.known_card()
                if fixed_card is not None:
                    next_last_numeric = self._advance_sequence(last_numeric_card, fixed_card)
                    if next_last_numeric is None:
                        return
                    dfs_slot(slot_cursor + 1, current_available_cards, next_last_numeric)
                    return

                for idx, candidate in enumerate(current_available_cards):
                    if slot.color is not None and candidate[0] != slot.color:
                        continue

                    next_last_numeric = self._advance_sequence(last_numeric_card, candidate)
                    if next_last_numeric is None:
                        continue

                    hidden_assignment[player_id][slot.slot_index] = candidate
                    dfs_slot(
                        slot_cursor + 1,
                        current_available_cards[:idx] + current_available_cards[idx + 1 :],
                        next_last_numeric,
                    )
                    del hidden_assignment[player_id][slot.slot_index]

            dfs_slot(0, available_cards, None)

        dfs_player(0, self.available_cards)

        if total_weight == 0:
            return {}, search_space_size, total_weight

        full_probability_matrix: FullProbabilityMatrix = {}
        for player_id, weighted_slots in position_weights.items():
            if not weighted_slots:
                continue
            full_probability_matrix[player_id] = {
                slot_index: {
                    card: weight / total_weight
                    for card, weight in weighted_cards.items()
                }
                for slot_index, weighted_cards in weighted_slots.items()
            }

        return full_probability_matrix, search_space_size, total_weight

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
    ) -> Dict[str, List[Dict[str, Any]]]:
        signals_by_player: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for action in game_state.actions:
            if action.action_type != "guess":
                continue

            guessed_card = action.guessed_card()
            if guessed_card is None:
                continue

            adjacent_cards: List[Card] = []
            guessed_value = guessed_card[1]
            if isinstance(guessed_value, int):
                guessed_color = guessed_card[0]
                if guessed_value > 0:
                    adjacent_cards.append((guessed_color, guessed_value - 1))
                if guessed_value < MAX_CARD_VALUE:
                    adjacent_cards.append((guessed_color, guessed_value + 1))

            signals_by_player[action.guesser_id].append(
                {
                    "exact_card": guessed_card,
                    "adjacent_cards": tuple(adjacent_cards),
                    "result": action.result,
                    "continued_turn": action.continued_turn,
                    "target_player_id": action.target_player_id,
                    "target_slot_index": action.target_slot_index,
                }
            )

        return signals_by_player

    def score_global_hypothesis(
        self,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        guess_signals_by_player: Dict[str, Sequence[Dict[str, Any]]],
        game_state: GameState,
    ) -> float:
        weight = 1.0

        for player_id, hypothesis in hypothesis_by_player.items():
            weight *= self._score_player_hypothesis(
                tuple(hypothesis.values()),
                guess_signals_by_player.get(player_id, ()),
            )
            if weight <= 0:
                return 0.0

        for signals in guess_signals_by_player.values():
            for signal in signals:
                target_player_id = signal["target_player_id"]
                target_slot_index = signal["target_slot_index"]
                if target_player_id is None or target_slot_index is None:
                    continue

                slot_card = self._resolve_slot_card(
                    game_state,
                    hypothesis_by_player,
                    target_player_id,
                    target_slot_index,
                )
                if slot_card is None:
                    continue

                if signal["result"]:
                    if slot_card != signal["exact_card"]:
                        return 0.0
                else:
                    if slot_card == signal["exact_card"]:
                        return 0.0

        return weight

    def _score_player_hypothesis(
        self,
        hypothesis: Sequence[Card],
        guess_signals: Sequence[Dict[str, Any]],
    ) -> float:
        if not guess_signals:
            return 1.0

        all_cards = set(hypothesis)
        numeric_cards = {card for card in hypothesis if card[1] != JOKER}
        weight = 1.0

        for signal in guess_signals:
            if signal["exact_card"] in all_cards:
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

    def _resolve_slot_card(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        player_id: str,
        slot_index: int,
    ) -> Optional[Card]:
        try:
            slot = game_state.get_slot(player_id, slot_index)
        except ValueError:
            return None

        known_card = slot.known_card()
        if known_card is not None:
            return known_card

        return hypothesis_by_player.get(player_id, {}).get(slot_index)


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
        player_id: str,
    ) -> Tuple[List[Dict[str, Any]], float]:
        risk_factor = self.calculate_risk_factor(my_hidden_count)
        moves: List[Dict[str, Any]] = []

        for target_slot_index, card_probs in probability_matrix.items():
            for card, probability in card_probs.items():
                expected_value, information_gain = self.calculate_ev(probability, risk_factor)
                moves.append(
                    {
                        "target_player_id": player_id,
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
        target_hidden_slots = self.game_state.target_hidden_slots()
        has_any_hidden_slots = any(self.inference_engine.hidden_slots_by_player.values())
        if not has_any_hidden_slots:
            return {
                "best_move": None,
                "top_moves": [],
                "probability_matrix": [],
                "full_probability_matrix": [],
                "search_space_size": 0,
                "opponent_hidden_count": 0,
                "risk_factor": self.decision_engine.calculate_risk_factor(self.game_state.my_hidden_count()),
                "should_stop": True,
            }

        guess_signals_by_player = self.psy_filter.build_guess_signals(self.game_state)
        full_probability_matrix, search_space_size, total_weight = self.inference_engine.infer_hidden_probabilities(
            guess_signals_by_player,
            self.psy_filter,
        )

        target_probability_matrix = full_probability_matrix.get(self.game_state.target_player_id, {})
        all_moves, risk_factor = self._evaluate_all_moves(full_probability_matrix)
        best_move = all_moves[0] if all_moves and all_moves[0]["expected_value"] > 0 else None

        return {
            "best_move": best_move,
            "top_moves": all_moves[:5],
            "probability_matrix": self._serialize_probability_matrix(
                target_probability_matrix,
                self.game_state.hidden_index_by_slot(self.game_state.target_player_id),
            ),
            "full_probability_matrix": self._serialize_full_probability_matrix(
                full_probability_matrix,
            ),
            "search_space_size": search_space_size,
            "opponent_hidden_count": len(target_hidden_slots),
            "risk_factor": risk_factor,
            "effective_weight_sum": total_weight,
            "should_stop": best_move is None,
        }

    def _evaluate_all_moves(
        self,
        full_probability_matrix: FullProbabilityMatrix,
    ) -> Tuple[List[Dict[str, Any]], float]:
        my_hidden_count = self.game_state.my_hidden_count()
        all_moves: List[Dict[str, Any]] = []
        risk_factor = self.decision_engine.calculate_risk_factor(my_hidden_count)

        for player_id, probability_matrix in full_probability_matrix.items():
            hidden_index_by_slot = self.game_state.hidden_index_by_slot(player_id)
            player_moves, _ = self.decision_engine.evaluate_moves(
                probability_matrix,
                my_hidden_count,
                hidden_index_by_slot,
                player_id,
            )
            all_moves.extend(player_moves)

        all_moves.sort(
            key=lambda move: (
                -move["expected_value"],
                -move["win_probability"],
                move["target_player_id"],
                move["target_slot_index"],
                card_sort_key((move["guess_card"][0], move["guess_card"][1])),
            )
        )
        return all_moves, risk_factor

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

    def _serialize_full_probability_matrix(
        self,
        full_probability_matrix: FullProbabilityMatrix,
    ) -> List[Dict[str, Any]]:
        serialized_players: List[Dict[str, Any]] = []

        for player_id in full_probability_matrix:
            hidden_index_by_slot = self.game_state.hidden_index_by_slot(player_id)
            serialized_players.append(
                {
                    "player_id": player_id,
                    "positions": self._serialize_probability_matrix(
                        full_probability_matrix[player_id],
                        hidden_index_by_slot,
                    ),
                }
            )

        return serialized_players
