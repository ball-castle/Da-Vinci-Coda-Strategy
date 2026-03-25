from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import log2
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.core.state import (
    CARD_COLORS,
    JOKER,
    MAX_CARD_VALUE,
    Card,
    GameState,
)

ProbabilityMatrix = Dict[int, Dict[Card, float]]
FullProbabilityMatrix = Dict[str, ProbabilityMatrix]
SlotKey = Tuple[str, int]

CARD_COLOR_RANK = {"B": 0, "W": 1}
SOFT_BEHAVIOR_BLEND = 0.28


@dataclass(frozen=True)
class HiddenPosition:
    player_id: str
    slot_index: int
    order_index: int
    color: Optional[str]


@dataclass
class HardConstraintSet:
    fixed_by_slot: Dict[SlotKey, Card]
    forbidden_by_slot: Dict[SlotKey, Set[Card]]


@dataclass(frozen=True)
class GuessSignal:
    guesser_id: str
    target_player_id: str
    target_slot_index: int
    guessed_card: Card
    result: bool
    continued_turn: Optional[bool]


def numeric_card_value(card: Optional[Card]) -> Optional[int]:
    if card is None:
        return None
    value = card[1]
    if value == JOKER or not isinstance(value, int):
        return None
    return int(value)


def card_sort_key(card: Card) -> Tuple[int, int]:
    color, value = card
    if value == JOKER:
        return (MAX_CARD_VALUE + 1, CARD_COLOR_RANK[color])
    return (int(value), CARD_COLOR_RANK[color])


def serialize_card(card: Card) -> List[Any]:
    return [card[0], card[1]]


def normalize_card_distribution(weights: Dict[Card, float]) -> Dict[Card, float]:
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {card: weight / total for card, weight in weights.items()}


def slot_key(player_id: str, slot_index: int) -> SlotKey:
    return (player_id, slot_index)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class HardConstraintCompiler:
    """Compile hard public evidence into fixed/forbidden slot constraints."""

    def __init__(self, game_state: GameState):
        self.game_state = game_state

    def compile(self) -> HardConstraintSet:
        fixed_by_slot: Dict[SlotKey, Card] = {}
        forbidden_by_slot: Dict[SlotKey, Set[Card]] = defaultdict(set)

        for player_id in self.game_state.inference_player_ids():
            for slot in self.game_state.resolved_ordered_slots(player_id):
                known = slot.known_card()
                if known is not None:
                    self._fix_card(
                        fixed_by_slot,
                        forbidden_by_slot,
                        slot_key(player_id, slot.slot_index),
                        known,
                    )

        for action in getattr(self.game_state, "actions", ()):
            if getattr(action, "action_type", None) != "guess":
                continue

            guessed_card = action.guessed_card()
            target_player_id = getattr(action, "target_player_id", None)
            target_slot_index = getattr(action, "target_slot_index", None)
            if guessed_card is not None and target_player_id is not None and target_slot_index is not None:
                key = slot_key(target_player_id, target_slot_index)
                if getattr(action, "result", False):
                    self._fix_card(fixed_by_slot, forbidden_by_slot, key, guessed_card)
                else:
                    forbidden_by_slot[key].add(guessed_card)

            revealed_card = self._revealed_card_from_action(action)
            revealed_player_id = getattr(action, "revealed_player_id", None)
            revealed_slot_index = getattr(action, "revealed_slot_index", None)
            if revealed_card is not None and revealed_player_id is not None and revealed_slot_index is not None:
                self._fix_card(
                    fixed_by_slot,
                    forbidden_by_slot,
                    slot_key(revealed_player_id, revealed_slot_index),
                    revealed_card,
                )

        return HardConstraintSet(
            fixed_by_slot=fixed_by_slot,
            forbidden_by_slot={key: set(value) for key, value in forbidden_by_slot.items()},
        )

    def _fix_card(
        self,
        fixed_by_slot: Dict[SlotKey, Card],
        forbidden_by_slot: Dict[SlotKey, Set[Card]],
        key: SlotKey,
        card: Card,
    ) -> None:
        current = fixed_by_slot.get(key)
        if current is not None and current != card:
            raise ValueError(
                f"Conflicting hard evidence for slot {key}: {current} vs {card}"
            )
        if card in forbidden_by_slot.get(key, set()):
            raise ValueError(
                f"Slot {key} cannot be both fixed to and forbidden from {card}"
            )
        fixed_by_slot[key] = card

    def _revealed_card_from_action(self, action: Any) -> Optional[Card]:
        if hasattr(action, "revealed_card"):
            revealed = action.revealed_card()
            if revealed is not None:
                return revealed

        color = getattr(action, "revealed_color", None)
        value = getattr(action, "revealed_value", None)
        if color is None or value is None:
            return None
        return (color, value)


class BehavioralLikelihoodModel:
    """Structured soft action likelihood model, separate from hard constraints."""

    SELF_EXACT_GUESS_PENALTY = 0.14
    SELF_ADJACENT_ANCHOR_BONUS = 1.08

    TARGET_IN_INTERVAL_BONUS = 1.15
    TARGET_NARROW_INTERVAL_BONUS = 1.10
    TARGET_OUTSIDE_INTERVAL_PENALTY = 0.80
    TARGET_NEIGHBOR_BONUS = 1.10
    TARGET_CLOSE_BONUS = 1.04
    TARGET_FAR_PENALTY = 0.97
    WRONG_COLOR_SLOT_PENALTY = 0.88

    CONTINUE_HIGH_ATTACKABILITY_BONUS = 1.10
    CONTINUE_LOW_ATTACKABILITY_PENALTY = 0.93
    STOP_LOW_ATTACKABILITY_BONUS = 1.04
    STOP_HIGH_ATTACKABILITY_PENALTY = 0.97

    ATTACKABILITY_TIGHT_THRESHOLD = 0.34
    CONTINUATION_PRIOR_BASE = 0.52
    CONTINUATION_ATTACKABILITY_GAIN = 1.35
    CONTINUATION_MIN = 0.08
    CONTINUATION_MAX = 0.95
    EPSILON = 1e-9

    def build_guess_signals(
        self,
        game_state: GameState,
    ) -> Dict[str, List[GuessSignal]]:
        signals_by_player: Dict[str, List[GuessSignal]] = defaultdict(list)

        for action in getattr(game_state, "actions", ()):
            if getattr(action, "action_type", None) != "guess":
                continue

            guessed_card = action.guessed_card()
            guesser_id = getattr(action, "guesser_id", None)
            target_player_id = getattr(action, "target_player_id", None)
            target_slot_index = getattr(action, "target_slot_index", None)
            if (
                guessed_card is None
                or guesser_id is None
                or target_player_id is None
                or target_slot_index is None
            ):
                continue

            signals_by_player[guesser_id].append(
                GuessSignal(
                    guesser_id=guesser_id,
                    target_player_id=target_player_id,
                    target_slot_index=target_slot_index,
                    guessed_card=guessed_card,
                    result=bool(getattr(action, "result", False)),
                    continued_turn=getattr(action, "continued_turn", None),
                )
            )

        return signals_by_player

    def continuation_profile(
        self,
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        acting_player_id: Optional[str],
    ) -> Dict[str, float]:
        if acting_player_id is None:
            return {
                "continue_rate": self.CONTINUATION_PRIOR_BASE,
                "observations": 0.0,
                "history_blend": 0.0,
            }

        signals = guess_signals_by_player.get(acting_player_id, ())
        observed = [
            signal
            for signal in signals
            if signal.result and signal.continued_turn is not None
        ]
        observations = len(observed)
        if observations == 0:
            return {
                "continue_rate": self.CONTINUATION_PRIOR_BASE,
                "observations": 0.0,
                "history_blend": 0.0,
            }

        continue_count = sum(1 for signal in observed if signal.continued_turn)
        continue_rate = (continue_count + 1.0) / (observations + 2.0)
        history_blend = min(0.5, 0.18 * observations)
        return {
            "continue_rate": continue_rate,
            "observations": float(observations),
            "history_blend": history_blend,
        }

    def estimate_continue_likelihood(
        self,
        full_probability_matrix: FullProbabilityMatrix,
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        acting_player_id: Optional[str],
        *,
        exclude_slot: Optional[SlotKey] = None,
    ) -> Dict[str, float]:
        attackability = self.estimate_matrix_attackability(
            full_probability_matrix,
            acting_player_id=acting_player_id,
            exclude_slot=exclude_slot,
        )
        attackability_prior = clamp(
            0.5 + self.CONTINUATION_ATTACKABILITY_GAIN * (attackability - self.ATTACKABILITY_TIGHT_THRESHOLD),
            self.CONTINUATION_MIN,
            self.CONTINUATION_MAX,
        )

        profile = self.continuation_profile(guess_signals_by_player, acting_player_id)
        history_blend = profile["history_blend"]
        continue_likelihood = ((1.0 - history_blend) * attackability_prior) + (history_blend * profile["continue_rate"])

        if attackability >= self.ATTACKABILITY_TIGHT_THRESHOLD and profile["continue_rate"] >= 0.60:
            continue_likelihood *= 1.03
        elif attackability < self.ATTACKABILITY_TIGHT_THRESHOLD and profile["continue_rate"] <= 0.45:
            continue_likelihood *= 0.98

        continue_likelihood = clamp(continue_likelihood, self.CONTINUATION_MIN, self.CONTINUATION_MAX)
        return {
            "continue_likelihood": continue_likelihood,
            "attackability": attackability,
            "history_continue_rate": profile["continue_rate"],
            "history_observations": profile["observations"],
        }

    def score_hypothesis(
        self,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        game_state: GameState,
    ) -> float:
        weight = 1.0

        for signals in guess_signals_by_player.values():
            for signal in signals:
                weight *= self._score_signal(game_state, hypothesis_by_player, signal)
                if weight <= 0.0:
                    return 0.0

        return weight

    def estimate_attackability(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        *,
        acting_player_id: Optional[str] = None,
        exclude_slot: Optional[SlotKey] = None,
    ) -> float:
        best = 0.0
        for player_id in game_state.inference_player_ids():
            if acting_player_id is not None and player_id == acting_player_id:
                continue

            for slot in game_state.resolved_ordered_slots(player_id):
                if slot.known_card() is not None:
                    continue
                key = slot_key(player_id, slot.slot_index)
                if exclude_slot is not None and key == exclude_slot:
                    continue

                low, high, width = self._slot_numeric_interval(
                    game_state,
                    hypothesis_by_player,
                    player_id,
                    slot.slot_index,
                )
                color_bonus = 1.0
                if getattr(slot, "color", None) is not None:
                    color_bonus = 1.12
                confidence = color_bonus / max(1.0, float(width + 1))
                if low is not None and high is not None and width <= 2:
                    confidence *= 1.08
                best = max(best, confidence)
        return best

    def estimate_matrix_attackability(
        self,
        full_probability_matrix: FullProbabilityMatrix,
        *,
        acting_player_id: Optional[str] = None,
        exclude_slot: Optional[SlotKey] = None,
    ) -> float:
        best = 0.0
        for player_id, probability_matrix in full_probability_matrix.items():
            if acting_player_id is not None and player_id == acting_player_id:
                continue
            for slot_index, slot_distribution in probability_matrix.items():
                key = slot_key(player_id, slot_index)
                if exclude_slot is not None and key == exclude_slot:
                    continue
                if not slot_distribution:
                    continue
                max_probability = max(slot_distribution.values())
                concentration = sum(probability * probability for probability in slot_distribution.values())
                effective_support = 1.0 / max(self.EPSILON, concentration)
                certainty = max_probability / max(1.0, effective_support ** 0.5)
                if len(slot_distribution) <= 2:
                    certainty *= 1.05
                best = max(best, certainty)
        return clamp(best, 0.0, 1.0)

    def _score_signal(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
    ) -> float:
        target_card = self._resolve_slot_card(
            game_state,
            hypothesis_by_player,
            signal.target_player_id,
            signal.target_slot_index,
        )
        if target_card is None:
            return 1.0

        weight = 1.0
        guesser_cards = self._player_cards(
            game_state,
            signal.guesser_id,
            hypothesis_by_player.get(signal.guesser_id, {}),
        )
        weight *= self._score_self_hand(guesser_cards, signal.guessed_card)
        weight *= self._score_target_slot(
            game_state,
            hypothesis_by_player,
            signal,
            target_card,
        )
        weight *= self._score_continue_decision(
            game_state,
            hypothesis_by_player,
            signal,
        )
        return max(self.EPSILON, weight)

    def _player_cards(
        self,
        game_state: GameState,
        player_id: str,
        hidden_cards_by_slot: Dict[int, Card],
    ) -> Sequence[Card]:
        cards: List[Card] = []
        for slot in game_state.resolved_ordered_slots(player_id):
            known = slot.known_card()
            if known is not None:
                cards.append(known)
                continue
            inferred = hidden_cards_by_slot.get(slot.slot_index)
            if inferred is not None:
                cards.append(inferred)
        return cards

    def _score_self_hand(
        self,
        player_cards: Sequence[Card],
        guessed_card: Card,
    ) -> float:
        weight = 1.0
        if guessed_card in player_cards:
            weight *= self.SELF_EXACT_GUESS_PENALTY

        guessed_numeric = numeric_card_value(guessed_card)
        if guessed_numeric is None:
            return weight

        same_color_values = {
            numeric_card_value(card)
            for card in player_cards
            if card[0] == guessed_card[0] and numeric_card_value(card) is not None
        }
        if (guessed_numeric - 1) in same_color_values or (guessed_numeric + 1) in same_color_values:
            weight *= self.SELF_ADJACENT_ANCHOR_BONUS
        return weight

    def _score_target_slot(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
        target_card: Card,
    ) -> float:
        guessed_card = signal.guessed_card
        weight = 1.0

        try:
            slot = game_state.get_slot(signal.target_player_id, signal.target_slot_index)
        except ValueError:
            return 1.0

        known_slot_color = getattr(slot, "color", None)
        if known_slot_color is not None and guessed_card[0] != known_slot_color:
            weight *= self.WRONG_COLOR_SLOT_PENALTY

        guessed_numeric = numeric_card_value(guessed_card)
        target_numeric = numeric_card_value(target_card)
        if guessed_numeric is None or target_numeric is None:
            return weight

        low, high, width = self._slot_numeric_interval(
            game_state,
            hypothesis_by_player,
            signal.target_player_id,
            signal.target_slot_index,
        )

        if low is not None and high is not None:
            if low <= guessed_numeric <= high:
                weight *= self.TARGET_IN_INTERVAL_BONUS
                if width <= 2:
                    weight *= self.TARGET_NARROW_INTERVAL_BONUS
            else:
                weight *= self.TARGET_OUTSIDE_INTERVAL_PENALTY

        distance = abs(guessed_numeric - target_numeric)
        if distance == 0:
            return weight
        if distance == 1:
            return weight * self.TARGET_NEIGHBOR_BONUS
        if distance <= 2:
            return weight * self.TARGET_CLOSE_BONUS
        return weight * (self.TARGET_FAR_PENALTY ** max(1, distance - 2))

    def _score_continue_decision(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        signal: GuessSignal,
    ) -> float:
        if not signal.result or signal.continued_turn is None:
            return 1.0

        attackability = self.estimate_attackability(
            game_state,
            hypothesis_by_player,
            acting_player_id=signal.guesser_id,
            exclude_slot=slot_key(signal.target_player_id, signal.target_slot_index),
        )
        if signal.continued_turn:
            if attackability >= self.ATTACKABILITY_TIGHT_THRESHOLD:
                return self.CONTINUE_HIGH_ATTACKABILITY_BONUS
            return self.CONTINUE_LOW_ATTACKABILITY_PENALTY

        if attackability < self.ATTACKABILITY_TIGHT_THRESHOLD:
            return self.STOP_LOW_ATTACKABILITY_BONUS
        return self.STOP_HIGH_ATTACKABILITY_PENALTY

    def _slot_numeric_interval(
        self,
        game_state: GameState,
        hypothesis_by_player: Dict[str, Dict[int, Card]],
        player_id: str,
        slot_index: int,
    ) -> Tuple[Optional[int], Optional[int], int]:
        slots = game_state.resolved_ordered_slots(player_id)
        index_by_slot = {slot.slot_index: idx for idx, slot in enumerate(slots)}
        order_index = index_by_slot.get(slot_index)
        if order_index is None:
            return None, None, MAX_CARD_VALUE

        lower: Optional[int] = None
        upper: Optional[int] = None

        cursor = order_index - 1
        while cursor >= 0:
            left_card = self._resolve_slot_card(game_state, hypothesis_by_player, player_id, slots[cursor].slot_index)
            left_value = numeric_card_value(left_card)
            if left_value is not None:
                lower = left_value
                break
            cursor -= 1

        cursor = order_index + 1
        while cursor < len(slots):
            right_card = self._resolve_slot_card(game_state, hypothesis_by_player, player_id, slots[cursor].slot_index)
            right_value = numeric_card_value(right_card)
            if right_value is not None:
                upper = right_value
                break
            cursor += 1

        effective_low = 0 if lower is None else lower
        effective_high = MAX_CARD_VALUE if upper is None else upper
        if effective_high < effective_low:
            effective_high = effective_low
        width = max(0, effective_high - effective_low)
        return lower, upper, width

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

class DaVinciInferenceEngine:
    """Infer hard posterior, soft posterior, and blended posterior for all hidden slots."""

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
        self.hard_constraints = HardConstraintCompiler(game_state).compile()
        self.preassigned_hidden: Dict[str, Dict[int, Card]] = {
            player_id: {} for player_id in self.inference_player_ids
        }
        self.search_positions: List[HiddenPosition] = []
        self.position_by_key: Dict[SlotKey, HiddenPosition] = {}
        self.player_slot_order: Dict[str, Dict[int, int]] = {}
        self.publicly_collapsed_slots: Set[SlotKey] = set()

        self._build_slot_views()
        self.available_cards = tuple(self._get_available_cards())
        self.base_domains = self._build_base_domains()
        self._validate_initial_domains()

    def _build_all_possible_cards(self) -> Tuple[Card, ...]:
        ordered_cards = [
            (color, value)
            for value in range(MAX_CARD_VALUE + 1)
            for color in CARD_COLORS
        ]
        ordered_cards.extend((color, JOKER) for color in CARD_COLORS)
        return tuple(ordered_cards)

    def _build_slot_views(self) -> None:
        for player_id, slots in self.player_slots.items():
            order_map: Dict[int, int] = {}
            for order_index, slot in enumerate(slots):
                order_map[slot.slot_index] = order_index
                key = slot_key(player_id, slot.slot_index)
                forced = self.hard_constraints.fixed_by_slot.get(key)
                known = slot.known_card()

                if known is not None:
                    if forced is not None and forced != known:
                        raise ValueError(
                            f"Slot {key} conflicts between state {known} and action evidence {forced}"
                        )
                    continue

                if forced is not None:
                    self.preassigned_hidden[player_id][slot.slot_index] = forced
                    self.publicly_collapsed_slots.add(key)
                    continue

                if slot.value is None:
                    hidden = HiddenPosition(
                        player_id=player_id,
                        slot_index=slot.slot_index,
                        order_index=order_index,
                        color=getattr(slot, "color", None),
                    )
                    self.search_positions.append(hidden)
                    self.position_by_key[key] = hidden
            self.player_slot_order[player_id] = order_map

    def _get_available_cards(self) -> Tuple[Card, ...]:
        known_cards = set(self.game_state.known_cards())
        fixed_hidden_cards = {
            card
            for cards_by_slot in self.preassigned_hidden.values()
            for card in cards_by_slot.values()
        }
        used_cards = known_cards | fixed_hidden_cards
        return tuple(
            card
            for card in self._build_all_possible_cards()
            if card not in used_cards
        )

    def _build_base_domains(self) -> Dict[SlotKey, Tuple[Card, ...]]:
        base_domains: Dict[SlotKey, Tuple[Card, ...]] = {}
        for position in self.search_positions:
            key = slot_key(position.player_id, position.slot_index)
            forbidden = self.hard_constraints.forbidden_by_slot.get(key, set())
            domain = []
            for card in self.available_cards:
                if position.color is not None and card[0] != position.color:
                    continue
                if card in forbidden:
                    continue
                domain.append(card)
            base_domains[key] = tuple(domain)
        return base_domains

    def _validate_initial_domains(self) -> None:
        if len(set(self.available_cards)) != len(self.available_cards):
            raise ValueError("Available card pool contains duplicates.")
        for key, domain in self.base_domains.items():
            if not domain:
                raise ValueError(f"Hidden slot {key} has empty hard domain.")

    def infer_hidden_probabilities(
        self,
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        behavior_model: BehavioralLikelihoodModel,
    ) -> Tuple[
        FullProbabilityMatrix,
        FullProbabilityMatrix,
        FullProbabilityMatrix,
        int,
        float,
    ]:
        if not self.search_positions and not any(self.preassigned_hidden.values()):
            return {}, {}, {}, 0, 0.0

        hard_position_weights: Dict[str, DefaultDict[int, DefaultDict[Card, float]]] = {
            player_id: defaultdict(lambda: defaultdict(float))
            for player_id in self.inference_player_ids
        }
        soft_position_weights: Dict[str, DefaultDict[int, DefaultDict[Card, float]]] = {
            player_id: defaultdict(lambda: defaultdict(float))
            for player_id in self.inference_player_ids
        }

        legal_world_count = 0
        total_soft_weight = 0.0
        current_assignment: Dict[str, Dict[int, Card]] = {
            player_id: dict(cards_by_slot)
            for player_id, cards_by_slot in self.preassigned_hidden.items()
        }
        dead_state_cache: Set[Tuple[Tuple[Card, ...], Tuple[Tuple[str, Tuple[Optional[Card], ...]], ...]]] = set()

        def dfs(remaining_positions: Tuple[HiddenPosition, ...], remaining_cards: Tuple[Card, ...]) -> bool:
            nonlocal legal_world_count, total_soft_weight

            state_signature = self._state_signature(current_assignment, remaining_cards)
            if state_signature in dead_state_cache:
                return False

            if not remaining_positions:
                legal_world_count += 1
                hypothesis_by_player = {
                    player_id: dict(cards_by_slot)
                    for player_id, cards_by_slot in current_assignment.items()
                    if cards_by_slot
                }
                soft_weight = behavior_model.score_hypothesis(
                    hypothesis_by_player,
                    guess_signals_by_player,
                    self.game_state,
                )
                total_soft_weight += soft_weight
                self._accumulate_world(
                    hard_position_weights,
                    soft_position_weights,
                    current_assignment,
                    soft_weight,
                )
                return True

            feasible_domains = self._feasible_domains(remaining_positions, remaining_cards, current_assignment)
            if any(len(domain) == 0 for _, domain in feasible_domains):
                dead_state_cache.add(state_signature)
                return False

            next_position, next_domain = min(feasible_domains, key=lambda item: (len(item[1]), item[0].player_id, item[0].slot_index))
            next_remaining_positions = tuple(
                position for position in remaining_positions if position != next_position
            )
            found_any_solution = False

            for candidate in next_domain:
                current_assignment[next_position.player_id][next_position.slot_index] = candidate
                next_remaining_cards = tuple(card for card in remaining_cards if card != candidate)
                found_any_solution = dfs(next_remaining_positions, next_remaining_cards) or found_any_solution
                del current_assignment[next_position.player_id][next_position.slot_index]

            if not found_any_solution:
                dead_state_cache.add(state_signature)
            return found_any_solution

        dfs(tuple(self.search_positions), self.available_cards)

        if legal_world_count == 0:
            return {}, {}, {}, 0, 0.0

        hard_matrix = self._normalize_matrix(hard_position_weights, legal_world_count)
        soft_matrix = (
            self._normalize_matrix(soft_position_weights, total_soft_weight)
            if total_soft_weight > 0
            else hard_matrix
        )
        blended_matrix = self._blend_probability_matrices(hard_matrix, soft_matrix)
        return hard_matrix, soft_matrix, blended_matrix, legal_world_count, total_soft_weight

    def _state_signature(
        self,
        current_assignment: Dict[str, Dict[int, Card]],
        remaining_cards: Tuple[Card, ...],
    ) -> Tuple[Tuple[Card, ...], Tuple[Tuple[str, Tuple[Optional[Card], ...]], ...]]:
        player_signatures: List[Tuple[str, Tuple[Optional[Card], ...]]] = []
        for player_id in sorted(self.player_slots):
            signature: List[Optional[Card]] = []
            for slot in self.player_slots[player_id]:
                known = slot.known_card()
                if known is not None:
                    signature.append(known)
                    continue
                signature.append(current_assignment[player_id].get(slot.slot_index))
            player_signatures.append((player_id, tuple(signature)))
        return tuple(sorted(remaining_cards, key=card_sort_key)), tuple(player_signatures)

    def _feasible_domains(
        self,
        remaining_positions: Tuple[HiddenPosition, ...],
        remaining_cards: Tuple[Card, ...],
        current_assignment: Dict[str, Dict[int, Card]],
    ) -> List[Tuple[HiddenPosition, Tuple[Card, ...]]]:
        remaining_card_set = set(remaining_cards)
        feasible: List[Tuple[HiddenPosition, Tuple[Card, ...]]] = []
        for position in remaining_positions:
            key = slot_key(position.player_id, position.slot_index)
            candidates = []
            for candidate in self.base_domains[key]:
                if candidate not in remaining_card_set:
                    continue
                if not self._fits_local_order(position, candidate, current_assignment):
                    continue
                candidates.append(candidate)
            candidates.sort(key=card_sort_key)
            feasible.append((position, tuple(candidates)))
        return feasible

    def _fits_local_order(
        self,
        position: HiddenPosition,
        candidate: Card,
        current_assignment: Dict[str, Dict[int, Card]],
    ) -> bool:
        if candidate[1] == JOKER:
            return True

        left_bound = self._nearest_numeric_bound(position.player_id, position.order_index, -1, current_assignment)
        if left_bound is not None and self._card_precedes(candidate, left_bound):
            return False

        right_bound = self._nearest_numeric_bound(position.player_id, position.order_index, 1, current_assignment)
        if right_bound is not None and self._card_precedes(right_bound, candidate):
            return False

        return True

    def _nearest_numeric_bound(
        self,
        player_id: str,
        order_index: int,
        direction: int,
        current_assignment: Dict[str, Dict[int, Card]],
    ) -> Optional[Card]:
        slots = self.player_slots[player_id]
        cursor = order_index + direction
        while 0 <= cursor < len(slots):
            slot = slots[cursor]
            card = slot.known_card()
            if card is None:
                card = current_assignment[player_id].get(slot.slot_index)
            if card is not None and card[1] != JOKER:
                return card
            cursor += direction
        return None

    def _card_precedes(self, left: Card, right: Card) -> bool:
        if left[1] == JOKER or right[1] == JOKER:
            return False
        if left[1] != right[1]:
            return int(left[1]) < int(right[1])
        return CARD_COLOR_RANK[left[0]] < CARD_COLOR_RANK[right[0]]

    def _accumulate_world(
        self,
        hard_position_weights: Dict[str, DefaultDict[int, DefaultDict[Card, float]]],
        soft_position_weights: Dict[str, DefaultDict[int, DefaultDict[Card, float]]],
        current_assignment: Dict[str, Dict[int, Card]],
        soft_weight: float,
    ) -> None:
        for player_id, cards_by_slot in current_assignment.items():
            for slot_index, card in cards_by_slot.items():
                hard_position_weights[player_id][slot_index][card] += 1.0
                soft_position_weights[player_id][slot_index][card] += soft_weight

    def _normalize_matrix(
        self,
        position_weights: Dict[str, DefaultDict[int, DefaultDict[Card, float]]],
        total_weight: float,
    ) -> FullProbabilityMatrix:
        if total_weight <= 0:
            return {}
        result: FullProbabilityMatrix = {}
        for player_id, weighted_slots in position_weights.items():
            if not weighted_slots:
                continue
            result[player_id] = {
                slot_index: {
                    card: weight / total_weight
                    for card, weight in weighted_cards.items()
                }
                for slot_index, weighted_cards in weighted_slots.items()
            }
        return result

    def _blend_probability_matrices(
        self,
        hard_matrix: FullProbabilityMatrix,
        soft_matrix: FullProbabilityMatrix,
    ) -> FullProbabilityMatrix:
        blended: FullProbabilityMatrix = {}
        for player_id in sorted(set(hard_matrix) | set(soft_matrix)):
            player_positions: ProbabilityMatrix = {}
            all_slot_indices = set(hard_matrix.get(player_id, {})) | set(soft_matrix.get(player_id, {}))
            for slot_index in sorted(all_slot_indices):
                hard_dist = hard_matrix.get(player_id, {}).get(slot_index, {})
                soft_dist = soft_matrix.get(player_id, {}).get(slot_index, hard_dist)
                cards = set(hard_dist) | set(soft_dist)
                blended_weights = {
                    card: ((1.0 - SOFT_BEHAVIOR_BLEND) * hard_dist.get(card, 0.0))
                    + (SOFT_BEHAVIOR_BLEND * soft_dist.get(card, 0.0))
                    for card in cards
                }
                player_positions[slot_index] = normalize_card_distribution(blended_weights)
            if player_positions:
                blended[player_id] = player_positions
        return blended


class DaVinciDecisionEngine:
    """Score moves by immediate EV, continuation likelihood, and stop-aware thresholding."""

    HIT_REWARD = 10.0
    MISS_BASE = 12.0
    MISS_ENDGAME_MULTIPLIER = 12.0
    INFORMATION_GAIN_WEIGHT = 0.45
    CONTINUATION_DISCOUNT = 0.72
    MAX_CANDIDATES_PER_SLOT = 8
    MIN_CANDIDATE_PROBABILITY = 1e-9

    STOP_MARGIN_BASE = 0.18
    STOP_MARGIN_RISK_SCALE = 0.035
    STOP_MARGIN_SHORT_HAND = 0.22
    STOP_MARGIN_LOW_CONFIDENCE = 0.28
    STOP_MARGIN_WEAK_CONTINUATION = 0.16
    STOP_MARGIN_WEAK_EDGE = 0.22
    STOP_MARGIN_LOW_ATTACKABILITY = 0.18
    STOP_EDGE_REFERENCE = 0.18
    LOW_CONFIDENCE_GUARD_MARGIN = 0.22
    WEAK_EDGE_GUARD_MARGIN = 0.18
    ATTACKABILITY_REFERENCE = BehavioralLikelihoodModel.ATTACKABILITY_TIGHT_THRESHOLD

    def calculate_risk_factor(self, my_hidden_count: int) -> float:
        exposure = 1.0 / max(1, my_hidden_count)
        return self.MISS_BASE + self.MISS_ENDGAME_MULTIPLIER * exposure

    def evaluate_all_moves(
        self,
        *,
        full_probability_matrix: FullProbabilityMatrix,
        my_hidden_count: int,
        hidden_index_by_player: Dict[str, Dict[int, int]],
        behavior_model: BehavioralLikelihoodModel,
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        acting_player_id: Optional[str],
        blocked_slots: Optional[Set[SlotKey]] = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        risk_factor = self.calculate_risk_factor(my_hidden_count)
        blocked_slots = blocked_slots or set()
        moves: List[Dict[str, Any]] = []

        for player_id in sorted(full_probability_matrix):
            probability_matrix = full_probability_matrix[player_id]
            hidden_index_by_slot = hidden_index_by_player.get(player_id, {})
            for target_slot_index, slot_distribution in probability_matrix.items():
                if slot_key(player_id, target_slot_index) in blocked_slots:
                    continue
                ranked_candidates = sorted(
                    slot_distribution.items(),
                    key=lambda item: (-item[1], card_sort_key(item[0])),
                )
                for card, probability in ranked_candidates[: self.MAX_CANDIDATES_PER_SLOT]:
                    if probability <= self.MIN_CANDIDATE_PROBABILITY:
                        continue
                    move = self._score_single_move(
                        full_probability_matrix=full_probability_matrix,
                        my_hidden_count=my_hidden_count,
                        risk_factor=risk_factor,
                        hidden_index_by_slot=hidden_index_by_slot,
                        player_id=player_id,
                        slot_index=target_slot_index,
                        card=card,
                        probability=probability,
                        slot_distribution=slot_distribution,
                        behavior_model=behavior_model,
                        guess_signals_by_player=guess_signals_by_player,
                        acting_player_id=acting_player_id,
                    )
                    moves.append(move)

        moves.sort(
            key=lambda move: (
                -move["expected_value"],
                -move["win_probability"],
                -move["continuation_value"],
                -move["continuation_likelihood"],
                move["target_player_id"],
                move["target_slot_index"],
                card_sort_key((move["guess_card"][0], move["guess_card"][1])),
            )
        )
        return moves, risk_factor

    def choose_best_move(
        self,
        all_moves: List[Dict[str, Any]],
        *,
        risk_factor: float,
        my_hidden_count: int,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        if not all_moves:
            stop_threshold = self._stop_threshold(
                risk_factor=risk_factor,
                my_hidden_count=my_hidden_count,
                best_move=None,
            )
            return None, {
                "evaluated_move_count": 0,
                "best_immediate_value": 0.0,
                "best_expected_value": 0.0,
                "best_win_probability": 0.0,
                "best_continuation_value": 0.0,
                "best_continuation_likelihood": 0.0,
                "stop_threshold": stop_threshold,
                "stop_score": stop_threshold,
                "continue_score": 0.0,
                "continue_margin": -stop_threshold,
                "recommend_stop": True,
                "decision_score_breakdown": {
                    "base_stop_threshold": stop_threshold,
                    "edge_pressure": 0.0,
                    "attackability_pressure": 0.0,
                },
                "stop_reason": "没有可评估的候选动作。",
            }

        best_move = all_moves[0]
        second_move = all_moves[1] if len(all_moves) > 1 else None
        stop_threshold = self._stop_threshold(
            risk_factor=risk_factor,
            my_hidden_count=my_hidden_count,
            best_move=best_move,
        )
        decision_snapshot = self._evaluate_continue_decision(
            best_move=best_move,
            second_move=second_move,
            stop_threshold=stop_threshold,
            my_hidden_count=my_hidden_count,
        )
        stop_reason = self._build_stop_reason(
            should_continue=decision_snapshot["should_continue"],
            best_move=best_move,
            stop_threshold=stop_threshold,
            stop_score=decision_snapshot["stop_score"],
            continue_score=decision_snapshot["continue_score"],
            continue_margin=decision_snapshot["continue_margin"],
            low_confidence_guard=decision_snapshot["low_confidence_guard"],
            weak_edge_guard=decision_snapshot["weak_edge_guard"],
        )

        decision_summary = {
            "evaluated_move_count": len(all_moves),
            "best_immediate_value": best_move.get("immediate_expected_value", best_move["expected_value"]),
            "best_expected_value": best_move["expected_value"],
            "best_win_probability": best_move["win_probability"],
            "best_continuation_value": best_move.get("continuation_value", 0.0),
            "best_continuation_likelihood": best_move.get("continuation_likelihood", 0.0),
            "best_attackability_after_hit": best_move.get("attackability_after_hit", 0.0),
            "best_gap": decision_snapshot["best_gap"],
            "stop_threshold": stop_threshold,
            "stop_score": decision_snapshot["stop_score"],
            "continue_score": decision_snapshot["continue_score"],
            "continue_margin": decision_snapshot["continue_margin"],
            "recommend_stop": not decision_snapshot["should_continue"],
            "decision_score_breakdown": decision_snapshot["decision_score_breakdown"],
            "stop_reason": stop_reason,
        }
        return (best_move if decision_snapshot["should_continue"] else None), decision_summary

    def _score_single_move(
        self,
        *,
        full_probability_matrix: FullProbabilityMatrix,
        my_hidden_count: int,
        risk_factor: float,
        hidden_index_by_slot: Dict[int, int],
        player_id: str,
        slot_index: int,
        card: Card,
        probability: float,
        slot_distribution: Dict[Card, float],
        behavior_model: BehavioralLikelihoodModel,
        guess_signals_by_player: Dict[str, Sequence[GuessSignal]],
        acting_player_id: Optional[str],
    ) -> Dict[str, Any]:
        info_gain = self._expected_slot_entropy_reduction(slot_distribution, card)
        continuation_value = 0.0
        continuation_likelihood = 0.0
        attackability_after_hit = 0.0
        history_continue_rate = behavior_model.CONTINUATION_PRIOR_BASE
        next_best_immediate_ev = 0.0
        post_hit_continuation_value = 0.0

        success_matrix = self._success_posterior(full_probability_matrix, player_id, slot_index, card)
        if success_matrix:
            next_best_immediate_ev = self._best_immediate_ev(
                success_matrix,
                my_hidden_count,
            )
            continuation_assessment = behavior_model.estimate_continue_likelihood(
                success_matrix,
                guess_signals_by_player,
                acting_player_id,
                exclude_slot=slot_key(player_id, slot_index),
            )
            continuation_likelihood = continuation_assessment["continue_likelihood"]
            attackability_after_hit = continuation_assessment["attackability"]
            history_continue_rate = continuation_assessment["history_continue_rate"]
            post_hit_continuation_value = (
                self.CONTINUATION_DISCOUNT
                * continuation_likelihood
                * max(0.0, next_best_immediate_ev)
            )
            continuation_value = (
                probability
                * post_hit_continuation_value
            )

        hit_reward = probability * self.HIT_REWARD
        miss_penalty = (1.0 - probability) * risk_factor
        info_bonus = info_gain * self.INFORMATION_GAIN_WEIGHT
        immediate_expected_value = hit_reward - miss_penalty + info_bonus
        expected_value = immediate_expected_value + continuation_value

        return {
            "target_player_id": player_id,
            "target_index": hidden_index_by_slot.get(slot_index, slot_index),
            "target_slot_index": slot_index,
            "guess_card": serialize_card(card),
            "win_probability": probability,
            "immediate_expected_value": immediate_expected_value,
            "expected_value": expected_value,
            "information_gain": info_gain,
            "continuation_value": continuation_value,
            "post_hit_continuation_value": post_hit_continuation_value,
            "next_best_immediate_value": next_best_immediate_ev,
            "continuation_likelihood": continuation_likelihood,
            "attackability_after_hit": attackability_after_hit,
            "history_continue_rate": history_continue_rate,
            "hit_reward": hit_reward,
            "miss_penalty": miss_penalty,
            "score_breakdown": {
                "hit_reward": hit_reward,
                "miss_penalty": miss_penalty,
                "information_gain_bonus": info_bonus,
                "immediate_expected_value": immediate_expected_value,
                "continuation_value": continuation_value,
                "post_hit_continuation_value": post_hit_continuation_value,
                "next_best_immediate_value": next_best_immediate_ev,
                "continuation_likelihood": continuation_likelihood,
                "attackability_after_hit": attackability_after_hit,
            },
            "recommendation_reason": self._build_reason(
                probability,
                info_gain,
                continuation_value,
                miss_penalty,
                continuation_likelihood,
            ),
            "target_scope": "player_slots",
        }

    def _evaluate_continue_decision(
        self,
        *,
        best_move: Dict[str, Any],
        second_move: Optional[Dict[str, Any]],
        stop_threshold: float,
        my_hidden_count: int,
    ) -> Dict[str, Any]:
        best_gap = best_move["expected_value"] - (second_move["expected_value"] if second_move else 0.0)

        edge_pressure = 0.0
        if second_move is not None and best_gap < self.STOP_EDGE_REFERENCE:
            edge_pressure = self.STOP_MARGIN_WEAK_EDGE * (
                (self.STOP_EDGE_REFERENCE - max(0.0, best_gap)) / self.STOP_EDGE_REFERENCE
            )

        attackability_after_hit = best_move.get("attackability_after_hit", 0.0)
        attackability_pressure = 0.0
        if attackability_after_hit < self.ATTACKABILITY_REFERENCE:
            attackability_pressure = self.STOP_MARGIN_LOW_ATTACKABILITY * (
                (self.ATTACKABILITY_REFERENCE - attackability_after_hit)
                / max(self.ATTACKABILITY_REFERENCE, 1e-9)
            )

        stop_score = stop_threshold + edge_pressure + attackability_pressure
        continue_score = best_move["expected_value"]
        continue_margin = continue_score - stop_score

        low_confidence_guard = (
            my_hidden_count <= 2
            and best_move["win_probability"] < 0.45
            and best_move.get("continuation_likelihood", 0.0) < 0.55
            and continue_margin < self.LOW_CONFIDENCE_GUARD_MARGIN
        )
        weak_edge_guard = (
            second_move is not None
            and best_gap < 0.10
            and continue_margin < self.WEAK_EDGE_GUARD_MARGIN
        )

        should_continue = (
            continue_margin > 0.0
            and not low_confidence_guard
            and not weak_edge_guard
        )
        return {
            "should_continue": should_continue,
            "continue_score": continue_score,
            "stop_score": stop_score,
            "continue_margin": continue_margin,
            "best_gap": best_gap,
            "low_confidence_guard": low_confidence_guard,
            "weak_edge_guard": weak_edge_guard,
            "decision_score_breakdown": {
                "base_stop_threshold": stop_threshold,
                "edge_pressure": edge_pressure,
                "attackability_pressure": attackability_pressure,
            },
        }

    def _stop_threshold(
        self,
        *,
        risk_factor: float,
        my_hidden_count: int,
        best_move: Optional[Dict[str, Any]],
    ) -> float:
        threshold = self.STOP_MARGIN_BASE + max(0.0, (risk_factor - self.HIT_REWARD) * self.STOP_MARGIN_RISK_SCALE)
        if my_hidden_count <= 2:
            threshold += self.STOP_MARGIN_SHORT_HAND
        if my_hidden_count <= 1:
            threshold += self.STOP_MARGIN_SHORT_HAND

        if best_move is not None:
            if best_move["win_probability"] < 0.5:
                threshold += self.STOP_MARGIN_LOW_CONFIDENCE * (0.5 - best_move["win_probability"]) / 0.5
            if best_move.get("continuation_likelihood", 0.0) < 0.5:
                threshold += self.STOP_MARGIN_WEAK_CONTINUATION * (0.5 - best_move.get("continuation_likelihood", 0.0)) / 0.5
        return threshold

    def _build_stop_reason(
        self,
        *,
        should_continue: bool,
        best_move: Dict[str, Any],
        stop_threshold: float,
        stop_score: float,
        continue_score: float,
        continue_margin: float,
        low_confidence_guard: bool,
        weak_edge_guard: bool,
    ) -> str:
        if should_continue:
            if best_move.get("continuation_value", 0.0) > 0.0:
                return (
                    f"继续评分 {continue_score:.2f} 高于停手评分 {stop_score:.2f}，"
                    "命中后继续收益也支持进攻。"
                )
            return f"继续评分 {continue_score:.2f} 高于停手评分 {stop_score:.2f}，可以继续进攻。"
        if low_confidence_guard:
            return "当前属于高暴露局面，命中率与续压潜力都偏弱，建议停手。"
        if weak_edge_guard:
            return "最佳动作与次优动作差距过小，当前进攻边际优势不足，建议停手。"
        if stop_score > stop_threshold:
            return (
                f"继续评分 {continue_score:.2f} 仍低于停手评分 {stop_score:.2f}；"
                f"当前续压空间不足，净优势 {continue_margin:.2f}。"
            )
        return f"最佳动作期望收益 {best_move['expected_value']:.2f} 未超过停手阈值 {stop_threshold:.2f}。"

    def _build_reason(
        self,
        probability: float,
        info_gain: float,
        continuation_value: float,
        miss_penalty: float,
        continuation_likelihood: float,
    ) -> str:
        if continuation_value > max(0.35, info_gain * self.INFORMATION_GAIN_WEIGHT):
            if continuation_likelihood >= 0.60:
                return "命中后继续收益较高，且续压概率也偏高，适合主动压回合。"
            return "命中后存在续压价值，但更依赖后续局面兑现。"
        if probability >= 0.62 and miss_penalty < self.HIT_REWARD:
            return "当前命中率较高，且风险处于可接受范围。"
        if info_gain * self.INFORMATION_GAIN_WEIGHT >= max(0.45, continuation_value):
            return "即便不是最稳的一猜，也能显著压缩该位置的不确定性。"
        return "这步主要依靠当前命中率与风险平衡取得正收益。"

    def _best_immediate_ev(
        self,
        full_probability_matrix: FullProbabilityMatrix,
        my_hidden_count: int,
    ) -> float:
        risk_factor = self.calculate_risk_factor(my_hidden_count)
        best_value = 0.0
        for probability_matrix in full_probability_matrix.values():
            for slot_distribution in probability_matrix.values():
                for card, probability in slot_distribution.items():
                    info_gain = self._expected_slot_entropy_reduction(slot_distribution, card)
                    value = (probability * self.HIT_REWARD) - ((1.0 - probability) * risk_factor) + (info_gain * self.INFORMATION_GAIN_WEIGHT)
                    best_value = max(best_value, value)
        return best_value

    def _success_posterior(
        self,
        full_probability_matrix: FullProbabilityMatrix,
        target_player_id: str,
        target_slot_index: int,
        guessed_card: Card,
    ) -> FullProbabilityMatrix:
        success_matrix: FullProbabilityMatrix = {}
        for player_id, probability_matrix in full_probability_matrix.items():
            player_positions: ProbabilityMatrix = {}
            for slot_index, slot_distribution in probability_matrix.items():
                if player_id == target_player_id and slot_index == target_slot_index:
                    continue

                adjusted = dict(slot_distribution)
                if guessed_card in adjusted:
                    adjusted.pop(guessed_card, None)
                    adjusted = normalize_card_distribution(adjusted)

                if adjusted:
                    player_positions[slot_index] = adjusted
            if player_positions:
                success_matrix[player_id] = player_positions
        return success_matrix

    def _expected_slot_entropy_reduction(
        self,
        slot_distribution: Dict[Card, float],
        guessed_card: Card,
    ) -> float:
        before = self._entropy(slot_distribution.values())
        hit_probability = slot_distribution.get(guessed_card, 0.0)
        if hit_probability <= 0.0:
            return 0.0
        if hit_probability >= 1.0:
            return before

        miss_distribution = {
            card: probability
            for card, probability in slot_distribution.items()
            if card != guessed_card
        }
        miss_distribution = normalize_card_distribution(miss_distribution)
        after_miss = self._entropy(miss_distribution.values())
        return before - ((1.0 - hit_probability) * after_miss)

    def _entropy(self, probabilities: Iterable[float]) -> float:
        entropy = 0.0
        for probability in probabilities:
            if probability <= 0.0:
                continue
            entropy -= probability * log2(probability)
        return entropy


class GameController:
    """Coordinate hard inference, soft weighting, posterior blending, and move scoring."""

    def __init__(self, game_state: GameState):
        self.game_state = game_state
        self.inference_engine = DaVinciInferenceEngine(game_state)
        self.behavior_model = BehavioralLikelihoodModel()
        self.decision_engine = DaVinciDecisionEngine()

    def run_turn(self) -> Dict[str, Any]:
        has_any_hidden_slots = bool(self.inference_engine.search_positions) or any(
            self.inference_engine.preassigned_hidden.values()
        )
        target_hidden_slots = self.game_state.target_hidden_slots()
        blocked_target_slots = {key for key in self.inference_engine.publicly_collapsed_slots if key[0] == getattr(self.game_state, "target_player_id", None)}
        my_hidden_count = self.game_state.my_hidden_count()
        default_risk = self.decision_engine.calculate_risk_factor(my_hidden_count)

        if not has_any_hidden_slots:
            return {
                "best_move": None,
                "top_moves": [],
                "probability_matrix": [],
                "full_probability_matrix": [],
                "hard_probability_matrix": [],
                "hard_full_probability_matrix": [],
                "soft_full_probability_matrix": [],
                "search_space_size": 0,
                "effective_weight_sum": 0.0,
                "opponent_hidden_count": 0,
                "risk_factor": default_risk,
                "should_stop": True,
            }

        guess_signals_by_player = self.behavior_model.build_guess_signals(self.game_state)
        hard_full_probability_matrix, soft_full_probability_matrix, full_probability_matrix, search_space_size, total_soft_weight = self.inference_engine.infer_hidden_probabilities(
            guess_signals_by_player,
            self.behavior_model,
        )

        hidden_index_by_player = {
            player_id: self.game_state.hidden_index_by_slot(player_id)
            for player_id in full_probability_matrix
        }
        all_moves, risk_factor = self.decision_engine.evaluate_all_moves(
            full_probability_matrix=full_probability_matrix,
            my_hidden_count=my_hidden_count,
            hidden_index_by_player=hidden_index_by_player,
            behavior_model=self.behavior_model,
            guess_signals_by_player=guess_signals_by_player,
            acting_player_id=self.game_state.self_player_id,
            blocked_slots=self.inference_engine.publicly_collapsed_slots,
        )
        best_move, decision_summary = self.decision_engine.choose_best_move(
            all_moves,
            risk_factor=risk_factor,
            my_hidden_count=my_hidden_count,
        )

        target_player_id = getattr(self.game_state, "target_player_id", None)
        target_probability_matrix = full_probability_matrix.get(target_player_id, {})
        target_hard_matrix = hard_full_probability_matrix.get(target_player_id, {})

        return {
            "best_move": best_move,
            "top_moves": all_moves[:10],
            "probability_matrix": self._serialize_probability_matrix(
                target_probability_matrix,
                self.game_state.hidden_index_by_slot(target_player_id) if target_player_id is not None else {},
            ),
            "full_probability_matrix": self._serialize_full_probability_matrix(
                full_probability_matrix,
            ),
            "hard_probability_matrix": self._serialize_probability_matrix(
                target_hard_matrix,
                self.game_state.hidden_index_by_slot(target_player_id) if target_player_id is not None else {},
            ),
            "hard_full_probability_matrix": self._serialize_full_probability_matrix(
                hard_full_probability_matrix,
            ),
            "soft_full_probability_matrix": self._serialize_full_probability_matrix(
                soft_full_probability_matrix,
            ),
            "search_space_size": search_space_size,
            "opponent_hidden_count": max(0, len(target_hidden_slots) - len(blocked_target_slots)),
            "risk_factor": risk_factor,
            "effective_weight_sum": total_soft_weight,
            "behavior_blend": SOFT_BEHAVIOR_BLEND,
            "decision_summary": decision_summary,
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
