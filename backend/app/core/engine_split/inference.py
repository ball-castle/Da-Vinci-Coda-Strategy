from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, replace
from math import ceil, exp, log2, sqrt
from pathlib import Path
from random import Random
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.core.state import (
    CARD_COLORS,
    JOKER,
    MAX_CARD_VALUE,
    Card,
    CardSlot,
    GameState,
    GuessAction,
    PlayerState,
)

ProbabilityMatrix = Dict[int, Dict[Card, float]]
FullProbabilityMatrix = Dict[str, ProbabilityMatrix]
SlotKey = Tuple[str, int]

CARD_COLOR_RANK = {"B": 0, "W": 1}
SOFT_BEHAVIOR_BLEND = 0.28
from .models import HardConstraintSet, FullProbabilityMatrix, ProbabilityMatrix, SlotKey, HiddenPosition, GuessSignal
from .utils import slot_key, card_sort_key, normalize_card_distribution
from .constraints import HardConstraintCompiler
from .behavior import BehavioralLikelihoodModel

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


