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
    action_index: int
    guesser_id: str
    target_player_id: str
    target_slot_index: int
    guessed_card: Card
    result: bool
    continued_turn: Optional[bool]


@dataclass
class SearchTreeNode:
    label: str
    prior: float
    rollout_value: float
    move: Optional[Dict[str, Any]] = None
    is_terminal: bool = False
    visits: float = 0.0
    value_sum: float = 0.0
    peak_value: float = 0.0
    positive_value_count: float = 0.0
    children: Optional[List["SearchTreeNode"]] = None
    success_matrix: Optional[FullProbabilityMatrix] = None
    my_hidden_count: int = 0
    hidden_index_by_player: Optional[Dict[str, Dict[int, int]]] = None
    behavior_model: Optional["BehavioralLikelihoodModel"] = None
    guess_signals_by_player: Optional[Dict[str, Sequence[GuessSignal]]] = None
    acting_player_id: Optional[str] = None
    behavior_guidance_profile: Optional[Dict[str, float]] = None
    game_state: Optional[GameState] = None
    behavior_map_hypothesis: Optional[Dict[str, Dict[int, Card]]] = None
    rollout_depth_remaining: int = 0
    perspective_sign: float = 1.0
    search_mode: str = "mcts"


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


