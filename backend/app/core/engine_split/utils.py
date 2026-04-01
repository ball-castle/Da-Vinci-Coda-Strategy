from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.core.state import Card, JOKER, MAX_CARD_VALUE
from .models import CARD_COLOR_RANK, SlotKey

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

