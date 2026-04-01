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
from .models import SlotKey, HardConstraintSet
from .utils import slot_key

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


