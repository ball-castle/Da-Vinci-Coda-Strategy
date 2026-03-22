from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple, Union

CardValue = Union[int, str]
Card = Tuple[str, CardValue]

CARD_COLORS = ("B", "W")
JOKER = "-"
MAX_CARD_VALUE = 11


@dataclass(frozen=True)
class CardSlot:
    slot_index: int
    color: Optional[str] = None
    value: Optional[CardValue] = None
    is_revealed: bool = False
    is_newly_drawn: bool = False

    def known_card(self) -> Optional[Card]:
        if self.color in CARD_COLORS and self.value is not None:
            return (self.color, self.value)
        return None


@dataclass
class PlayerState:
    player_id: str
    slots: List[CardSlot] = field(default_factory=list)

    def ordered_slots(self) -> List[CardSlot]:
        return sorted(self.slots, key=lambda slot: slot.slot_index)

    def hidden_slots(self) -> List[CardSlot]:
        return [slot for slot in self.ordered_slots() if slot.value is None]

    def known_cards(self) -> List[Card]:
        return [card for slot in self.ordered_slots() if (card := slot.known_card()) is not None]


@dataclass(frozen=True)
class GuessAction:
    guesser_id: str
    target_player_id: str
    target_slot_index: Optional[int] = None
    guessed_color: Optional[str] = None
    guessed_value: Optional[CardValue] = None
    result: bool = False
    continued_turn: Optional[bool] = None
    revealed_player_id: Optional[str] = None
    revealed_slot_index: Optional[int] = None
    revealed_color: Optional[str] = None
    revealed_value: Optional[CardValue] = None
    action_type: str = "guess"

    def guessed_card(self) -> Optional[Card]:
        if self.guessed_color in CARD_COLORS and self.guessed_value is not None:
            return (self.guessed_color, self.guessed_value)
        return None

    def revealed_card(self) -> Optional[Card]:
        if self.revealed_color in CARD_COLORS and self.revealed_value is not None:
            return (self.revealed_color, self.revealed_value)
        return None


@dataclass
class GameState:
    self_player_id: str
    target_player_id: str
    players: Dict[str, PlayerState]
    actions: List[GuessAction] = field(default_factory=list)

    def get_player(self, player_id: str) -> PlayerState:
        try:
            return self.players[player_id]
        except KeyError as exc:
            raise ValueError(f"Unknown player_id: {player_id}") from exc

    def self_player(self) -> PlayerState:
        return self.get_player(self.self_player_id)

    def target_player(self) -> PlayerState:
        return self.get_player(self.target_player_id)

    def get_slot(self, player_id: str, slot_index: int) -> CardSlot:
        for slot in self.resolved_ordered_slots(player_id):
            if slot.slot_index == slot_index:
                return slot
        raise ValueError(f"Unknown slot_index {slot_index} for player_id {player_id}")

    def resolved_ordered_slots(self, player_id: str) -> List[CardSlot]:
        player = self.get_player(player_id)
        revealed_by_slot = self.action_revealed_cards_by_player().get(player_id, {})
        resolved_slots: List[CardSlot] = []

        for slot in player.ordered_slots():
            revealed_card = revealed_by_slot.get(slot.slot_index)
            if revealed_card is None:
                resolved_slots.append(slot)
                continue

            resolved_slots.append(
                replace(
                    slot,
                    color=revealed_card[0],
                    value=revealed_card[1],
                    is_revealed=True,
                )
            )

        return resolved_slots

    def known_cards(self) -> List[Card]:
        known_cards: List[Card] = []
        for player_id in self.players:
            for slot in self.resolved_ordered_slots(player_id):
                if card := slot.known_card():
                    known_cards.append(card)
        return known_cards

    def my_hidden_count(self) -> int:
        return sum(1 for slot in self.resolved_ordered_slots(self.self_player_id) if not slot.is_revealed)

    def target_hidden_slots(self) -> List[CardSlot]:
        return [slot for slot in self.resolved_ordered_slots(self.target_player_id) if slot.value is None]

    def inference_player_ids(self) -> List[str]:
        ordered_ids = [self.target_player_id]
        ordered_ids.extend(
            player_id
            for player_id in self.players
            if player_id not in {self.self_player_id, self.target_player_id}
        )
        return ordered_ids

    def hidden_index_by_slot(self, player_id: str) -> Dict[int, int]:
        return {
            slot.slot_index: hidden_index
            for hidden_index, slot in enumerate(
                slot for slot in self.resolved_ordered_slots(player_id) if slot.value is None
            )
        }

    def action_revealed_cards_by_player(self) -> Dict[str, Dict[int, Card]]:
        revealed_by_player: Dict[str, Dict[int, Card]] = {}

        for action in self.actions:
            if action.revealed_player_id is None or action.revealed_slot_index is None:
                continue
            revealed_card = action.revealed_card()
            if revealed_card is None:
                continue

            player_reveals = revealed_by_player.setdefault(action.revealed_player_id, {})
            player_reveals[action.revealed_slot_index] = revealed_card

        return revealed_by_player


def build_legacy_game_state(
    my_cards: Sequence[Card],
    public_cards: Dict[str, Sequence[Card]],
    opponent_total_cards: int,
    actions: Optional[Sequence[GuessAction]] = None,
    self_player_id: str = "me",
    target_player_id: str = "opponent",
) -> GameState:
    my_public_cards = set(public_cards.get(self_player_id, []))
    my_slots = [
        CardSlot(
            slot_index=slot_index,
            color=color,
            value=value,
            is_revealed=(color, value) in my_public_cards,
        )
        for slot_index, (color, value) in enumerate(my_cards)
    ]

    target_public_cards = list(public_cards.get(target_player_id, []))
    target_slots = [
        CardSlot(
            slot_index=slot_index,
            color=color,
            value=value,
            is_revealed=True,
        )
        for slot_index, (color, value) in enumerate(target_public_cards)
    ]

    hidden_count = max(0, opponent_total_cards - len(target_public_cards))
    for offset in range(hidden_count):
        target_slots.append(
            CardSlot(
                slot_index=len(target_slots),
                color=None,
                value=None,
                is_revealed=False,
            )
        )

    players = {
        self_player_id: PlayerState(player_id=self_player_id, slots=my_slots),
        target_player_id: PlayerState(player_id=target_player_id, slots=target_slots),
    }

    return GameState(
        self_player_id=self_player_id,
        target_player_id=target_player_id,
        players=players,
        actions=list(actions or []),
    )
