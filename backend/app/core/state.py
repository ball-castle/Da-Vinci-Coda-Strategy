from dataclasses import dataclass, field
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
    action_type: str = "guess"

    def guessed_card(self) -> Optional[Card]:
        if self.guessed_color in CARD_COLORS and self.guessed_value is not None:
            return (self.guessed_color, self.guessed_value)
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

    def known_cards(self) -> List[Card]:
        known_cards: List[Card] = []
        for player in self.players.values():
            known_cards.extend(player.known_cards())
        return known_cards

    def my_hidden_count(self) -> int:
        return sum(1 for slot in self.self_player().ordered_slots() if not slot.is_revealed)

    def target_hidden_slots(self) -> List[CardSlot]:
        return self.target_player().hidden_slots()

    def hidden_index_by_slot(self, player_id: str) -> Dict[int, int]:
        player = self.get_player(player_id)
        return {
            slot.slot_index: hidden_index
            for hidden_index, slot in enumerate(player.hidden_slots())
        }


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
