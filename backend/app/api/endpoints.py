from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.engine import GameController
from app.core.session import GameSessionManager
from app.core.state import (
    CARD_COLORS,
    JOKER,
    MAX_CARD_VALUE,
    CardSlot,
    GameState,
    GuessAction,
    PlayerState,
    build_legacy_game_state,
)

router = APIRouter()
session_manager = GameSessionManager()


class OpponentAction(BaseModel):
    type: str = "guess"
    target_player_id: Optional[str] = None
    target_slot_index: Optional[int] = None
    target_color: Optional[str] = None
    target_num: Any = None
    result: bool = False
    continued_turn: Optional[bool] = None
    revealed_player_id: Optional[str] = None
    revealed_slot_index: Optional[int] = None
    revealed_color: Optional[str] = None
    revealed_value: Any = None


class CardSlotPayload(BaseModel):
    slot_index: int
    color: Optional[str] = None
    value: Any = None
    is_revealed: bool = False
    is_newly_drawn: bool = False


class PlayerStatePayload(BaseModel):
    player_id: str
    slots: List[CardSlotPayload] = Field(default_factory=list)


class GuessActionPayload(BaseModel):
    guesser_id: str
    target_player_id: str
    target_slot_index: Optional[int] = None
    guessed_color: Optional[str] = None
    guessed_value: Any = None
    result: bool = False
    continued_turn: Optional[bool] = None
    revealed_player_id: Optional[str] = None
    revealed_slot_index: Optional[int] = None
    revealed_color: Optional[str] = None
    revealed_value: Any = None
    action_type: str = "guess"


class GameStatePayload(BaseModel):
    self_player_id: str = "me"
    target_player_id: str = "opponent"
    players: List[PlayerStatePayload]
    actions: List[GuessActionPayload] = Field(default_factory=list)


class TurnRequest(BaseModel):
    session_id: Optional[str] = None
    state: Optional[GameStatePayload] = None
    my_cards: Optional[List[List[Any]]] = None
    public_cards: Optional[Dict[str, List[List[Any]]]] = None
    opponent_card_count: Optional[int] = None
    opponent_hidden_count: Optional[int] = None
    opponent_history: List[OpponentAction] = Field(default_factory=list)


def _coerce_color(color: Optional[str], field_name: str) -> Optional[str]:
    if color is None:
        return None
    if color not in CARD_COLORS:
        raise HTTPException(status_code=400, detail=f"{field_name} 非法颜色: {color}。只支持 B 或 W。")
    return color


def _coerce_card_value(value: Any, field_name: str) -> Any:
    if value is None:
        return None
    if value == JOKER:
        return value
    if not isinstance(value, int):
        raise HTTPException(status_code=400, detail=f"{field_name} 非法牌值: {value}。必须是 0-11 或 '-'.")
    if not 0 <= value <= MAX_CARD_VALUE:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} 非法牌值: {value}。必须落在 0 到 {MAX_CARD_VALUE} 之间。",
        )
    return value


def _coerce_card(raw_card: List[Any]) -> Tuple[str, Any]:
    if len(raw_card) != 2:
        raise HTTPException(status_code=400, detail="卡牌格式必须是 [颜色, 数值]。")

    color = _coerce_color(raw_card[0], "card")
    value = _coerce_card_value(raw_card[1], "card")
    if color is None:
        raise HTTPException(status_code=400, detail="卡牌必须提供颜色。")
    return color, value


def _build_legacy_actions(opponent_history: List[OpponentAction]) -> List[GuessAction]:
    actions: List[GuessAction] = []
    for action in opponent_history:
        actions.append(
            GuessAction(
                guesser_id="opponent",
                target_player_id=action.target_player_id or "me",
                target_slot_index=action.target_slot_index,
                guessed_color=_coerce_color(action.target_color, "opponent_history.target_color"),
                guessed_value=_coerce_card_value(action.target_num, "opponent_history.target_num"),
                result=action.result,
                continued_turn=action.continued_turn,
                revealed_player_id=action.revealed_player_id,
                revealed_slot_index=action.revealed_slot_index,
                revealed_color=_coerce_color(action.revealed_color, "opponent_history.revealed_color"),
                revealed_value=_coerce_card_value(action.revealed_value, "opponent_history.revealed_value"),
                action_type=action.type,
            )
        )
    return actions


def _build_structured_state(payload: GameStatePayload) -> GameState:
    players: Dict[str, PlayerState] = {}

    for player_payload in payload.players:
        if player_payload.player_id in players:
            raise HTTPException(status_code=400, detail=f"重复的 player_id: {player_payload.player_id}。")

        slots: List[CardSlot] = []
        used_slot_indexes = set()
        for slot_payload in player_payload.slots:
            if slot_payload.slot_index < 0:
                raise HTTPException(status_code=400, detail="slot_index 不能是负数。")
            if slot_payload.slot_index in used_slot_indexes:
                raise HTTPException(
                    status_code=400,
                    detail=f"玩家 {player_payload.player_id} 存在重复 slot_index: {slot_payload.slot_index}。",
                )

            color = _coerce_color(slot_payload.color, "slot.color")
            value = _coerce_card_value(slot_payload.value, "slot.value")
            if value is not None and color is None:
                raise HTTPException(status_code=400, detail="当 slot.value 已知时，slot.color 也必须提供。")
            if slot_payload.is_revealed and value is None:
                raise HTTPException(status_code=400, detail="明牌 slot 必须提供 value。")

            used_slot_indexes.add(slot_payload.slot_index)
            slots.append(
                CardSlot(
                    slot_index=slot_payload.slot_index,
                    color=color,
                    value=value,
                    is_revealed=slot_payload.is_revealed,
                    is_newly_drawn=slot_payload.is_newly_drawn,
                )
            )

        players[player_payload.player_id] = PlayerState(
            player_id=player_payload.player_id,
            slots=slots,
        )

    if payload.self_player_id not in players:
        raise HTTPException(status_code=400, detail="state.self_player_id 不存在于 players 中。")
    if payload.target_player_id not in players:
        raise HTTPException(status_code=400, detail="state.target_player_id 不存在于 players 中。")

    actions: List[GuessAction] = []
    for action_payload in payload.actions:
        if action_payload.guesser_id not in players:
            raise HTTPException(status_code=400, detail=f"未知 guesser_id: {action_payload.guesser_id}。")
        if action_payload.target_player_id not in players:
            raise HTTPException(status_code=400, detail=f"未知 target_player_id: {action_payload.target_player_id}。")
        if action_payload.revealed_player_id is not None and action_payload.revealed_player_id not in players:
            raise HTTPException(status_code=400, detail=f"未知 revealed_player_id: {action_payload.revealed_player_id}。")

        actions.append(
            GuessAction(
                guesser_id=action_payload.guesser_id,
                target_player_id=action_payload.target_player_id,
                target_slot_index=action_payload.target_slot_index,
                guessed_color=_coerce_color(action_payload.guessed_color, "action.guessed_color"),
                guessed_value=_coerce_card_value(action_payload.guessed_value, "action.guessed_value"),
                result=action_payload.result,
                continued_turn=action_payload.continued_turn,
                revealed_player_id=action_payload.revealed_player_id,
                revealed_slot_index=action_payload.revealed_slot_index,
                revealed_color=_coerce_color(action_payload.revealed_color, "action.revealed_color"),
                revealed_value=_coerce_card_value(action_payload.revealed_value, "action.revealed_value"),
                action_type=action_payload.action_type,
            )
        )

    return GameState(
        self_player_id=payload.self_player_id,
        target_player_id=payload.target_player_id,
        players=players,
        actions=actions,
    )


def _build_game_state(req: TurnRequest) -> GameState:
    if req.state is not None:
        return _build_structured_state(req.state)

    if req.my_cards is None or req.public_cards is None or req.opponent_card_count is None:
        raise HTTPException(
            status_code=400,
            detail="必须提供 state，或者同时提供 my_cards / public_cards / opponent_card_count。",
        )

    my_cards = [_coerce_card(card) for card in req.my_cards]
    public_cards = {
        player: [_coerce_card(card) for card in cards]
        for player, cards in req.public_cards.items()
    }

    opponent_revealed_count = len(public_cards.get("opponent", []))
    if req.opponent_hidden_count is not None:
        if req.opponent_hidden_count < 0:
            raise HTTPException(status_code=400, detail="opponent_hidden_count 不能是负数。")
        opponent_total_cards = req.opponent_hidden_count + opponent_revealed_count
    else:
        opponent_total_cards = req.opponent_card_count

    if opponent_total_cards < opponent_revealed_count:
        raise HTTPException(
            status_code=400,
            detail="对手总牌数不能小于已公开的对手明牌数量。",
        )

    return build_legacy_game_state(
        my_cards=my_cards,
        public_cards=public_cards,
        opponent_total_cards=opponent_total_cards,
        actions=_build_legacy_actions(req.opponent_history),
    )


@router.post("/turn")
def calculate_turn(req: TurnRequest):
    # Ensure session tracking
    session_id = session_manager.get_or_create_session(req.session_id)
    session_data = session_manager.get_session(session_id)
    
    game_state = _build_game_state(req)
    
    # Optional: Persist new actions into the session history
    if req.state and req.state.actions:
        session_data['history'] = req.state.actions
    
    controller = GameController(game_state)
    result = controller.run_turn()
    result["session_id"] = session_id
    result["input_summary"] = {
        "self_player_id": game_state.self_player_id,
        "target_player_id": game_state.target_player_id,
        "player_count": len(game_state.players),
        "target_total_slots": len(game_state.target_player().ordered_slots()),
        "target_hidden_count": result["opponent_hidden_count"],
        "action_count": len(game_state.actions),
    }
    return result
