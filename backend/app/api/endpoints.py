from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.engine import CARD_COLORS, JOKER, MAX_CARD_VALUE, GameController

router = APIRouter()


class OpponentAction(BaseModel):
    type: str = "guess"
    target_color: Optional[str] = None
    target_num: Any = None
    result: bool = False


class TurnRequest(BaseModel):
    my_cards: List[List[Any]]
    public_cards: Dict[str, List[List[Any]]]
    opponent_card_count: int
    opponent_hidden_count: Optional[int] = None
    opponent_history: List[OpponentAction] = Field(default_factory=list)


def _coerce_card(raw_card: List[Any]) -> Tuple[str, Any]:
    if len(raw_card) != 2:
        raise HTTPException(status_code=400, detail="卡牌格式必须是 [颜色, 数值]。")

    color, value = raw_card
    if color not in CARD_COLORS:
        raise HTTPException(status_code=400, detail=f"非法颜色: {color}。只支持 B 或 W。")

    if value != JOKER and not isinstance(value, int):
        raise HTTPException(status_code=400, detail=f"非法牌值: {value}。必须是 0-11 或 '-'.")

    if isinstance(value, int) and not 0 <= value <= MAX_CARD_VALUE:
        raise HTTPException(
            status_code=400,
            detail=f"非法牌值: {value}。必须落在 0 到 {MAX_CARD_VALUE} 之间。",
        )

    return color, value


@router.post("/turn")
def calculate_turn(req: TurnRequest):
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

    controller = GameController(my_cards, public_cards)
    for action in req.opponent_history:
        controller.update_opponent_action(
            {
                "type": action.type,
                "target_color": action.target_color,
                "target_num": action.target_num,
                "result": action.result,
            }
        )

    result = controller.run_turn(opponent_total_cards)
    result["input_summary"] = {
        "opponent_total_cards": opponent_total_cards,
        "opponent_revealed_count": opponent_revealed_count,
        "opponent_hidden_count": result["opponent_hidden_count"],
    }
    return result
