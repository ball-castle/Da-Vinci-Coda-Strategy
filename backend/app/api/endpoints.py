"""
API端点模块

提供游戏状态计算和决策推荐的REST API接口。
主要功能：
- 接收游戏状态数据
- 执行MCTS算法计算
- 返回AI决策建议

作者：DaVinci POMDP AI Team
"""
from typing import Any, Dict, List, Optional, Tuple
import logging

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
from app.exceptions import ValidationError, GameStateError, MCTSComputationError

logger = logging.getLogger(__name__)

router = APIRouter()
session_manager = GameSessionManager()


class OpponentAction(BaseModel):
    """对手行动记录模型"""
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
    """卡槽数据模型"""
    slot_index: int
    color: Optional[str] = None
    value: Any = None
    is_revealed: bool = False
    is_newly_drawn: bool = False


class PlayerStatePayload(BaseModel):
    """玩家状态数据模型"""
    player_id: str
    slots: List[CardSlotPayload] = Field(default_factory=list)


class GuessActionPayload(BaseModel):
    """猜测行动数据模型"""
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
    """游戏状态数据模型"""
    self_player_id: str = "me"
    target_player_id: str = "opponent"
    players: List[PlayerStatePayload]
    actions: List[GuessActionPayload] = Field(default_factory=list)


class TurnRequest(BaseModel):
    """
    回合计算请求模型
    
    支持两种输入格式：
    1. 结构化状态（state）
    2. 旧版格式（my_cards, public_cards等）
    """
    session_id: Optional[str] = None
    state: Optional[GameStatePayload] = None
    my_cards: Optional[List[List[Any]]] = None
    public_cards: Optional[Dict[str, List[List[Any]]]] = None
    opponent_card_count: Optional[int] = None
    opponent_hidden_count: Optional[int] = None
    opponent_history: List[OpponentAction] = Field(default_factory=list)



def _coerce_color(color: Optional[str], field_name: str) -> Optional[str]:
    """验证并转换颜色值"""
    if color is None:
        return None
    if color not in CARD_COLORS:
        raise ValidationError(
            f"{field_name} 非法颜色: {color}",
            detail={"field": field_name, "value": color, "allowed": list(CARD_COLORS)}
        )
    return color


def _coerce_card_value(value: Any, field_name: str) -> Any:
    """验证并转换牌值"""
    if value is None:
        return None
    if value == JOKER:
        return value
    if not isinstance(value, int):
        raise ValidationError(
            f"{field_name} 非法牌值: {value}",
            detail={"field": field_name, "value": value, "type": type(value).__name__}
        )
    if not 0 <= value <= MAX_CARD_VALUE:
        raise ValidationError(
            f"{field_name} 牌值超出范围: {value}",
            detail={"field": field_name, "value": value, "min": 0, "max": MAX_CARD_VALUE}
        )
    return value


def _coerce_card(raw_card: List[Any]) -> Tuple[str, Any]:
    """验证并转换卡牌数据"""
    if len(raw_card) != 2:
        raise ValidationError(
            "卡牌格式错误",
            detail={"expected_format": "[颜色, 数值]", "received": raw_card}
        )

    color = _coerce_color(raw_card[0], "card.color")
    value = _coerce_card_value(raw_card[1], "card.value")
    if color is None:
        raise ValidationError(
            "卡牌必须提供颜色",
            detail={"card": raw_card}
        )
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
            raise ValidationError(f"重复的 player_id: {player_payload.player_id}", detail={"player_id": player_payload.player_id})

        slots: List[CardSlot] = []
        used_slot_indexes = set()
        for slot_payload in player_payload.slots:
            if slot_payload.slot_index < 0:
                raise ValidationError("slot_index 不能是负数", detail={"slot_index": slot_payload.slot_index})
            if slot_payload.slot_index in used_slot_indexes:
                raise ValidationError(f"玩家 {player_payload.player_id} 存在重复 slot_index: {slot_payload.slot_index}", detail={"player_id": player_payload.player_id, "slot_index": slot_payload.slot_index})

            color = _coerce_color(slot_payload.color, "slot.color")
            value = _coerce_card_value(slot_payload.value, "slot.value")
            if value is not None and color is None:
                raise ValidationError("当 slot.value 已知时，slot.color 也必须提供")
            if slot_payload.is_revealed and value is None:
                raise ValidationError("明牌 slot 必须提供 value")

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
        raise GameStateError("state.self_player_id 不存在于 players 中", detail={"self_player_id": payload.self_player_id})
    if payload.target_player_id not in players:
        raise GameStateError("state.target_player_id 不存在于 players 中", detail={"target_player_id": payload.target_player_id})

    actions: List[GuessAction] = []
    for action_payload in payload.actions:
        if action_payload.guesser_id not in players:
            raise GameStateError(f"未知 guesser_id: {action_payload.guesser_id}", detail={"guesser_id": action_payload.guesser_id})
        if action_payload.target_player_id not in players:
            raise GameStateError(f"未知 target_player_id: {action_payload.target_player_id}", detail={"target_player_id": action_payload.target_player_id})
        if action_payload.revealed_player_id is not None and action_payload.revealed_player_id not in players:
            raise GameStateError(f"未知 revealed_player_id: {action_payload.revealed_player_id}", detail={"revealed_player_id": action_payload.revealed_player_id})

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
        raise ValidationError("必须提供 state，或者同时提供 my_cards / public_cards / opponent_card_count")

    my_cards = [_coerce_card(card) for card in req.my_cards]
    public_cards = {
        player: [_coerce_card(card) for card in cards]
        for player, cards in req.public_cards.items()
    }

    opponent_revealed_count = len(public_cards.get("opponent", []))
    if req.opponent_hidden_count is not None:
        if req.opponent_hidden_count < 0:
            raise ValidationError("opponent_hidden_count 不能是负数", detail={"opponent_hidden_count": req.opponent_hidden_count})
        opponent_total_cards = req.opponent_hidden_count + opponent_revealed_count
    else:
        opponent_total_cards = req.opponent_card_count

    if opponent_total_cards < opponent_revealed_count:
        raise GameStateError("对手总牌数不能小于已公开的对手明牌数量", detail={"opponent_total": opponent_total_cards, "revealed": opponent_revealed_count})

    return build_legacy_game_state(
        my_cards=my_cards,
        public_cards=public_cards,
        opponent_total_cards=opponent_total_cards,
        actions=_build_legacy_actions(req.opponent_history),
    )


@router.post("/turn")
def calculate_turn(req: TurnRequest):
    """
    计算当前回合的最佳策略
    
    Args:
        req: 包含游戏状态和会话信息的请求
        
    Returns:
        包含AI建议和分析结果的响应
        
    Raises:
        ValidationError: 输入数据验证失败
        GameStateError: 游戏状态不一致
        MCTSComputationError: MCTS计算失败
    """
    try:
        # 确保会话跟踪
        session_id = session_manager.get_or_create_session(req.session_id)
        session_data = session_manager.get_session(session_id)
        
        logger.info(f"处理回合请求 - session_id: {session_id}")
        
        # 构建游戏状态
        try:
            game_state = _build_game_state(req)
        except (ValidationError, GameStateError) as e:
            logger.warning(f"游戏状态构建失败: {e.message}")
            raise
        except Exception as e:
            logger.error(f"游戏状态构建异常: {str(e)}", exc_info=True)
            raise GameStateError(f"游戏状态构建失败: {str(e)}")
        
        # 可选：将新动作持久化到会话历史
        if req.state and req.state.actions:
            session_data['history'] = req.state.actions
            logger.debug(f"更新会话历史 - 动作数: {len(req.state.actions)}")
        
        # 执行MCTS计算
        try:
            controller = GameController(game_state)
            result = controller.run_turn()
        except Exception as e:
            logger.error(f"MCTS计算失败: {str(e)}", exc_info=True)
            raise MCTSComputationError(
                "AI决策计算失败",
                detail={"error": str(e), "game_state_valid": True}
            )
        
        # 添加会话和状态摘要信息
        result["session_id"] = session_id
        result["input_summary"] = {
            "self_player_id": game_state.self_player_id,
            "target_player_id": game_state.target_player_id,
            "player_count": len(game_state.players),
            "target_total_slots": len(game_state.target_player().ordered_slots()),
            "target_hidden_count": result.get("opponent_hidden_count", 0),
            "action_count": len(game_state.actions),
        }
        
        logger.info(f"回合计算成功 - session_id: {session_id}, 决策: {result.get('action', 'unknown')}")
        return result
        
    except (ValidationError, GameStateError, MCTSComputationError):
        # 已知的业务异常，直接抛出
        raise
    except Exception as e:
        # 未知异常，记录并包装
        logger.error(f"calculate_turn 未知异常: {str(e)}", exc_info=True)
        raise MCTSComputationError(
            "处理请求时发生未知错误",
            detail={"error_type": type(e).__name__}
        )


