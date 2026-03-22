from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple, Optional
from app.core.engine import GameController

router = APIRouter()

class OpponentAction(BaseModel):
    type: str # 'guess'
    target_color: str # 'B' or 'W'
    target_num: Any # int or '-'
    result: bool

class TurnRequest(BaseModel):
    my_cards: List[List[Any]] # e.g. [['B', 2], ['W', 5]]
    public_cards: Dict[str, List[List[Any]]]
    opponent_card_count: int
    opponent_history: Optional[List[OpponentAction]] = []

@router.post("/turn")
def calculate_turn(req: TurnRequest):
    # Convert incoming arrays back to tuples for the engine
    my_cards = [(c[0], c[1]) for c in req.my_cards]
    public_cards = {k: [(c[0], c[1]) for c in v] for k, v in req.public_cards.items()}
    
    controller = GameController(my_cards, public_cards)
    
    # Load opponent history into the controller
    if req.opponent_history:
        for action in req.opponent_history:
            controller.update_opponent_action({
                'type': action.type,
                'target_color': action.target_color,
                'target_num': action.target_num,
                'result': action.result
            })
            
    best_move, search_space_size = controller.run_turn(req.opponent_card_count)
    
    return {
        "best_move": best_move,
        "search_space_size": search_space_size
    }
