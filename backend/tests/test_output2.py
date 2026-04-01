import json
from app.core.engine import GameController
from app.models.domain import GameState, PlayerState, Card, GuessAction
from app.common.enums import Color, CardValue
state = GameState(self_player_id='p1', target_player_id='p2', players={'p1': PlayerState(player_id='p1', slots=[Card(color=Color.BLACK, value=CardValue.V1), Card(color=Color.WHITE, value=CardValue.V2)]), 'p2': PlayerState(player_id='p2', slots=[Card(color=Color.BLACK, value=CardValue.UNKNOWN), Card(color=Color.WHITE, value=CardValue.UNKNOWN)])}, actions=[])
controller = GameController(state)
result = controller.run_turn()
print(json.dumps(result['best_move'], indent=2))

