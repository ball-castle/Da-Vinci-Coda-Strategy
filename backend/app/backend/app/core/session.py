from uuid import uuid4
from typing import Dict, Any, List

class GameSessionManager:
    def __init__(self):
        self.sessions = {}

    def get_or_create(self, game_id: str = None) -> str:
        if not game_id or game_id not in self.sessions:
            game_id = str(uuid4())
            self.sessions[game_id] = {'history': [], 'player_models': {}}
        return game_id

    def update_session(self, game_id: str, action: Dict[str, Any]):
        if game_id in self.sessions:
            self.sessions[game_id]['history'].append(action)

session_manager = GameSessionManager()
