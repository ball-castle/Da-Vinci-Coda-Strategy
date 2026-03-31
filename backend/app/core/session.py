from uuid import uuid4
from typing import Dict, Any, List

class GameSessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def get_or_create_session(self, session_id: str = None) -> str:
        if not session_id or session_id not in self.sessions:
            session_id = str(uuid4())
            self.sessions[session_id] = {'history': [], 'player_models': {}}
        return session_id

    def get_session(self, session_id: str) -> Dict[str, Any]:
        return self.sessions.get(session_id, {})

session_manager = GameSessionManager()
