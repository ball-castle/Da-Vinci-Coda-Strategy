"""
pytest配置文件

定义全局fixtures和测试配置
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.state import GameState, PlayerState, CardSlot


@pytest.fixture
def client():
    """FastAPI测试客户端"""
    return TestClient(app)


@pytest.fixture
def sample_game_state():
    """示例游戏状态"""
    return GameState(
        self_player_id="me",
        target_player_id="opponent",
        players={
            "me": PlayerState(
                player_id="me",
                slots=[
                    CardSlot(slot_index=0, color="B", value=3, is_revealed=True),
                    CardSlot(slot_index=1, color="W", value=7, is_revealed=True),
                ]
            ),
            "opponent": PlayerState(
                player_id="opponent",
                slots=[
                    CardSlot(slot_index=0, color="B", value=5, is_revealed=True),
                    CardSlot(slot_index=1, color=None, value=None, is_revealed=False),
                    CardSlot(slot_index=2, color=None, value=None, is_revealed=False),
                ]
            )
        },
        actions=[]
    )


@pytest.fixture
def sample_turn_request():
    """示例回合请求"""
    return {
        "my_cards": [["B", 3], ["W", 7]],
        "public_cards": {
            "opponent": [["B", 5]]
        },
        "opponent_card_count": 3,
        "opponent_history": []
    }
