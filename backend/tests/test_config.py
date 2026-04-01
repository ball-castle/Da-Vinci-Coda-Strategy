"""
配置模块测试
"""
import pytest
from app.config import get_settings, Settings


class TestSettings:
    """配置类测试"""
    
    def test_settings_singleton(self):
        """测试配置单例模式"""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
    
    def test_default_values(self):
        """测试默认配置值"""
        settings = get_settings()
        assert settings.app_name == "Da Vinci Coda API"
        assert settings.app_version == "1.0.0"
        assert settings.port == 8000
        assert settings.mcts_iterations == 1000
        assert settings.mcts_exploration_constant == 1.414
    
    def test_cors_origins_is_list(self):
        """测试CORS配置是列表类型"""
        settings = get_settings()
        assert isinstance(settings.cors_origins, list)
        assert len(settings.cors_origins) > 0
