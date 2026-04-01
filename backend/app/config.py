"""
配置管理模块
提供统一的配置访问接口，支持环境变量和默认值
"""
import os
from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置类"""
    
    # 应用基本信息
    app_name: str = "Da Vinci Coda API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # 服务器配置
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False
    
    # CORS配置
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    cors_credentials: bool = True
    cors_methods: list[str] = ["*"]
    cors_headers: list[str] = ["*"]
    
    # MCTS算法配置
    mcts_iterations: int = 1000
    mcts_exploration_constant: float = 1.414
    mcts_simulation_depth: int = 50
    mcts_timeout_seconds: float = 5.0
    
    # API配置
    api_prefix: str = "/api"
    api_timeout: int = 30
    
    # 日志配置
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 数据库配置（如需要）
    database_url: Optional[str] = None
    
    # 语音服务配置（讯飞等）
    voice_api_url: Optional[str] = None
    voice_api_key: Optional[str] = None
    voice_api_secret: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    获取配置单例
    使用lru_cache确保配置只被加载一次
    """
    return Settings()


# 便捷访问
settings = get_settings()
