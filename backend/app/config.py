"""
配置管理模块
提供统一的配置访问接口，支持环境变量和默认值
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

_BOOLEAN_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_BOOLEAN_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}


def _is_parseable_bool(raw_value: str) -> bool:
    normalized = raw_value.strip().lower()
    return normalized in _BOOLEAN_TRUE_VALUES or normalized in _BOOLEAN_FALSE_VALUES


@contextmanager
def _ignore_invalid_process_debug_env() -> Iterator[None]:
    """
    Ignore unrelated process-level DEBUG values such as `release`.

    Many shells and toolchains set a global DEBUG environment variable for their
    own purposes. If it is not a boolean, pydantic-settings will fail before the
    app can even start. We only suppress the process variable when it is clearly
    not parseable as a boolean, while keeping `.env` support intact.
    """

    original_debug = os.environ.get("DEBUG")
    should_temporarily_remove = (
        original_debug is not None and not _is_parseable_bool(original_debug)
    )

    if not should_temporarily_remove:
        yield
        return

    os.environ.pop("DEBUG", None)
    try:
        yield
    finally:
        os.environ["DEBUG"] = original_debug


class Settings(BaseSettings):
    """应用配置类"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

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


@lru_cache()
def get_settings() -> Settings:
    """
    获取配置单例
    使用lru_cache确保配置只被加载一次
    """
    with _ignore_invalid_process_debug_env():
        return Settings()


# 便捷访问
settings = get_settings()
