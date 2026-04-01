"""
自定义异常类
定义业务逻辑相关的异常，便于统一处理
"""
from typing import Any, Optional


class DaVinciException(Exception):
    """
    DaVinci应用基础异常类
    所有自定义异常都应继承此类
    """
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        detail: Optional[Any] = None
    ):
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(message)


class ValidationError(DaVinciException):
    """数据验证错误"""
    def __init__(self, message: str, detail: Optional[Any] = None):
        super().__init__(message, status_code=400, detail=detail)


class GameStateError(DaVinciException):
    """游戏状态错误"""
    def __init__(self, message: str, detail: Optional[Any] = None):
        super().__init__(message, status_code=422, detail=detail)


class SessionNotFoundError(DaVinciException):
    """会话未找到"""
    def __init__(self, session_id: str):
        super().__init__(
            f"会话未找到: {session_id}",
            status_code=404,
            detail={"session_id": session_id}
        )


class MCTSComputationError(DaVinciException):
    """MCTS计算错误"""
    def __init__(self, message: str, detail: Optional[Any] = None):
        super().__init__(message, status_code=500, detail=detail)


class ConfigurationError(DaVinciException):
    """配置错误"""
    def __init__(self, message: str, detail: Optional[Any] = None):
        super().__init__(message, status_code=500, detail=detail)
