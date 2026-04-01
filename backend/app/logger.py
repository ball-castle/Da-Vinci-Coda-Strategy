"""
日志系统模块

提供结构化日志记录功能，支持不同日志级别和格式化输出
"""
import logging
import sys
from typing import Optional
from datetime import datetime
from pathlib import Path

from app.config import settings


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器（用于终端输出）"""
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # 添加颜色
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = False
):
    """
    配置日志系统
    
    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径
        enable_console: 是否启用控制台输出
        enable_file: 是否启用文件输出
    """
    # 使用配置中的日志级别
    level = log_level or settings.log_level
    level_num = getattr(logging, level.upper(), logging.INFO)
    
    # 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level_num)
    
    # 清除现有处理器
    root_logger.handlers.clear()
    
    # 控制台处理器（带颜色）
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level_num)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # 文件处理器
    if enable_file and log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level_num)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # 记录日志系统启动
    logger = logging.getLogger(__name__)
    logger.info(f"日志系统已初始化 - 级别: {level}")
    if enable_file and log_file:
        logger.info(f"日志文件: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    获取命名日志记录器
    
    Args:
        name: 日志记录器名称（通常使用 __name__）
        
    Returns:
        配置好的日志记录器
    """
    return logging.getLogger(name)


class RequestLogger:
    """请求日志记录器（用于FastAPI中间件）"""
    
    def __init__(self):
        self.logger = get_logger("app.requests")
    
    def log_request(
        self,
        method: str,
        path: str,
        client_host: str,
        status_code: Optional[int] = None,
        duration: Optional[float] = None
    ):
        """记录HTTP请求"""
        msg = f"{method} {path} - Client: {client_host}"
        
        if status_code is not None:
            msg += f" - Status: {status_code}"
        
        if duration is not None:
            msg += f" - Duration: {duration:.3f}s"
        
        # 根据状态码选择日志级别
        if status_code is None:
            self.logger.info(msg)
        elif status_code < 400:
            self.logger.info(msg)
        elif status_code < 500:
            self.logger.warning(msg)
        else:
            self.logger.error(msg)


class MCTSLogger:
    """MCTS计算日志记录器"""
    
    def __init__(self):
        self.logger = get_logger("app.mcts")
    
    def log_computation(
        self,
        session_id: str,
        iterations: int,
        duration: float,
        result: str
    ):
        """记录MCTS计算过程"""
        self.logger.info(
            f"MCTS计算完成 - Session: {session_id}, "
            f"迭代: {iterations}, 耗时: {duration:.3f}s, 结果: {result}"
        )
    
    def log_error(self, session_id: str, error: str):
        """记录MCTS计算错误"""
        self.logger.error(
            f"MCTS计算失败 - Session: {session_id}, 错误: {error}"
        )


# 全局日志记录器实例
request_logger = RequestLogger()
mcts_logger = MCTSLogger()
