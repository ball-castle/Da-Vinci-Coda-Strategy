"""
错误处理中间件和异常处理器
提供统一的错误响应格式和日志记录
"""
import logging
import traceback
from typing import Union

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.exceptions import DaVinciException

# 配置日志
logger = logging.getLogger(__name__)


def create_error_response(
    status_code: int,
    message: str,
    detail: Union[str, dict, list, None] = None,
    error_type: str = "error"
) -> JSONResponse:
    """
    创建统一格式的错误响应
    
    Args:
        status_code: HTTP状态码
        message: 错误消息
        detail: 详细错误信息
        error_type: 错误类型
    
    Returns:
        JSONResponse对象
    """
    content = {
        "success": False,
        "error": {
            "type": error_type,
            "message": message,
        }
    }
    
    if detail is not None:
        content["error"]["detail"] = detail
    
    return JSONResponse(
        status_code=status_code,
        content=content
    )


async def davinci_exception_handler(request: Request, exc: DaVinciException) -> JSONResponse:
    """
    处理自定义DaVinci异常
    """
    logger.error(
        f"DaVinci异常: {exc.message}",
        extra={
            "status_code": exc.status_code,
            "detail": exc.detail,
            "path": request.url.path
        }
    )
    
    return create_error_response(
        status_code=exc.status_code,
        message=exc.message,
        detail=exc.detail,
        error_type=exc.__class__.__name__
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    处理HTTP异常
    """
    logger.warning(
        f"HTTP异常: {exc.detail}",
        extra={
            "status_code": exc.status_code,
            "path": request.url.path
        }
    )
    
    return create_error_response(
        status_code=exc.status_code,
        message=str(exc.detail),
        error_type="HTTPException"
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    处理请求验证异常（Pydantic）
    """
    logger.warning(
        "请求验证失败",
        extra={
            "errors": exc.errors(),
            "path": request.url.path
        }
    )
    
    # 格式化验证错误
    formatted_errors = []
    for error in exc.errors():
        formatted_errors.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        message="请求数据验证失败",
        detail=formatted_errors,
        error_type="ValidationError"
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    处理所有未捕获的异常
    """
    # 记录完整的错误栈
    logger.error(
        f"未处理的异常: {str(exc)}",
        extra={
            "path": request.url.path,
            "traceback": traceback.format_exc()
        },
        exc_info=True
    )
    
    # 在生产环境中隐藏详细错误信息
    from app.config import settings
    
    if settings.debug:
        detail = {
            "exception_type": exc.__class__.__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc().split("\n")
        }
    else:
        detail = None
    
    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        message="服务器内部错误，请稍后重试",
        detail=detail,
        error_type="InternalServerError"
    )


def setup_exception_handlers(app):
    """
    为FastAPI应用设置所有异常处理器
    
    Args:
        app: FastAPI应用实例
    """
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException
    
    # 自定义异常
    app.add_exception_handler(DaVinciException, davinci_exception_handler)
    
    # HTTP异常
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    
    # 验证异常
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    
    # 通用异常（捕获所有其他异常）
    app.add_exception_handler(Exception, general_exception_handler)
    
    logger.info("异常处理器已配置")
