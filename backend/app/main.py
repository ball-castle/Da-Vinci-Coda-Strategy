from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import endpoints
from app.config import settings
from app.middleware import setup_exception_handlers
from app.logger import setup_logging, get_logger

# 配置日志系统
setup_logging(
    log_level=settings.log_level,
    enable_console=True,
    enable_file=False  # 可在需要时启用文件日志
)
logger = get_logger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug
)

# 设置异常处理器
setup_exception_handlers(app)

# Setup CORS using configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_credentials,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)

app.include_router(endpoints.router, prefix=settings.api_prefix)

@app.get("/")
def root():
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "version": settings.app_version
    }

logger.info(f"{settings.app_name} v{settings.app_version} 启动成功")
