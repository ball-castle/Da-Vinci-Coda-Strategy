"""
兼容旧导入路径的引擎导出层。

历史测试和脚本仍然会从 `app.core.engine` 导入类，这里统一转发到
当前拆分后的 `engine_split` 实现，避免导入路径漂移导致测试收集失败。
"""

from .engine_split import (
    BehavioralLikelihoodModel,
    DaVinciDecisionEngine,
    DaVinciInferenceEngine,
    GameController,
)

__all__ = [
    "BehavioralLikelihoodModel",
    "DaVinciDecisionEngine",
    "DaVinciInferenceEngine",
    "GameController",
]
