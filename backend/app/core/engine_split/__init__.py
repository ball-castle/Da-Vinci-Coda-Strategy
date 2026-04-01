"""
Engine模块 - 拆分版本

将原 engine.py (15,796行) 拆分为7个模块，提高可维护性。
所有逻辑完全保持不变。
"""
from .models import (
    CARD_COLOR_RANK,
    SOFT_BEHAVIOR_BLEND,
    HiddenPosition,
    HardConstraintSet,
    GuessSignal,
    SearchTreeNode,
    ProbabilityMatrix,
    FullProbabilityMatrix,
    SlotKey,
)
from .utils import (
    numeric_card_value,
    card_sort_key,
    serialize_card,
    normalize_card_distribution,
    slot_key,
    clamp,
)
from .constraints import HardConstraintCompiler
from .behavior import BehavioralLikelihoodModel
from .inference import DaVinciInferenceEngine
from .decision import DaVinciDecisionEngine
from .controller import GameController

__all__ = [
    "CARD_COLOR_RANK", "SOFT_BEHAVIOR_BLEND",
    "ProbabilityMatrix", "FullProbabilityMatrix", "SlotKey",
    "HiddenPosition", "HardConstraintSet", "GuessSignal", "SearchTreeNode",
    "numeric_card_value", "card_sort_key", "serialize_card",
    "normalize_card_distribution", "slot_key", "clamp",
    "HardConstraintCompiler", "BehavioralLikelihoodModel",
    "DaVinciInferenceEngine", "DaVinciDecisionEngine", "GameController",
]
