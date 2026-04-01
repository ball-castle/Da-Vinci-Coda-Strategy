# 🚀 MCTS性能优化建议

## 当前状况分析

经过代码审查，MCTS引擎实现在 `backend/app/core/engine.py` 中，包含：
- 概率矩阵计算
- 约束求解
- 树搜索和模拟
- 后验概率更新

## 🎯 优化建议

### 1. **添加结果缓存** (高优先级)

**问题**: 相同游戏状态会重复计算

**解决方案**: 使用LRU缓存

```python
# backend/app/core/cache.py
from functools import lru_cache
from typing import Tuple
import hashlib
import json

def hash_game_state(state: GameState) -> str:
    """生成游戏状态的哈希值用于缓存"""
    state_dict = {
        'players': {
            pid: [
                (slot.color, slot.value, slot.is_revealed)
                for slot in player.ordered_slots()
            ]
            for pid, player in state.players.items()
        },
        'actions': [(a.guesser_id, a.target_player_id, a.guessed_value, a.result) for a in state.actions],
    }
    state_str = json.dumps(state_dict, sort_keys=True)
    return hashlib.md5(state_str.encode()).hexdigest()

@lru_cache(maxsize=1000)
def cached_mcts_result(state_hash: str, iterations: int) -> dict:
    """缓存MCTS计算结果"""
    # 实际的MCTS计算会被包装在这里
    pass
```

### 2. **并行化MCTS模拟** (高优先级)

**问题**: MCTS模拟是CPU密集型，单线程执行慢

**解决方案**: 使用多进程或异步处理

```python
# backend/app/core/parallel_mcts.py
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
import multiprocessing

def run_simulation_batch(state: GameState, num_simulations: int, num_workers: int = None) -> List[SimulationResult]:
    """并行运行MCTS模拟"""
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    batch_size = num_simulations // num_workers
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            future = executor.submit(run_simulations, state, batch_size)
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            results.extend(future.result())
        
        return results
```

### 3. **优化概率计算** (中优先级)

**问题**: 概率矩阵计算可能有冗余

**解决方案**: 增量更新而非全量重算

```python
def incremental_probability_update(
    previous_probs: ProbabilityMatrix,
    new_action: GuessAction
) -> ProbabilityMatrix:
    """基于新动作增量更新概率而非全量重算"""
    updated = previous_probs.copy()
    
    # 只更新受影响的概率
    affected_slots = get_affected_slots(new_action)
    for slot in affected_slots:
        updated[slot] = recalculate_slot_probability(updated, slot, new_action)
    
    return updated
```

### 4. **添加计算超时保护** (高优先级)

**问题**: 某些复杂局面可能导致计算超时

**解决方案**: 已在config中配置，需在engine中实施

```python
# backend/app/core/engine.py
import time
from app.config import settings

class GameController:
    def run_turn(self):
        start_time = time.time()
        timeout = settings.mcts_timeout_seconds
        
        while not self.is_complete():
            if time.time() - start_time > timeout:
                logger.warning(f"MCTS计算超时 ({timeout}s)，返回当前最佳结果")
                return self.get_best_partial_result()
            
            self.run_iteration()
        
        return self.get_result()
```

### 5. **内存优化** (中优先级)

**问题**: 大型搜索树可能占用大量内存

**解决方案**: 
- 限制树深度
- 定期清理不太可能访问的节点
- 使用轻量级数据结构

```python
class MCTSNode:
    __slots__ = ['state', 'parent', 'children', 'visits', 'value']  # 减少内存占用
    
    def prune_unlikely_children(self, threshold: float = 0.01):
        """剪枝访问次数低的子节点"""
        if not self.children:
            return
        
        total_visits = sum(child.visits for child in self.children)
        self.children = [
            child for child in self.children
            if child.visits / total_visits > threshold
        ]
```

### 6. **算法参数优化** (低优先级)

**当前配置** (在.env中):
```bash
MCTS_ITERATIONS=1000
MCTS_EXPLORATION_CONSTANT=1.414
MCTS_SIMULATION_DEPTH=50
MCTS_TIMEOUT_SECONDS=5.0
```

**调优建议**:
```bash
# 快速模式（实时响应）
MCTS_ITERATIONS=500
MCTS_TIMEOUT_SECONDS=2.0

# 平衡模式（默认）
MCTS_ITERATIONS=1000
MCTS_TIMEOUT_SECONDS=5.0

# 深度分析模式
MCTS_ITERATIONS=5000
MCTS_TIMEOUT_SECONDS=15.0
```

### 7. **添加性能监控** (中优先级)

```python
# backend/app/core/profiling.py
import time
from functools import wraps
from app.logger import get_logger

logger = get_logger(__name__)

def profile_mcts(func):
    """MCTS性能分析装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        
        logger.info(
            f"MCTS {func.__name__} - "
            f"Duration: {duration:.3f}s, "
            f"Iterations: {kwargs.get('iterations', 'N/A')}"
        )
        
        return result
    return wrapper
```

## 📊 预期性能提升

| 优化 | 预期提升 | 实施难度 |
|------|---------|---------|
| 结果缓存 | 50-80% | 低 |
| 并行化 | 2-4x | 中 |
| 增量更新 | 20-30% | 中 |
| 超时保护 | 100% (稳定性) | 低 |
| 内存优化 | 30-50% | 中 |
| 参数调优 | 10-20% | 低 |

## 🔧 实施优先级

### 立即实施 (1周内)
1. ✅ 添加结果缓存
2. ✅ 实施超时保护
3. ✅ 添加性能监控

### 短期实施 (2-4周)
4. 并行化MCTS模拟
5. 优化概率计算

### 长期优化 (1-3个月)
6. 内存优化和剪枝策略
7. 算法参数自适应调优

## ⚠️ 注意事项

1. **并行化权衡**: 多进程会增加内存占用，需根据服务器资源调整
2. **缓存失效**: 游戏规则变化时需清理缓存
3. **精度vs速度**: 减少迭代次数会降低AI质量，需权衡
4. **生产部署**: 建议使用Redis做分布式缓存

## 📈 性能测试建议

```python
# backend/tests/test_performance.py
import pytest
import time
from app.core.engine import GameController

def test_mcts_performance():
    """MCTS性能基准测试"""
    controller = GameController(test_game_state)
    
    start = time.time()
    result = controller.run_turn()
    duration = time.time() - start
    
    # 性能要求
    assert duration < 5.0, f"MCTS计算超时: {duration}s"
    assert result is not None
    
    print(f"MCTS Performance: {duration:.3f}s")
```

## 🎯 总结

MCTS性能优化是一个渐进的过程。建议：
1. 先实施低成本高回报的优化（缓存、超时）
2. 监控生产环境性能指标
3. 根据实际使用情况调整参数
4. 必要时考虑算法级别的优化

当前代码已经有较好的结构，主要缺少：
- ✅ 缓存机制
- ✅ 并行计算
- ✅ 超时保护
- ✅ 性能监控

这些都是可以逐步添加的改进点。
