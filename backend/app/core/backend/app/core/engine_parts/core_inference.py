from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
from functools import lru_cache
from .bitmask_utils import FULL_MASK, create_mask, get_numbers_from_mask, bit_count

@dataclass
class FastSlot:
    \"\"\"轻量级插槽，全部用整型/位掩码存储属性\"\"\"
    slot_id: int
    color: int # 0: Black, 1: White
    is_known: bool
    mask: int # 位掩码，表示这张牌当前可能的数字范围

class FastInferenceEngine:
    def __init__(self, slots: List[FastSlot], black_pool_mask: int, white_pool_mask: int):
        self.slots = slots
        # 初始化双池
        self.black_pool_mask = black_pool_mask
        self.white_pool_mask = white_pool_mask
        
        # 结果聚合器
        self.valid_combinations = 0
        self.slot_successful_masks: Dict[int, int] = {slot.slot_id: 0 for slot in slots if not slot.is_known}

    @lru_cache(maxsize=100000)
    def _dfs_search(self, slot_idx: int, current_black_pool: int, current_white_pool: int, last_val: int = -1) -> int:
        \"\"\"
        高度优化的位运算 DFS，避免传递一切巨大的引用类型。
        返回从该节点开始成功到底的组合枝叶数。
        \"\"\"
        if slot_idx == len(self.slots):
            return 1 # 找到一条合法解
            
        slot = self.slots[slot_idx]
        
        # 对于已经揭开身份的牌，它只起推进指针和验证序列上升的作用
        if slot.is_known:
            known_val = get_numbers_from_mask(slot.mask)[0]
            # 规则：必须严格大于等于（如果加入等于的情况要考虑颜色层级，简单起见我们设计基础上升要求）
            if known_val < last_val: 
                return 0
            return self._dfs_search(slot_idx + 1, current_black_pool, current_white_pool, known_val)

        # 接下来处理未揭开的暗牌
        # 取交集：该槽位自身的硬约束 mask AND 当前全场剩余池的 mask
        current_pool = current_black_pool if slot.color == 0 else current_white_pool
        available_mask = slot.mask & current_pool
        
        # 左侧压迫截断：只能挑比上一张牌 (last_val) 大的数字（这步直接把 last_val 的低位全部强行抹零）
        if last_val >= 0:
            # 例如 last_val 是 3，那我们需要保留 4~11，即屏蔽 0,1,2,3
            # 即通过掩码左移再保留高位
            limit_mask = FULL_MASK & ~((1 << (last_val)) - 1)
            available_mask &= limit_mask

        # 如果无路可走，剪枝死胡同
        if available_mask == 0:
            return 0
            
        total_branches = 0
        
        # 位遍历：提取所有可用选项并进行子树派生
        # (可以通过位运算快速遍历，这里简单用刚才的工具函数)
        for val in get_numbers_from_mask(available_mask):
            val_bit = 1 << val
            next_black_pool = current_black_pool
            next_white_pool = current_white_pool
            
            # 从池子中抹除该张牌 (异或清除)
            if slot.color == 0:
                next_black_pool &= ~val_bit
            else:
                next_white_pool &= ~val_bit
                
            branches_from_here = self._dfs_search(slot_idx + 1, next_black_pool, next_white_pool, val)
            
            if branches_from_here > 0:
                total_branches += branches_from_here
                # 记录成功推演过的数值叠加到最终的可用域里
                self.slot_successful_masks[slot.slot_id] |= val_bit
                
        return total_branches

    def solve(self):
        # 清空内部数据发起最源头 DFS
        self.valid_combinations = self._dfs_search(0, self.black_pool_mask, self.white_pool_mask)
        return self.valid_combinations, self.slot_successful_masks
