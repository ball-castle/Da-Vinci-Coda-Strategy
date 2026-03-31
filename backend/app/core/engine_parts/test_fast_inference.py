import time
from typing import List, Dict
from functools import lru_cache

# Constants
FULL_MASK = 0b111111111111

def create_mask(available_numbers: List[int]) -> int:
    mask = 0
    for num in available_numbers:
        if 0 <= num <= 11:
            mask |= (1 << num)
    return mask

def get_numbers_from_mask(mask: int) -> List[int]:
    return [i for i in range(12) if (mask & (1 << i))]

class FastSlot:
    def __init__(self, slot_id: int, color: int, is_known: bool, mask: int):
        self.slot_id = slot_id
        self.color = color
        self.is_known = is_known
        self.mask = mask

class FastInferenceEngine:
    def __init__(self, slots: List[FastSlot], black_pool_mask: int, white_pool_mask: int):
        self.slots = slots
        self.black_pool_mask = black_pool_mask
        self.white_pool_mask = white_pool_mask
        self.valid_combinations = 0
        self.slot_successful_masks: Dict[int, int] = {slot.slot_id: 0 for slot in slots if not slot.is_known}

    @lru_cache(maxsize=100000)
    def _dfs_search(self, slot_idx: int, current_black_pool: int, current_white_pool: int, last_val: int = -1) -> int:
        if slot_idx == len(self.slots):
            return 1
            
        slot = self.slots[slot_idx]
        if slot.is_known:
            known_val = get_numbers_from_mask(slot.mask)[0]
            if known_val < last_val: 
                return 0
            return self._dfs_search(slot_idx + 1, current_black_pool, current_white_pool, known_val)

        current_pool = current_black_pool if slot.color == 0 else current_white_pool
        available_mask = slot.mask & current_pool
        if last_val >= 0:
            limit_mask = FULL_MASK & ~((1 << (last_val)) - 1)
            available_mask &= limit_mask
            
        if available_mask == 0:
            return 0
            
        total_branches = 0
        for val in get_numbers_from_mask(available_mask):
            val_bit = 1 << val
            next_black_pool = current_black_pool
            next_white_pool = current_white_pool
            if slot.color == 0:
                next_black_pool &= ~val_bit
            else:
                next_white_pool &= ~val_bit
                
            branches_from_here = self._dfs_search(slot_idx + 1, next_black_pool, next_white_pool, val)
            if branches_from_here > 0:
                total_branches += branches_from_here
                self.slot_successful_masks[slot.slot_id] |= val_bit
                
        return total_branches

    def solve(self):
        self.valid_combinations = self._dfs_search(0, self.black_pool_mask, self.white_pool_mask)
        return self.valid_combinations, self.slot_successful_masks

if __name__ == '__main__':
    print("=== 面向达芬奇密码的按位掩码深度搜索性能压测 ===")
    slots = []
    # 生成最极端的全盲状态：11张未知卡牌的排序树推断！
    for i in range(11):
        slots.append(FastSlot(slot_id=i, color=i % 2, is_known=False, mask=FULL_MASK))
        
    engine = FastInferenceEngine(slots=slots, black_pool_mask=FULL_MASK, white_pool_mask=FULL_MASK)
    start_time = time.time()
    total_valid, aggregated = engine.solve()
    end_time = time.time()
    
    print(f"找出的有效组合数: {total_valid}")
    print(f"纯位运算耗时: {end_time - start_time:.6f} 秒")
