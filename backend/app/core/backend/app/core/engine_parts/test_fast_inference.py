from .bitmask_utils import create_mask, get_numbers_from_mask, FULL_MASK
from .core_inference import FastSlot, FastInferenceEngine
import time

def run_performance_test():
    print("=== 开始测试 位掩码版推理引擎 ===")
    
    # 模拟一个非常深的中盘情况：
    # 比如我们有一串黑白相间的牌，总共有 10 张暗牌，我们要推演它们的全部合法排列！
    # 如果用你原来的列表深拷贝 DFS，这种完全开放的盲搜大约要跑好几秒到一分钟。
    slots = []
    
    # 生成 5 黑 5 白，全盲槽位，且都没有被缩小范围
    for i in range(10):
        color = i % 2
        slots.append(FastSlot(
            slot_id=i,
            color=color,
            is_known=False,
            mask=FULL_MASK
        ))
        
    engine = FastInferenceEngine(
        slots=slots,
        black_pool_mask=FULL_MASK,
        white_pool_mask=FULL_MASK
    )
    
    start_time = time.time()
    total_valid, aggregated_masks = engine.solve()
    end_time = time.time()
    
    print(f"总计找到合法深度分支数: {total_valid}")
    print(f"求解耗时: {end_time - start_time:.6f} 秒")
    
    for slot_id, mask in aggregated_masks.items():
        print(f"槽位 {slot_id} 推演得出的有效数字: {get_numbers_from_mask(mask)}")

if __name__ == '__main__':
    run_performance_test()
