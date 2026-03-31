import sys
import os

# 将 backend 路径加入 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from app.core.engine import GameController, HiddenPosition
import time

def test_original_engine_speed():
    # 构造一个稍微复杂点的初始情况
    controller = GameController()
    
    # 初始化三个玩家
    controller.add_player('P1')
    controller.add_player('P2')
    controller.add_player('P3')
    
    # 模拟发牌 (给 P1 一些未知牌, 这里只提供给引擎做结构推演，实际内部自己生成 slot)
    
    # 手动制造一个需要巨大搜索树的盲猜场面来测试原本的 engine.py
    # 注意：直接调用原 engine.py 可能极其费时。我们仅用来做小样本横向比对。
    print('Ready to integration testing baseline...')
    pass

if __name__ == '__main__':
    test_original_engine_speed()
