import asyncio
import sys
import os

# 把后端根目录加进 sys.path 防止导入模块报错
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.engine import DaVinciDecisionEngine

class PlayerAgent:
    def __init__(self, name: str, is_smart: bool = True):
        self.name = name
        self.is_smart = bool(is_smart)
        # 如果是聪明机器人，则加载我们的高级决策引擎
        self.engine = DaVinciDecisionEngine() if self.is_smart else None

    def get_action(self, game_state: dict):
        if not self.is_smart:
            # 随机机器人的占位降级逻辑 (Random Guesser)
            # 在实际业务中这里可以引入随机生成数字探测
            return {"action_type": "guess", "target": "opponent", "slot": 0, "guess": 5}
        
        # 将真实盘面推给引擎进行高级决策
        # 注意: 这里的 game_state 格式需要与你之前发给 API 的 Payload 格式一致
        return self.engine.analyze_turn(game_state)

def simulate_game(player1: PlayerAgent, player2: PlayerAgent):
    """
    模拟一局完整的达芬奇密码对战。
    (目前为 Arena 发动机与发牌器脚手架。你需要在这里实现黑白随机发牌与判断倒下的逻辑)
    返回获胜者的名字或者 'Draw'。
    """
    turn_count = 0
    max_turns = 100
    
    # 设想的系统：发牌池、每人的物理暗牌、暴露牌。
    # 模拟主循环
    while turn_count < max_turns:
        # TODO: 
        # 1. 向 Player 1 放送当前视觉可见盘面
        # 2. Player 1 生成一个决策 (Guess)
        # 3. Server 解析是否命中，公布翻牌，并把动作放入全局 history
        # 4. 判断是否全倒
        # 5. 换 Player 2 走
        
        turn_count += 1
    
    # 作为掩码测试，当前假定 Player1 常胜
    return player1.name

def run_arena(num_games: int = 100):
    print(f"=== [Da Vinci] 算法评估竞技场 (AI Arena) ===")
    print(f"正在构建虚拟博弈环境，对战总局数: {num_games} 局...")
    
    # 这里定义比赛的双方：可以是一个跑最新版代码的AI，对抗旧代码AI或者暴力随机AI
    bot_v1 = PlayerAgent("Alpha DaVinci(MCTS+Bitmask)", is_smart=True)
    bot_v2 = PlayerAgent("Random Weak Bot", is_smart=False)
    
    score_board = {bot_v1.name: 0, bot_v2.name: 0, "Draw": 0}
    
    print("开始战斗...")
    for i in range(num_games):
        winner = simulate_game(bot_v1, bot_v2)
        if winner in score_board:
            score_board[winner] += 1
            
        if (i+1) % 10 == 0:
            print(f"已完成 {i+1} 局 / 共 {num_games} 局")
            
    print("\n=================")
    print("🥊 最终战报 🥊")
    print("=================")
    for name, wins in score_board.items():
        win_rate = (wins / num_games) * 100
        print(f" {name:<30} || 胜场: {wins:<4} || 胜率: {win_rate:.1f}%")

if __name__ == "__main__":
    run_arena(100)
