import itertools
from collections import defaultdict

class DaVinciInferenceEngine:
    """推理引擎：基于 DFS 和规则剪枝，生成合法状态空间"""
    def __init__(self, my_cards, public_cards):
        self.my_cards = my_cards          # 我的手牌 e.g., [('B', 2), ('W', 5)]
        self.public_cards = public_cards  # 场上已知的明牌 (字典格式，记录各个玩家的明牌)
        self.all_possible_cards = [(c, n) for c in ['B', 'W'] for n in range(12)] + [('B', '-'), ('W', '-')]
        
    def get_available_cards(self):
        """排除已知卡牌（我的牌 + 所有人的明牌），获取剩余的未知卡牌池"""
        known_cards = set(self.my_cards)
        for player, cards in self.public_cards.items():
            known_cards.update(cards)
        return [card for card in self.all_possible_cards if card not in known_cards]

    def validate_sequence(self, sequence):
        """
        DFS 剪枝条件：验证卡牌序列是否严格满足游戏排序规则。
        【优化点】：完整补全 Joker ('-') 的处理逻辑。
        Joker 可以放在任何位置，所以我们在校验绝对大小规则时，只需要确保序列中【非 Joker】的卡牌满足递增规则即可。
        """
        # 提取出所有非 Joker 的牌及其相对原始位置（虽然这里只校验相对顺序，不需要原索引，但为了严谨直接提取子集）
        non_jokers = [card for card in sequence if card[1] != '-']
        
        for i in range(1, len(non_jokers)):
            prev, curr = non_jokers[i-1], non_jokers[i]
            # 规则 1：数字必须递增
            if curr[1] < prev[1]:
                return False
            # 规则 2：数字相同时，黑色必须在白色左边
            if curr[1] == prev[1] and not (prev[0] == 'B' and curr[0] == 'W'):
                return False
        return True

    def generate_valid_hypotheses(self, opponent_card_count):
        """DFS 生成对手可能的合法手牌组合"""
        available = self.get_available_cards()
        valid_hypotheses = []
        
        for combo in itertools.combinations(available, opponent_card_count):
            # 对组合进行初始的合法性排序：数字从小到大，百搭牌暂定放最后，同数字黑在白前
            # 真实游戏中，如果对手有 Joker，其位置是固定的，所以对于包含 Joker 的组合，实际上会衍生出多个不同的排列
            # 为了简化当前的搜索空间，我们假设对手在拿牌时，Joker 的位置会被合理放置以符合非 Joker 牌的顺序
            sorted_combo = sorted(combo, key=lambda x: (x[1] if x[1] != '-' else 99, 0 if x[0] == 'B' else 1))
            
            if self.validate_sequence(sorted_combo):
                valid_hypotheses.append(tuple(sorted_combo))
                
        return valid_hypotheses


class PsychologicalFilter:
    """心理战与概率过滤器：基于贝叶斯更新和正向推理调整权重"""
    def __init__(self):
        # 设定惩罚和奖励系数
        self.PENALTY_SELF_ABSENCE = 0.1  # 对手猜过的牌，自己大概率没有（极低权重）
        self.BOOST_ANCHOR = 1.2          # 对手猜过的牌的相邻数字，自己可能有（锚点效应提升权重）

    def apply_positive_inference(self, hypotheses, opponent_history):
        """
        正向推理：利用对手的历史行为，重新分配各个假设组合的后验概率权重。
        opponent_history 格式示例: [{'type': 'guess', 'target_color': 'B', 'target_num': 5, 'result': False}, ...]
        """
        weighted_hypotheses = []
        
        for hypothesis in hypotheses:
            weight = 1.0
            for card in hypothesis:
                if card[1] == '-': # 暂时跳过对 Joker 的心理战推断
                    continue
                    
                # 应用贝叶斯权重更新
                if self._guessed_this_card(card, opponent_history):
                    # 心理学逻辑：玩家极少去猜测自己手里已经有的精确卡牌
                    weight *= self.PENALTY_SELF_ABSENCE
                
                if self._guessed_adjacent_card(card, opponent_history):
                    # 心理学逻辑：玩家有时会利用自己手里的牌作为“锚点”去推测相邻的牌
                    weight *= self.BOOST_ANCHOR
                    
            weighted_hypotheses.append((hypothesis, weight))
            
        return weighted_hypotheses

    def _guessed_this_card(self, card, opponent_history):
        """检查对手是否曾经精确猜测过这张牌"""
        for action in opponent_history:
            if action.get('type') == 'guess' and action.get('target_color') == card[0] and action.get('target_num') == card[1]:
                return True
        return False
        
    def _guessed_adjacent_card(self, card, opponent_history):
        """检查对手是否曾经猜测过与这张牌数字相邻的牌"""
        for action in opponent_history:
            if action.get('type') == 'guess':
                guessed_num = action.get('target_num')
                if guessed_num != '-' and abs(card[1] - guessed_num) == 1:
                    return True
        return False


class DaVinciDecisionEngine:
    """决策引擎：基于期望值 (EV) 选择最佳行动"""
    def __init__(self):
        self.BASE_REWARD = 10.0 # 猜中的基础正收益

    def calculate_ev(self, probability, risk_factor):
        """
        计算特定猜测的期望值。
        probability: 猜中该卡牌的后验概率
        risk_factor: 当前猜错的代价（己方未翻开牌越少，代价越大）
        """
        reward = probability * self.BASE_REWARD 
        penalty = (1.0 - probability) * risk_factor 
        return reward - penalty

    def get_best_move(self, weighted_hypotheses, my_hidden_count):
        """整合概率，输出最高 EV 的决策"""
        if not weighted_hypotheses:
            return None

        # 1. 归一化总权重并计算每个位置上每张牌的边缘概率 (Marginal Probability)
        position_probs = defaultdict(lambda: defaultdict(float))
        total_weight = sum(w for h, w in weighted_hypotheses)
        
        if total_weight == 0:
            return None

        for hypothesis, weight in weighted_hypotheses:
            for idx, card in enumerate(hypothesis):
                position_probs[idx][card] += weight / total_weight

        # 2. 动态计算己方风险系数
        # 己方隐藏牌越少，猜错被迫翻牌的风险呈指数上升
        risk_factor = 20.0 / max(1, my_hidden_count)  # 避免除以 0

        best_move = None
        max_ev = float('-inf')

        # 3. 遍历所有可能的目标，寻找 EV 最大的行动
        for target_idx, card_probs in position_probs.items():
            for card, prob in card_probs.items():
                ev = self.calculate_ev(prob, risk_factor)
                if ev > max_ev:
                    max_ev = ev
                    best_move = {
                        'target_index': target_idx,
                        'guess_card': card,
                        'win_probability': prob,
                        'expected_value': ev
                    }

        return best_move


class GameController:
    """主循环：调度上述所有模块"""
    def __init__(self, my_cards, public_cards):
        self.my_cards = my_cards
        self.public_cards = public_cards
        self.inference_engine = DaVinciInferenceEngine(my_cards, public_cards)
        self.psy_filter = PsychologicalFilter()
        self.decision_engine = DaVinciDecisionEngine()
        self.opponent_history = [] 

    def update_opponent_action(self, action_dict):
        """记录对手的行为日志，供心理战过滤器使用"""
        self.opponent_history.append(action_dict)

    def run_turn(self, opponent_card_count):
        """执行一个回合的思考"""
        # 1. 状态空间搜索 (DFS)
        raw_hypotheses = self.inference_engine.generate_valid_hypotheses(opponent_card_count)
        
        # 2. 心理战/概率过滤 (贝叶斯后验更新)
        weighted_hypotheses = self.psy_filter.apply_positive_inference(raw_hypotheses, self.opponent_history)
        
        # 3. 期望值决策 (EV 计算)
        my_hidden_count = len([c for c in self.my_cards if c not in self.public_cards.get('me', [])])
        best_move = self.decision_engine.get_best_move(weighted_hypotheses, my_hidden_count)
        
        return best_move, len(raw_hypotheses)
