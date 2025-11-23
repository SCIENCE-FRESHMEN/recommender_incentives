import numpy as np
import random
import math


class RewardMechanism:
    """奖励机制基类"""

    def __init__(self):
        pass

    def compute_reward(self, creator_idx, scores):
        """计算给定创作者的奖励"""
        raise NotImplementedError

    def compute_total_reward(self, scores):
        """计算所有创作者的总奖励"""
        return sum(self.compute_reward(i, scores) for i in range(len(scores)))


class M3Mechanism(RewardMechanism):
    """基于绩效的单调机制 (M³)"""

    def __init__(self, mechanism_type='exposure', K=5, beta=0.05):
        """
        初始化M³机制

        参数:
        mechanism_type: 机制类型 ('zero', 'exposure' 或 'engagement')
        K: top-K参数
        beta: 温度参数
        """
        self.mechanism_type = mechanism_type
        self.K = K
        self.beta = beta

    def compute_reward(self, creator_idx, scores):
        """计算奖励"""
        n = len(scores)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_indices]

        if self.mechanism_type == 'zero':
            # M(0) 机制: 奖励等于匹配分数
            return scores[creator_idx]

        elif self.mechanism_type == 'exposure':
            # 基于曝光的机制
            if creator_idx in sorted_indices[:self.K]:
                numerator = math.exp(self.beta * scores[creator_idx])
                denominator = sum(math.exp(self.beta * s) for s in sorted_scores[:self.K])
                return numerator / denominator
            else:
                return 0.0

        elif self.mechanism_type == 'engagement':
            # 基于参与度的机制
            if creator_idx in sorted_indices[:self.K]:
                numerator = math.exp(self.beta * scores[creator_idx])
                denominator = sum(math.exp(self.beta * s) for s in sorted_scores[:self.K])
                # 计算总用户福利
                total_welfare = sum(math.log2(1 + math.exp(self.beta * s)) for s in sorted_scores)
                return (numerator / denominator) * total_welfare
            else:
                return 0.0

        else:
            raise ValueError(f"Unknown mechanism type: {self.mechanism_type}")


class BRMechanism(RewardMechanism):
    """后向奖励机制 (BRM)"""

    def __init__(self, f_params):
        """
        初始化BRM机制

        参数:
        f_params: 函数参数，长度为n的数组
        """
        self.f_params = np.array(f_params)
        assert len(self.f_params) > 0, "f_params cannot be empty"
        # 确保参数是非增的
        for i in range(len(self.f_params) - 1):
            assert self.f_params[i] >= self.f_params[i + 1], "f_params must be non-increasing"

    def compute_reward(self, creator_idx, scores):
        """计算BRM奖励"""
        n = len(scores)
        # 按分数降序排序
        sorted_indices = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_indices]

        # 找到创作者在排序中的位置
        rank = np.where(sorted_indices == creator_idx)[0][0]

        # 获取当前创作者和下一个创作者的分数
        current_score = sorted_scores[rank]
        next_score = sorted_scores[rank + 1] if rank + 1 < n else 0.0

        # 计算奖励 (积分简化为乘法，因为f是常数)
        f_val = self.f_params[rank] if rank < len(self.f_params) else 0.0
        return f_val * (current_score - next_score)


class BRCMOptimal(BRMechanism):
    """理论最优BRCM机制"""

    def __init__(self, r_weights, n_creators):
        """
        初始化理论最优BRCM

        参数:
        r_weights: 折扣权重
        n_creators: 创作者数量
        """
        # 确保r_weights长度至少为n_creators
        if len(r_weights) < n_creators:
            r_weights = np.pad(r_weights, (0, n_creators - len(r_weights)), 'constant')
        super().__init__(r_weights[:n_creators])


class BRCM1(BRMechanism):
    """简化BRCM机制 (BRCM₁)"""

    def __init__(self, K=5, n_creators=10):
        """
        初始化BRCM₁

        参数:
        K: 前K个创作者获得奖励
        n_creators: 创作者数量
        """
        f_params = np.array([1.0] * K + [0.0] * (n_creators - K))
        super().__init__(f_params)