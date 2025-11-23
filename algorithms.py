import numpy as np
import random
import math
from tqdm import tqdm
from mechanisms import BRMechanism

class Simulator:
    """模拟内容创作者策略演变 (Algorithm 1)"""

    def __init__(self, env, mechanism, learning_rate=0.1, seed=42):
        """
        初始化模拟器

        参数:
        env: 环境
        mechanism: 奖励机制
        learning_rate: 学习率
        seed: 随机种子
        """
        np.random.seed(seed)
        random.seed(seed)

        self.env = env
        self.mechanism = mechanism
        self.learning_rate = learning_rate
        self.seed = seed

        # 初始化策略
        self.strategies = env.initial_strategies.copy()

    def creator_utility(self, creator_idx, strategy, other_strategies):
        """计算创作者效用"""
        # 临时替换策略
        all_strategies = np.vstack([other_strategies[:creator_idx],
                                    strategy,
                                    other_strategies[creator_idx:]])

        total_reward = 0.0

        # 计算在所有用户上的平均奖励
        for user in self.env.users:
            # 计算所有内容的匹配分数
            scores = np.array([self.env.matching_score(s, user) for s in all_strategies])
            # 计算该创作者的奖励
            reward = self.mechanism.compute_reward(creator_idx, scores)
            total_reward += reward

        # 平均奖励
        avg_reward = total_reward / len(self.env.users)
        # 减去成本
        cost = self.env.creator_cost(creator_idx, strategy)

        return avg_reward - cost

    def step(self):
        """执行一步模拟，返回是否发生了策略更新"""
        updated = False

        # 随机选择一个创作者
        creator_idx = random.randint(0, self.env.n_creators - 1)

        # 保存当前策略
        current_strategy = self.strategies[creator_idx].copy()
        other_strategies = np.vstack([self.strategies[:creator_idx],
                                      self.strategies[creator_idx + 1:]])

        # 计算当前效用
        current_utility = self.creator_utility(creator_idx, current_strategy, other_strategies)

        # 生成随机改进方向
        random_direction = np.random.normal(0, 1, self.env.d)
        random_direction = random_direction / np.linalg.norm(random_direction)

        # 尝试新策略
        new_strategy = current_strategy + self.learning_rate * random_direction
        new_strategy = new_strategy / np.linalg.norm(new_strategy)  # 保持在单位球面上

        # 计算新效用
        new_utility = self.creator_utility(creator_idx, new_strategy, other_strategies)

        # 如果新策略更好，则更新
        if new_utility > current_utility:
            self.strategies[creator_idx] = new_strategy
            updated = True

        return updated

    def run(self, steps, record_interval=10):
        """运行模拟并记录社会福利历史

        参数:
        steps: 总步数
        record_interval: 记录间隔

        返回:
        strategies: 最终策略
        welfare_history: 社会福利历史
        """
        welfare_history = []

        for step in range(steps):
            self.step()
            if step % record_interval == 0:
                welfare = self.env.social_welfare(self.strategies)
                welfare_history.append(welfare)

        return self.strategies.copy(), welfare_history


class WelfareOptimizer:
    """社会福利优化器 (Algorithm 2)"""

    def __init__(self, env, initial_f, learning_rate=0.1, seed=42):
        """
        初始化优化器

        参数:
        env: 环境
        initial_f: 初始f参数
        learning_rate: 学习率
        seed: 随机种子
        """
        np.random.seed(seed)
        random.seed(seed)

        self.env = env
        self.f = np.array(initial_f)
        self.learning_rate = learning_rate
        self.seed = seed

    def optimize(self, total_epochs, inner_steps, welfare_evaluator=None):
        """
        优化BRCM参数

        参数:
        total_epochs: 总轮数
        inner_steps: 内部模拟步数
        welfare_evaluator: 福利评估函数 (可选)

        返回:
        best_f: 最优f参数
        welfare_history: 福利历史
        """
        best_f = self.f.copy()
        best_welfare = -np.inf
        welfare_history = []

        for epoch in tqdm(range(total_epochs), desc="Optimizing BRCM"):
            # 保存当前参数
            current_f = self.f.copy()

            # 随机扰动参数
            perturbation_idx = random.randint(0, len(self.f) - 1)
            perturbation_dir = random.choice([-1, 1])
            self.f[perturbation_idx] += self.learning_rate * perturbation_dir

            # 确保参数非增
            for i in range(1, len(self.f)):
                if self.f[i] > self.f[i - 1]:
                    self.f[i] = self.f[i - 1]

            # 确保参数非负
            self.f = np.maximum(self.f, 0)

            # 创建机制并模拟
            mechanism = BRMechanism(self.f)
            simulator = Simulator(self.env, mechanism, learning_rate=0.1, seed=self.seed)

            # 运行内部模拟
            strategies, _ = simulator.run(inner_steps)

            # 评估社会福利
            if welfare_evaluator is None:
                current_welfare = self.env.social_welfare(strategies)
            else:
                current_welfare = welfare_evaluator(strategies)

            welfare_history.append(current_welfare)

            # 如果福利提高，保留扰动；否则恢复
            if current_welfare > best_welfare:
                best_welfare = current_welfare
                best_f = self.f.copy()
            else:
                self.f = current_f.copy()

        return best_f, welfare_history