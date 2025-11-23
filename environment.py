import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import cosine


class Environment:
    """内容创作者竞争环境"""

    def __init__(self, n_creators=10, d=10, user_clusters=None, cluster_centers=None,
                 cost_type='zero', cost_weight=0.5, seed=42):
        """
        初始化环境

        参数:
        n_creators: 内容创作者数量
        d: 嵌入维度
        user_clusters: 每个用户簇的大小列表
        cluster_centers: 预定义的簇中心
        cost_type: 成本类型 ('zero' 或 'quadratic')
        cost_weight: 二次成本权重
        seed: 随机种子
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.n_creators = n_creators
        self.d = d
        self.cost_type = cost_type
        self.cost_weight = cost_weight
        self.seed = seed

        # 设置用户聚类
        if user_clusters is None:
            # 默认设置：8个簇，分为3组
            self.user_clusters = [20, 10, 8, 5, 3, 3, 2, 1]
        else:
            self.user_clusters = user_clusters

        self.n_clusters = len(self.user_clusters)
        self.total_users = sum(self.user_clusters)

        # 生成或设置簇中心
        if cluster_centers is None:
            self.cluster_centers = self._generate_cluster_centers()
        else:
            self.cluster_centers = cluster_centers

        # 生成用户
        self.users = self._generate_users()

        # 生成创作者初始策略和成本中心
        self.cost_centers = self._generate_cost_centers() if cost_type == 'quadratic' else None
        self.initial_strategies = self._generate_initial_strategies()

        # 定义折扣权重 (top-K环境，K=5)
        self.r = np.array([1 / np.log2(k + 1) for k in range(1, 6)] + [0] * (n_creators - 5))

    def _generate_cluster_centers(self):
        """在单位球面上生成随机簇中心"""
        centers = np.random.normal(0, 1, (self.n_clusters, self.d))
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        return centers

    def _generate_users(self):
        """生成用户群体"""
        users = []
        for i, cluster_size in enumerate(self.user_clusters):
            cluster_center = self.cluster_centers[i]
            # 生成高斯分布的用户
            cluster_users = np.random.normal(cluster_center, 0.3, (cluster_size, self.d))
            # 归一化到单位球面
            norms = np.linalg.norm(cluster_users, axis=1, keepdims=True)
            cluster_users = cluster_users / norms
            users.append(cluster_users)

        return np.vstack(users)

    def _generate_cost_centers(self):
        """生成创作者的成本中心 (用于G2)"""
        cost_centers = np.random.normal(0, 1, (self.n_creators, self.d))
        cost_centers = cost_centers / np.linalg.norm(cost_centers, axis=1, keepdims=True)
        return cost_centers

    def _generate_initial_strategies(self):
        """生成创作者的初始策略"""
        if self.cost_type == 'zero':
            # G1环境: 所有创作者初始策略设为最大用户群的中心
            max_cluster_idx = np.argmax(self.user_clusters)
            return np.tile(self.cluster_centers[max_cluster_idx], (self.n_creators, 1))
        else:
            # G2环境: 初始策略为各自的成本中心
            return self.cost_centers.copy()

    def matching_score(self, content, user):
        """计算内容与用户的匹配分数 (移位内积)"""
        score = np.dot(content, user)
        # 移位并缩放至[0,1]范围
        return (score + 1) / 2

    def creator_cost(self, creator_idx, strategy):
        """计算创作者的生产成本"""
        if self.cost_type == 'zero':
            return 0.0
        elif self.cost_type == 'quadratic':
            cost_center = self.cost_centers[creator_idx]
            return self.cost_weight * np.linalg.norm(strategy - cost_center) ** 2
        else:
            raise ValueError(f"Unknown cost type: {self.cost_type}")

    def user_utility(self, user, strategies):
        """计算用户效用"""
        # 计算所有内容的匹配分数
        scores = np.array([self.matching_score(s, user) for s in strategies])
        # 按匹配分数降序排序
        sorted_indices = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_indices]

        # 计算加权效用
        utility = 0.0
        for k in range(min(len(self.r), len(sorted_scores))):
            utility += self.r[k] * sorted_scores[k]

        return utility

    def social_welfare(self, strategies):
        """计算社会福利"""
        total_user_utility = 0.0

        # 计算所有用户的总效用
        for user in self.users:
            total_user_utility += self.user_utility(user, strategies)

        # 平均用户效用
        avg_user_utility = total_user_utility / len(self.users)

        return avg_user_utility

    def get_user_group_utilities(self, strategies):
        """计算各用户群体的平均效用"""
        group_utilities = []
        start_idx = 0

        for cluster_size in self.user_clusters:
            group_utility = 0.0
            for i in range(start_idx, start_idx + cluster_size):
                user = self.users[i]
                group_utility += self.user_utility(user, strategies)
            group_utilities.append(group_utility / cluster_size)
            start_idx += cluster_size

        return np.array(group_utilities)