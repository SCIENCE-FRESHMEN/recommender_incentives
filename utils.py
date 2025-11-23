import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import requests
import zipfile
from io import BytesIO
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
from tqdm import tqdm

# 修复：添加对 Environment 类的导入
from environment import Environment  # 这是关键修复
from mechanisms import BRMechanism  # 同时也添加这个导入，可能也需要

def ensure_directory_exists(path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)


def download_movielens(data_dir='data/movielens'):
    """下载MovieLens-1m数据集"""
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

    ensure_directory_exists(data_dir)

    zip_path = os.path.join(data_dir, "ml-1m.zip")
    extracted_path = os.path.join(data_dir, "ml-1m")

    if not os.path.exists(extracted_path):
        print("Downloading MovieLens-1m dataset...")
        response = requests.get(url)
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(data_dir)
        print("Download completed!")

    return extracted_path


class MovieLensDataset(Dataset):
    """MovieLens数据集类"""

    def __init__(self, ratings_df):
        self.user_ids = ratings_df['user_id'].values
        self.movie_ids = ratings_df['movie_id'].values
        self.ratings = ratings_df['rating'].values.astype(np.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            'user': torch.tensor(self.user_ids[idx], dtype=torch.long),
            'movie': torch.tensor(self.movie_ids[idx], dtype=torch.long),
            'rating': torch.tensor(self.ratings[idx], dtype=torch.float)
        }


class DeepMatrixFactorization(nn.Module):
    """深度矩阵分解模型"""

    def __init__(self, n_users, n_movies, embedding_dim=32):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # 神经网络层
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # 初始化
        self.user_embedding.weight.data.normal_(0, 0.01)
        self.movie_embedding.weight.data.normal_(0, 0.01)
        self.user_bias.weight.data.zero_()
        self.movie_bias.weight.data.zero_()

    def forward(self, user_ids, movie_ids):
        # 获取嵌入
        user_embed = self.user_embedding(user_ids)
        movie_embed = self.movie_embedding(movie_ids)

        # 获取偏置
        user_b = self.user_bias(user_ids).squeeze()
        movie_b = self.movie_bias(movie_ids).squeeze()

        # 拼接嵌入
        concat_embed = torch.cat([user_embed, movie_embed], dim=1)

        # 预测评分
        base_pred = torch.sum(user_embed * movie_embed, dim=1) + user_b + movie_b + self.global_bias
        nn_pred = self.fc_layers(concat_embed).squeeze()

        # 最终预测
        prediction = base_pred + nn_pred

        return prediction


def plot_results(results, title, save_path=None):
    """绘制社会福利曲线"""
    plt.figure(figsize=(12, 8))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.']

    for i, (mech_name, histories) in enumerate(results.items()):
        # 填充不等长的历史记录
        max_len = max(len(h) for h in histories)
        padded_histories = []
        for h in histories:
            if len(h) < max_len:
                padded_h = np.pad(h, (0, max_len - len(h)), 'edge')
                padded_histories.append(padded_h)
            else:
                padded_histories.append(h)

        # 计算平均值和标准差
        mean_history = np.mean(padded_histories, axis=0)
        std_history = np.std(padded_histories, axis=0)

        # 绘制平均曲线
        x = np.linspace(0, 1, len(mean_history))
        plt.plot(x, mean_history,
                 color=colors[i % len(colors)],
                 linestyle=linestyles[i % len(linestyles)],
                 linewidth=2.5,
                 label=mech_name)

        # 填充标准差区域
        plt.fill_between(x,
                         mean_history - 0.5 * std_history,
                         mean_history + 0.5 * std_history,
                         color=colors[i % len(colors)],
                         alpha=0.2)

    plt.xlabel('Simulation Progress (Normalized)', fontsize=14)
    plt.ylabel('Average User Utility', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        ensure_directory_exists(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_group_utilities(group_results, title, save_path=None):
    """绘制各组用户效用"""
    plt.figure(figsize=(12, 8))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    group_names = ['Majority Group', 'Minority Group', 'Niche Group']

    # 计算每种机制下各组的平均效用
    mechanism_names = list(group_results.keys())
    n_groups = 3  # 我们只关注前3组
    n_mechanisms = len(mechanism_names)

    avg_utilities = np.zeros((n_mechanisms, n_groups))
    std_utilities = np.zeros((n_mechanisms, n_groups))

    for i, mech_name in enumerate(mechanism_names):
        utilities = np.array(group_results[mech_name])
        # 只取前3组
        avg_utilities[i] = np.mean(utilities[:, :n_groups], axis=0)
        std_utilities[i] = np.std(utilities[:, :n_groups], axis=0)

    # 创建条形图
    x = np.arange(n_groups)
    width = 0.8 / n_mechanisms

    for i, mech_name in enumerate(mechanism_names):
        offset = (i - n_mechanisms / 2) * width + width / 2
        bars = plt.bar(x + offset, avg_utilities[i], width=width,
                       yerr=0.5 * std_utilities[i], capsize=5,
                       color=colors[i % len(colors)], alpha=0.8,
                       label=mech_name)

        # 在条形上添加数值标签
        for j, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.xlabel('User Groups', fontsize=14)
    plt.ylabel('Average Utility', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xticks(x, group_names, fontsize=12)
    plt.legend(fontsize=12, loc='best')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        ensure_directory_exists(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def prepare_movielens_environment(cost_type='zero', seed=42):
    """准备MovieLens环境"""
    # 下载数据集
    data_dir = download_movielens()

    # 加载评分数据
    ratings_path = os.path.join(data_dir, "ratings.dat")
    ratings_df = pd.read_csv(ratings_path, sep='::', header=None,
                             names=['user_id', 'movie_id', 'rating', 'timestamp'],
                             engine='python')

    # 数据预处理
    print("Preprocessing MovieLens data...")

    # 过滤活跃用户和热门电影
    user_counts = ratings_df['user_id'].value_counts()
    movie_counts = ratings_df['movie_id'].value_counts()

    # 保留评分次数在50-500之间的用户
    active_users = user_counts[(user_counts >= 50) & (user_counts <= 500)].index

    # 保留被评分次数在50-500之间的电影
    popular_movies = movie_counts[(movie_counts >= 50) & (movie_counts <= 500)].index

    # 过滤数据 - 修复SettingWithCopyWarning
    filtered_ratings = ratings_df.copy()
    filtered_ratings = filtered_ratings[
        (filtered_ratings['user_id'].isin(active_users)) &
        (filtered_ratings['movie_id'].isin(popular_movies))
        ].copy()

    print(f"Original ratings: {len(ratings_df)}, Filtered ratings: {len(filtered_ratings)}")

    # 重新索引用户和电影
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(filtered_ratings['user_id'].unique())}
    movie_id_map = {old_id: new_id for new_id, old_id in enumerate(filtered_ratings['movie_id'].unique())}

    # 修复SettingWithCopyWarning - 使用.loc
    filtered_ratings.loc[:, 'user_id'] = filtered_ratings['user_id'].map(user_id_map)
    filtered_ratings.loc[:, 'movie_id'] = filtered_ratings['movie_id'].map(movie_id_map)

    n_users = len(user_id_map)
    n_movies = len(movie_id_map)

    print(f"Number of users: {n_users}, Number of movies: {n_movies}")

    # 准备数据集
    dataset = MovieLensDataset(filtered_ratings)

    # 划分训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 训练深度矩阵分解模型
    print("Training Deep Matrix Factorization model...")
    model = DeepMatrixFactorization(n_users, n_movies, embedding_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_rmse = float('inf')
    patience = 3
    no_improve = 0

    for epoch in range(20):  # 减少训练轮数以加快实验
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            user_ids = batch['user'].to(device)
            movie_ids = batch['movie'].to(device)
            ratings = batch['rating'].to(device)

            optimizer.zero_grad()
            predictions = model(user_ids, movie_ids)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                user_ids = batch['user'].to(device)
                movie_ids = batch['movie'].to(device)
                ratings = batch['rating'].to(device)

                predictions = model(user_ids, movie_ids)
                loss = criterion(predictions, ratings)
                test_loss += loss.item()

        train_rmse = np.sqrt(train_loss / len(train_loader))
        test_rmse = np.sqrt(test_loss / len(test_loader))

        print(f"Epoch {epoch + 1}: Train RMSE = {train_rmse:.4f}, Test RMSE = {test_rmse:.4f}")

        # 早停
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"Best Test RMSE: {best_rmse:.4f}")

    # 加载最佳模型
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    # 获取用户和电影嵌入
    user_embeddings = model.user_embedding.weight.cpu().detach().numpy()
    movie_embeddings = model.movie_embedding.weight.cpu().detach().numpy()

    # 归一化嵌入
    user_embeddings = user_embeddings / np.linalg.norm(user_embeddings, axis=1, keepdims=True)
    movie_embeddings = movie_embeddings / np.linalg.norm(movie_embeddings, axis=1, keepdims=True)

    # 构建环境
    print("Building environment...")

    # 选择2550个用户和1783部电影
    max_users = min(2550, n_users)
    max_movies = min(1783, n_movies)

    selected_users = np.random.choice(n_users, max_users, replace=False)
    selected_movies = np.random.choice(n_movies, max_movies, replace=False)

    env_users = user_embeddings[selected_users]
    env_movies = movie_embeddings[selected_movies]

    # 定义匹配分数函数
    def matching_score(content, user):
        score = np.dot(content, user)
        # 归一化到[0,1]
        return np.clip((score + 1) / 2, 0, 1)

    # 创建环境
    env = Environment(
        n_creators=10,
        d=32,
        user_clusters=[300, 300, 300, 300, 300, 300, 300, 300, 300, 250][:8],  # 8个大致相等的群体
        cluster_centers=env_users[:8],  # 使用前8个用户作为簇中心
        cost_type=cost_type,
        seed=seed
    )

    # 重写环境的匹配分数函数
    env.matching_score = matching_score
    env.users = env_users
    env.movies = env_movies

    return env