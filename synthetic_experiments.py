import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from environment import Environment
from mechanisms import M3Mechanism, BRCMOptimal, BRCM1, BRMechanism
from algorithms import Simulator, WelfareOptimizer
from utils import ensure_directory_exists, plot_results, plot_group_utilities


def run_synthetic_experiment(env_type='G1', n_trials=5, steps=200, seed=42):
    """运行合成数据集实验"""
    np.random.seed(seed)

    # 实验配置
    mechanism_configs = {
        'M(0)': {'type': 'zero'},
        'M(expo.)': {'type': 'exposure', 'K': 5, 'beta': 0.05},
        'M(enga.)': {'type': 'engagement', 'K': 5, 'beta': 0.05},
        'BRCM*': {'type': 'optimal'},
        'BRCM_a': {'type': 'dynamic'},
        'BRCM_1': {'type': 'simple', 'K': 5}
    }

    # 结果字典
    results = {name: [] for name in mechanism_configs.keys()}
    group_results = {name: [] for name in mechanism_configs.keys()}

    # 创建环境
    if env_type == 'G1':
        cost_type = 'zero'
    elif env_type == 'G2':
        cost_type = 'quadratic'
    else:
        raise ValueError(f"Unknown environment type: {env_type}")

    for trial in tqdm(range(n_trials), desc=f"Running {env_type} experiments"):
        # 创建环境
        env = Environment(
            n_creators=10,
            d=10,
            user_clusters=[20, 10, 8, 5, 3, 3, 2, 1],  # 8个簇，分为3组
            cost_type=cost_type,
            seed=seed + trial
        )

        # 为每种机制运行实验
        for mech_name, config in mechanism_configs.items():
            if config['type'] == 'zero':
                mechanism = M3Mechanism(mechanism_type='zero')
            elif config['type'] == 'exposure':
                mechanism = M3Mechanism(
                    mechanism_type='exposure',
                    K=config['K'],
                    beta=config['beta']
                )
            elif config['type'] == 'engagement':
                mechanism = M3Mechanism(
                    mechanism_type='engagement',
                    K=config['K'],
                    beta=config['beta']
                )
            elif config['type'] == 'optimal':
                mechanism = BRCMOptimal(env.r, env.n_creators)
            elif config['type'] == 'simple':
                mechanism = BRCM1(K=config['K'], n_creators=env.n_creators)
            elif config['type'] == 'dynamic':
                # 动态优化BRCM
                initial_f = np.array([1, 1, 1, 1, 1] + [0] * (env.n_creators - 5))
                optimizer = WelfareOptimizer(env, initial_f, learning_rate=0.1, seed=seed)
                optimal_f, _ = optimizer.optimize(total_epochs=50, inner_steps=5)  # 减少轮数以加快实验
                mechanism = BRMechanism(optimal_f)
            else:
                raise ValueError(f"Unknown mechanism type: {config['type']}")

            # 运行模拟
            simulator = Simulator(env, mechanism, learning_rate=0.1, seed=seed)
            strategies, welfare_history = simulator.run(steps, record_interval=steps // 10)

            # 记录结果
            results[mech_name].append(welfare_history)

            # 记录各组用户效用
            group_utilities = env.get_user_group_utilities(strategies)
            group_results[mech_name].append(group_utilities)

    return results, group_results


def main():
    """主函数"""
    # 创建输出目录
    ensure_directory_exists('results/synthetic')

    # 运行G1环境实验
    print("Running G1 environment experiments (zero cost)...")
    g1_results, g1_group_results = run_synthetic_experiment(
        env_type='G1',
        n_trials=5,
        steps=200,
        seed=42
    )

    # 运行G2环境实验
    print("Running G2 environment experiments (quadratic cost)...")
    g2_results, g2_group_results = run_synthetic_experiment(
        env_type='G2',
        n_trials=5,
        steps=200,
        seed=42
    )

    # 保存结果
    np.save('results/synthetic/g1_results.npy', g1_results)
    np.save('results/synthetic/g1_group_results.npy', g1_group_results)
    np.save('results/synthetic/g2_results.npy', g2_results)
    np.save('results/synthetic/g2_group_results.npy', g2_group_results)

    # 可视化结果
    plot_results(g1_results, 'G1 Environment (Zero Cost)', 'results/synthetic/g1_plot.png')
    plot_results(g2_results, 'G2 Environment (Quadratic Cost)', 'results/synthetic/g2_plot.png')

    # 绘制各组用户效用
    plot_group_utilities(g1_group_results, 'G1 Group Utilities', 'results/synthetic/g1_group_plot.png')
    plot_group_utilities(g2_group_results, 'G2 Group Utilities', 'results/synthetic/g2_group_plot.png')

    print("Synthetic experiments completed successfully!")


if __name__ == "__main__":
    main()