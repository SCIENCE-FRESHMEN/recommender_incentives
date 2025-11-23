# Rethinking Incentives in Recommender Systems

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Implementation of the paper "Rethinking Incentives in Recommender Systems: Are Monotone Rewards Always Beneficial?" which explores how reward mechanisms in recommendation platforms affect content creator behavior and social welfare.

## Overview

This repository contains a complete implementation of the theoretical models and experiments described in the paper. The core contribution is the Backward Rewarding Mechanism (BRM), which significantly outperforms traditional Merit-based Monotone Mechanisms (M³) in maximizing social welfare while ensuring content diversity.

Key features:
- Implementation of both M³ mechanisms (exposure-based and engagement-based)
- Backward Rewarding Mechanisms (BRM) with theoretical guarantees
- Algorithm 2 for empirical welfare optimization when welfare metrics are not explicitly defined
- Synthetic environment experiments (G1 and G2 settings)
- MovieLens-1m dataset experiments
- Visualization tools for social welfare curves and group utilities

## Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/recommender-incentives.git
cd recommender-incentives
```

2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # Linux/Mac
# env\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- numpy>=1.21.0
- pandas>=1.3.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- scikit-learn>=1.0.0
- torch>=1.10.0
- tqdm>=4.62.0
- requests>=2.26.0

## Project Structure

```
recommender_incentives/
├── environment.py          # Environment model for user-creator interactions
├── mechanisms.py           # Reward mechanisms (M³ and BRM implementations)
├── algorithms.py           # Core algorithms (Simulator and WelfareOptimizer)
├── utils.py                # Utility functions (data loading, visualization)
├── synthetic_experiment.py # Synthetic dataset experiments
├── movielens_experiment.py # MovieLens dataset experiments
├── main.py                 # Main entry point for running experiments
├── data/                   # Data directory (MovieLens dataset will be downloaded here)
└── results/                # Results directory (output plots and data)
    ├── synthetic/          # Synthetic experiment results
    └── movielens/          # MovieLens experiment results
```

## Running Experiments

### Synthetic Experiments

Run the synthetic dataset experiments for both G1 (zero cost) and G2 (quadratic cost) environments:

```bash
python synthetic_experiment.py
```

### MovieLens Experiments

Run the MovieLens-1m dataset experiments:

```bash
python movielens_experiment.py
```

### Custom Experiments

You can customize experiment parameters by modifying the arguments in the main functions of the experiment files. Key parameters include:
- `n_trials`: Number of experimental trials (default: 5 for synthetic, 3 for MovieLens)
- `steps`: Number of simulation steps (default: 200 for synthetic, 100 for MovieLens)
- `env_type`: Environment type ('G1' or 'G2')
- `seed`: Random seed for reproducibility

## Results

After running the experiments, results will be saved in the `results/` directory:

1. **Social Welfare Curves**: Plots showing how social welfare evolves over simulation steps for different mechanisms
2. **Group Utilities**: Bar charts comparing average utilities across user groups (majority, minority, niche)
3. **Numerical Data**: Numpy arrays containing raw experimental results for further analysis

Example results from the paper:
- BRM mechanisms consistently outperform M³ mechanisms in social welfare
- BRCM_a (algorithm-optimized BRM) often achieves even better welfare than theoretically optimal BRCM*
- BRM mechanisms significantly improve utility for minority and niche user groups while maintaining majority group satisfaction

## Citation

If you use this code in your research, please cite the original paper:

```
@article{yao2023rethinking,
  title={Rethinking Incentives in Recommender Systems: Are Monotone Rewards Always Beneficial?},
  author={Yao, Fan and Li, Chuanhao and Sankararaman, Karthik Abinav and Liao, Yiming and Zhu, Yan and Wang, Qifan and Wang, Hongning and Xu, Haifeng},
  journal={Journal of Machine Learning Research},
  volume={1},
  pages={1-48},
  year={2021}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This implementation is based on the research paper by Yao et al. (2021)
- MovieLens dataset provided by GroupLens Research
- Special thanks to the authors for sharing their theoretical insights
