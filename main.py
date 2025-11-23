import argparse
import time
import os
import sys

# Add the current directory to Python path to ensure imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def run_synthetic_experiments():
    """Run synthetic dataset experiments"""
    print("Starting synthetic experiments...")
    start_time = time.time()

    try:
        # Try to import from synthetic_experiment.py
        from synthetic_experiment import main as synthetic_main
    except ImportError:
        try:
            # Try alternative naming convention
            from synthetic_experiments import main as synthetic_main
        except ImportError:
            print("Error: Could not find synthetic experiment module.")
            print("Please ensure you have either 'synthetic_experiment.py' or 'synthetic_experiments.py' in your project directory.")
            return

    synthetic_main()

    elapsed_time = time.time() - start_time
    print(f"Synthetic experiments completed in {elapsed_time:.2f} seconds")

def run_movielens_experiments():
    """Run MovieLens experiments"""
    print("Starting MovieLens experiments...")
    start_time = time.time()

    try:
        # Try to import from movielens_experiment.py
        from movielens_experiment import main as movielens_main
    except ImportError:
        try:
            # Try alternative naming convention
            from movielens_experiments import main as movielens_main
        except ImportError:
            print("Error: Could not find MovieLens experiment module.")
            print("Please ensure you have either 'movielens_experiment.py' or 'movielens_experiments.py' in your project directory.")
            return

    movielens_main()

    elapsed_time = time.time() - start_time
    print(f"MovieLens experiments completed in {elapsed_time:.2f} seconds")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run experiments for "Rethinking Incentives in Recommender Systems"')
    parser.add_argument('--experiment', type=str, choices=['synthetic', 'movielens', 'all'],
                        default='all', help='Which experiment to run')
    args = parser.parse_args()

    if args.experiment == 'synthetic' or args.experiment == 'all':
        run_synthetic_experiments()

    if args.experiment == 'movielens' or args.experiment == 'all':
        run_movielens_experiments()

    print("All experiments completed successfully!")

if __name__ == "__main__":
    main()