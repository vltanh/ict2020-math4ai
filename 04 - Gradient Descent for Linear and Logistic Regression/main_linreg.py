import os
import argparse

from linear_regression import linear_regression

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(0)
sns.set()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d',
                        help='Dataset (boston|diabetes)',
                        default='boston')
    parser.add_argument('--output', '-o',
                        help='Output directory (will be created if not existed)',
                        default='output')
    parser.add_argument('--learning_rate', '-lr',
                        type=float,
                        help='Learning rate',
                        default=0.0001)
    parser.add_argument('--max_nsteps', '-n',
                        type=int,
                        help='Max number of iterations',
                        default=10000)
    return parser.parse_args()


def load_data(dataset_str):
    if dataset_str == 'boston':
        X, y = datasets.load_boston(return_X_y=True)
    elif dataset_str == 'diabetes':
        X, y = datasets.load_diabetes(return_X_y=True)
    else:
        raise 'Wrong dataset argument, only (boston|diabetes) are accepted!'
    return X, y


# Parse arguments
args = parse_args()

# Save results
output_dir = f'{args.output}/{args.dataset}'
os.makedirs(output_dir, exist_ok=True)

f = open(f'{output_dir}/log.txt', 'w')

# Load dataset
X, y = load_data(args.dataset)

# Dataset description
f.write(f'Dataset: {args.dataset}\n')
f.write(f'Number of samples: {X.shape[0]}\n')
f.write(f'Number of features: {X.shape[1]}\n')
f.write('=======\n')

# Initial weights
W_init = np.random.randn(X.shape[1] + 1)

# Hyperparameters
lr = args.learning_rate
max_nsteps = args.max_nsteps

# Solve Linear Regression by Gradient Descent
W, h = linear_regression(X, y, W_init, lr, max_nsteps)

# Log hyperparameters
f.write(f'Learning rate: {lr}\n')
f.write(f'Number of iterations: {max_nsteps}\n')
f.write('=======\n')

# Visualize Result
f.write('Result:\n')
for i, w in enumerate(W):
    f.write(f'\tw[{i:3d}] = {w}\n')
f.write(f'Final loss: {h[-1]}')

sns.lineplot(x=range(len(h)), y=h)
plt.savefig(output_dir + '/' + args.dataset)
plt.close()
