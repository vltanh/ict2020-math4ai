import os
import argparse

from softmax_regression import softmax_regression

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(0)
sns.set()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d',
                        help='Dataset (iris|digits)',
                        default='iris')
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
    if dataset_str == 'iris':
        X, y = datasets.load_iris(return_X_y=True)
    elif dataset_str == 'digits':
        X, y = datasets.load_digits(return_X_y=True)
    elif dataset_str == 'wine':
        X, y = datasets.load_wine(return_X_y=True)
    elif dataset_str == 'breast_cancer':
        X, y = datasets.load_breast_cancer(return_X_y=True)
    else:
        raise 'Wrong dataset argument, only (iris|digits) are accepted!'
    return X, y


# Parse arguments
args = parse_args()

# Load dataset
X, y = load_data(args.dataset)
N, D = X.shape
C = y.max() + 1

# Save results
output_dir = f'{args.output}/{args.dataset}'
os.makedirs(output_dir, exist_ok=True)

f = open(f'{output_dir}/log.txt', 'w')

# Dataset description
f.write(f'Dataset: {args.dataset}\n')
f.write(f'Number of samples: {N}\n')
f.write(f'Number of features: {D}\n')
f.write(f'Number of classes: {C}\n')
f.write('=======\n')

# Initial weights
W_init = np.random.randn(D + 1, C)

# Hyperparameters
lr = args.learning_rate
max_nsteps = args.max_nsteps

# Solve Softmax Regression by Gradient Descent
W, h = softmax_regression(X, y, W_init, lr, max_nsteps)
Wh, Lh = list(zip(*h))

# Log hyperparameters
f.write(f'Learning rate: {lr}\n')
f.write(f'Number of iterations: {max_nsteps}\n')
f.write('=======\n')

# Visualize Result
f.write('Result:\n')
for i, w in enumerate(W):
    f.write(f'\tw[{i:3d}] =')
    f.write(f'{ " ".join(list(map(str, w))) }\n')
f.write(f'Final loss: {Lh[-1]}')

sns.lineplot(x=range(len(Lh)), y=Lh)
plt.savefig(output_dir + '/' + args.dataset)
plt.close()
