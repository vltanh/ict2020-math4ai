import os
import argparse

import pca

from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d',
                        help='Dataset (iris|digits|wine|breast-cancer)',
                        default='iris')
    parser.add_argument('--output', '-o',
                        help='Output directory (will be created if not existed)',
                        default='output')
    return parser.parse_args()


def load_data(dataset_str):
    if dataset_str == 'iris':
        X, y = datasets.load_iris(return_X_y=True)
    elif dataset_str == 'digits':
        X, y = datasets.load_digits(return_X_y=True)
    elif dataset_str == 'wine':
        X, y = datasets.load_wine(return_X_y=True)
    elif dataset_str == 'breast-cancer':
        X, y = datasets.load_breast_cancer(return_X_y=True)
    else:
        raise 'Wrong dataset argument, only (iris|digits|wine|breast-cancer) are accepted!'
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

# My PCA compression
Uk, x_avg, Z = pca.compress(X, k=2)
Xk = pca.decode(Uk, x_avg, Z)

sk_pca = PCA(2).fit(X)
Z_sk = sk_pca.transform(X)
Xk_sk = sk_pca.inverse_transform(Z_sk)

# Calculate reconstruction loss
error = np.linalg.norm(X - Xk) / np.linalg.norm(X)
f.write(f'[My] Error ||X-Xk||/||X|| (Frobenius): {error:.05f}\n')

error = np.linalg.norm(X - Xk_sk) / np.linalg.norm(X)
f.write(f'[Scikit] Error ||X-Xk_sk||/||X|| (Frobenius): {error:.05f}\n')

# Calculate differences between methods
error = np.linalg.norm(Xk - Xk_sk)
f.write(f'[My vs Scikit] Error ||Xk-Xk_sk|| (Frobenius): {error:.05f}\n')

# Plot reduction
sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=y, palette='Set2')
plt.savefig(f'{output_dir}/reduced')
plt.close()
