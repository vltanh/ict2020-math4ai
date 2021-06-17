import argparse
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_blobs

from softmax_regression import softmax_regression

sns.set()
np.random.seed(0)

COLOR = ['blue', 'orange', 'red', 'green', 'violet']


def graph(a, b, c, x_min=-1, x_max=1, y_min=-1, y_max=1, mode='=', ax=None, color=None, alpha=0.2):
    '''
    Graph the halfspace determined by $ax + by + c = 0$.
    '''
    assert mode in ['=', '>', '<', '>=', '<=']
    ax = ax or plt.gca()

    is_halfspace = mode != '='
    include_boundary = '=' in mode
    boundary_style = 'dashed' if not include_boundary else 'solid'

    top = [y_max, y_max]
    bottom = [y_min, y_min]
    left = [x_min, x_min]
    right = [x_max, x_max]

    if a == 0 and b == 0 and c == 0 and include_boundary:
        ax.fill_betweenx([y_min, y_max], left, right, color=color, alpha=alpha)
    elif a == 0 and b == 0:
        raise Exception("Error!, No way I can graph when a = b = 0!")
    elif a == 0:
        y = - c / b
        ax.hlines(y, x_min, x_max, color=color, linestyle=boundary_style)
        if is_halfspace:
            if '>' in mode:
                if b > 0:
                    ax.fill_betweenx([max(y, y_min), y_max],
                                     left, right, color=color, alpha=alpha)
                else:
                    ax.fill_betweenx([y_min, min(y, y_max)],
                                     left, right, color=color, alpha=alpha)
            else:
                if b > 0:
                    ax.fill_betweenx([y_min, min(y, y_max)],
                                     left, right, color=color, alpha=alpha)
                else:
                    ax.fill_betweenx([max(y, y_min), y_max],
                                     left, right, color=color, alpha=alpha)
    elif b == 0:
        x = - c / a
        ax.plot([x, x], [y_min, y_max], color=color, linestyle=boundary_style)
        if is_halfspace:
            if '>' in mode:
                if a > 0:
                    ax.fill_between([max(x, x_min), x_max],
                                    bottom, top, color=color, alpha=alpha)
                else:
                    ax.fill_between([x_min, min(x, x_max)],
                                    bottom, top, color=color, alpha=alpha)
            else:
                if a > 0:
                    ax.fill_between([x_min, min(x, x_max)],
                                    bottom, top, color=color, alpha=alpha)
                else:
                    ax.fill_between([max(x, x_min), x_max],
                                    bottom, top, color=color, alpha=alpha)
    else:
        y_left = (- a * x_min - c) / b
        y_right = (- a * x_max - c) / b
        ax.plot([x_min, x_max], [y_left, y_right],
                color=color, linestyle=boundary_style)
        if is_halfspace:
            if '>' in mode:
                if b > 0:
                    ax.fill_between([x_min, x_max],
                                    [y_left, y_right], top, color=color, alpha=alpha)
                else:
                    ax.fill_between([x_min, x_max],
                                    bottom, [y_left, y_right], color=color, alpha=alpha)
            else:
                if b > 0:
                    ax.fill_between([x_min, x_max],
                                    bottom, [y_left, y_right], color=color, alpha=alpha)
                else:
                    ax.fill_between([x_min, x_max],
                                    [y_left, y_right], top, color=color, alpha=alpha)


def visualize_2d(X, y, Wh, Lh, mode='ovr'):
    X_ = X.copy()
    y_ = y.copy()

    assert mode in ['ovo', 'ovr']

    plt.ion()

    fig, ax = plt.subplots(1, 2)

    x_min = X[:, 0].min() - 0.5
    x_max = X[:, 0].max() + 0.5
    y_min = X[:, 1].min() - 0.5
    y_max = X[:, 1].max() + 0.5

    ax[0].set_xlim(x_min, x_max)
    ax[0].set_ylim(y_min, y_max)

    q = np.quantile(Lh, 0.8)
    ax[1].set_xlim(0, len(Lh))
    ax[1].set_ylim(q - 0.1 * np.std(Lh), q + np.std(Lh))

    sns.scatterplot(x=X[:, 0], y=X[:, 1], ax=ax[0], hue=y)

    X = np.array([X.min() - 1, X.max() + 1])

    C = Wh[0].shape[1]
    loss, = ax[1].plot([], [])

    step = 100
    for i in range(0, len(Wh), step):
        W = Wh[i]

        ax[0].cla()
        ax[0].set_xlim(x_min, x_max)
        ax[0].set_ylim(y_min, y_max)
        sns.scatterplot(x=X_[:, 0], y=X_[:, 1], ax=ax[0],
                        hue=y_, palette=COLOR[:C])

        if mode == 'ovo':
            for ci, (wi, wj) in enumerate(combinations(W.T, 2)):
                d = wi - wj
                graph(d[1], d[2], d[0],
                      x_min, x_max, y_min, y_max,
                      mode='=', ax=ax[0],
                      color=COLOR[ci])
        elif mode == 'ovr':
            for ci, w in enumerate(W.T):
                graph(w[1], w[2], w[0] + np.log(C - 1),
                      x_min, x_max, y_min, y_max,
                      mode='>', ax=ax[0],
                      color=COLOR[ci])

        loss.set_data(range(i), Lh[:i])

        fig.canvas.draw()
        fig.canvas.flush_events()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m',
                        help='Mode One vs One/Rest',
                        default='ovr',
                        choices={'ovo', 'ovr'})
    return parser.parse_args()


def main(args):
    # Data
    N, C = 1000, 3
    X, y = make_blobs(n_samples=N, centers=C, n_features=2)

    # Initial weights
    W_init = np.random.randn(3, C)

    # Hyperparameters
    lr = 0.1
    max_nsteps = 100000

    # Solve Softmax Regression by Gradient Descent
    W, h = softmax_regression(X, y, W_init, lr, max_nsteps)
    Wh, Lh = list(zip(*h))

    visualize_2d(X, y, Wh, Lh, mode=args.mode)
    plt.pause(10)


if __name__ == '__main__':
    import traceback
    try:
        args = parse_args()
        main(args)
    except Exception as e:
        traceback.print_exc()
