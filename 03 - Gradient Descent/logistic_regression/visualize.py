import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_blobs

from logistic_regression import logistic_regression

sns.set()
np.random.seed(0)


def visualize_2d(X, y, Wh, Lh, thres=0.5):
    plt.ion()

    fig, ax = plt.subplots(1, 2)

    ax[0].set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    ax[0].set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)

    q = np.quantile(Lh, 0.9999)
    ax[1].set_xlim(-0.5, len(Lh))
    ax[1].set_ylim(0, q)

    sns.scatterplot(x=X[:, 0], y=X[:, 1], ax=ax[0], hue=y)

    X = np.array([X.min() - 1, X.max() + 1])
    line, = ax[0].plot([], [], 'r')

    loss, = ax[1].plot([], [])

    step = 100
    for i in range(0, len(Wh), step):
        W = Wh[i]

        y = (- np.log(1 / thres - 1) - W[0] - W[1] * X) / W[2]
        line.set_data(X, y)

        loss.set_data(range(i), Lh[:i])

        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.pause(10)


def main():
    # Data
    N = 1000
    X, y = make_blobs(n_samples=N, centers=2, n_features=2)

    # Initial weights
    W_init = np.random.randn(3)

    # Hyperparameters
    lr = 0.1
    max_nsteps = 100000

    # Solve Logistic Regression by Gradient Descent
    W, h = logistic_regression(X, y, W_init, lr, max_nsteps)
    Wh, Lh = list(zip(*h))

    # Visualize Result
    print('Result:')
    print(W)
    print('Final loss:', Lh[-1])

    sns.lineplot(x=range(len(Lh)), y=Lh)
    plt.savefig('logreg')
    plt.close()

    visualize_2d(X, y, Wh, Lh)


if __name__ == '__main__':
    import traceback
    try:
        main()
    except Exception as e:
        traceback.print_exc()
