import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from linear_regression import linear_regression

sns.set()
np.random.seed(0)


def visualize_2d(X, y, f, Wh, Lh):
    plt.ion()

    fig, ax = plt.subplots(1, 2)

    ax[0].set_xlim(X.min() - 0.5, X.max() + 0.5)
    ax[0].set_ylim(y.min() - 0.5, y.max() + 0.5)

    q = np.quantile(Lh, 0.9999)
    ax[1].set_xlim(-0.5, len(Lh))
    ax[1].set_ylim(0, q)

    ax[0].scatter(X.reshape(-1,), y)

    X = np.array([X.min() - 1, X.max() + 1])

    ax[0].plot(X, f(X), 'g-')
    line, = ax[0].plot([], [], 'r')

    loss, = ax[1].plot([0], [Lh[0]])

    step = 100
    for i in range(0, len(Wh), step):
        W = Wh[i]

        y = W[0] + W[1] * X
        line.set_data(X, y)

        loss.set_data(range(i), Lh[:i])

        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.pause(10)


def main():
    # Data
    def f(x):
        return 8 + 9 * x

    N = 100
    X = 3 * np.random.rand(N) - 6
    err = np.random.randn(N)
    y = f(X) + err
    X = X.reshape(-1, 1)

    # Initial weights
    W_init = np.random.randn(2)

    # Hyperparameters
    lr = 0.01
    max_nsteps = 100000

    # Solve Linear Regression by Gradient Descent
    W, h = linear_regression(X, y, W_init, lr, max_nsteps)
    Wh, Lh = list(zip(*h))

    # Visualize Result
    print('Result:', W)
    print('Final loss:', Lh[-1])

    visualize_2d(X, y, f, Wh, Lh)


if __name__ == '__main__':
    import traceback
    try:
        main()
    except Exception as e:
        traceback.print_exc()
