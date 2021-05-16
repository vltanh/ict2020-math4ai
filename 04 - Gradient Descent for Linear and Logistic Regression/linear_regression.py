import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
np.random.seed(0)


def linear_regression(X, y, W_init, lr, max_nsteps):
    N, D = X.shape

    # Assign bias values
    X_ = np.ones((N, D+1))
    X_[:, 1:] = X

    # Initialize weights
    W = W_init

    # Log history
    history = []

    step = 0
    while True:
        # Forward:
        #   h_i = sum_{j=1}^{D} W_j * X_{i,j}
        # === Vectorize ===
        #   h = X.T W
        h = X_ @ W

        # Loss:
        #   L = 1/N sum_{i=1}^{N} 1/2 (h_i - y_i)^2
        # === Vectorize ===
        #   L = 1/2N (h - y).T (h - y)
        loss = ((h - y) ** 2).mean() / 2
        history.append(loss)

        # Gradient:
        #   dL/dW_j = 1/N sum_{i=1}^{N} dh_i/dW_j * (h_i - y_i)
        #           = 1/N sum_{i=1}^{N} X_{i, j} * (h_i - y_i)
        # === Vectorize ===
        #   dL/dW = 1/N (h - y).T dh/dW = (h - y).T X / N
        dW = (h - y) @ X_ / N

        # Update:
        #   W_j := W_j - \alpha * dL/dW_j
        # === Vectorize ===
        #   W := W - \alpha * dL/dW
        W = W - lr * dW

        # Stop condition
        step += 1
        if step > max_nsteps:
            break

    return W, history


def test():
    # Data
    X = np.array([
        [1., 2.],
        [3., 4.],
        [-1., -5.]
    ])
    y = np.array([3., 6., 2.])

    # Initial weights
    W_init = np.random.randn(3)

    # Hyperparameters
    lr = 0.01
    max_nsteps = 100000

    # Solve Linear Regression by Gradient Descent
    W, h = linear_regression(X, y, W_init, lr, max_nsteps)

    # Visualize Result
    print('Result:', W)
    print('Final loss:', h[-1])

    sns.lineplot(x=range(len(h)), y=h)
    plt.savefig('linreg')
    plt.close()


if __name__ == '__main__':
    try:
        test()
    except Exception as e:
        print(e)
