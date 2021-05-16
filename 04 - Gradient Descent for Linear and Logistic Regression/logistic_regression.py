import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
np.random.seed(0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    # sigma'(x) = sigma(x) * (1 - sigma(x))
    sx = sigmoid(x)
    return sx * (1 - sx)


def logistic_regression(X, y, W_init, lr, max_nsteps):
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
        #   h_i = sigmoid(- sum_{j=1}^{D} W_j * X_{i,j})
        # === Vectorize ===
        #   h = sigmoid(X.T W)
        h = sigmoid(X_ @ W)

        # Loss:
        #   L = 1/N sum_{i=1}^{N} - y_i * ln(h_i) - (1 - y_i) * ln(1 - h_i)
        loss = - (y * np.log(h) + (1 - y) * np.log(1 - h)).mean()
        history.append(loss)

        # Gradient:
        #   dL/dW_j = 1/N sum_{i=1}^{N} - dh_i/dW_j * (y_i / h_i + (1 - y_i)/(1 - h_i))
        #           = 1/N sum_{i=1}^{N} - X_{i, j} * h_i * (1 - h_i) * (y_i / h_i - (1 - y_i)/(1 - h_i))
        #           = 1/N sum_{i=1}^{N} - X_{i, j} * (y_i * (1 - h_i) - (1 - y_i) * h_i)
        #           = 1/N sum_{i=1}^{N} - X_{i, j} * (y_i - h_i)
        #           = 1/N sum_{i=1}^{N} X_{i, j} * (h_i - y_i)
        # === Vectorize ===
        #   dL/dW = (h - y).T X
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
    y = np.array([1, 0, 1])

    # Initial weights
    W_init = np.random.randn(3)

    # Hyperparameters
    lr = 0.01
    max_nsteps = 100000

    # Solve Linear Regression by Gradient Descent
    W, h = logistic_regression(X, y, W_init, lr, max_nsteps)

    # Visualize Result
    print('Result:', W)
    print('Final loss:', h[-1])

    sns.lineplot(x=range(len(h)), y=h)
    plt.savefig('logreg')
    plt.close()


if __name__ == '__main__':
    try:
        test()
    except Exception as e:
        print(e)
