import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
np.random.seed(0)


def softmax(x):
    t = np.exp(x - x.max(axis=1)[:, None])
    return t / t.sum(axis=1)[:, None]


def d_softmax(x):
    # d/dx_j softmax(x)_i = softmax(x)_i * (delta_{i, j} - softmax(x)_j)
    s = softmax(x)
    J = np.diag(s) - np.outer(s, s)
    return J


def softmax_regression(X, y, W_init, lr, max_nsteps):
    N, D = X.shape
    C = y.max() + 1

    # Assign bias values
    X_ = np.ones((N, D+1))
    X_[:, 1:] = X

    # Initialize weights
    W = W_init

    # Log history
    history = []

    step = 1
    while True:
        # Forward:
        #   h_i = softmax(- sum_{j=1}^{D} W_j * X_{i,j})
        # === Vectorize ===
        #   h = softmax(X.T W)
        h = softmax(X_ @ W)

        # Loss:
        #   L = 1/N sum_{i=1}^{N} sum_{c=1}^{C} - y^{(i)}_c * ln(h^{(i)}_c)
        loss = -np.log(
            np.take_along_axis(np.clip(h, 1e-6, 1-1e-6), y[:, None], 1)
        ).mean()

        # Gradient:
        #   dL/dW_{i,j}
        #   = - sum_{k=1}^{N} sum_{c=1}^{C} y_c [delta_{c, i} - h_i] x_j
        # === Vectorize ===
        #   dL/dW = (h - y).T X
        dW = X_.T @ (h - np.eye(C)[y]) / N

        # Update:
        #   W_j := W_j - \alpha * dL/dW_j
        # === Vectorize ===
        #   W := W - \alpha * dL/dW
        W = W - lr * dW

        # Stop condition
        history.append((W, loss))
        step += 1
        if step > max_nsteps:
            break

    return W, history
