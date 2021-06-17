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
        h_ = np.clip(h, a_min=1e-6, a_max=1-1e-6)
        loss = - (y * np.log(h_) + (1 - y) * np.log(1 - h_)).mean()

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
        history.append((W, loss))
        step += 1
        if step > max_nsteps:
            break

    return W, history
