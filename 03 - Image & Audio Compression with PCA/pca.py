import numpy as np


def compress(X, k):
    N = X.shape[0]

    # Compute the mean vector
    x_mean = X.mean(axis=0)

    # Center the data
    X_centered = X - x_mean

    # Compute the covariance matrix
    S = X_centered @ X_centered.T / N

    # Compute the eigenvalues and eigenvectors
    eig_vals, eig_vecs = np.linalg.eigh(S)

    # Re-arrange the eigenvectors by ascending eigenvalues
    rank = np.argsort(eig_vals)[-k:][::-1]
    Uk = eig_vecs[:, rank]

    # Compress the data by projecting
    Z = Uk.T @ X_centered

    return Uk, x_mean, Z


def decode(Uk, x_mean, Z):
    return Uk @ Z + x_mean
