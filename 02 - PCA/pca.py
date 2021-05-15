import numpy as np


def compress(X, k):
    N = X.shape[0]

    # Compute the mean vector
    x_avg = X.mean(axis=0)

    # Center the data
    X_centered = X - x_avg

    # Compute the covariance matrix
    S = np.cov(X_centered, rowvar=False)

    # Compute the eigenvalues and eigenvectors
    eig_vals, eig_vecs = np.linalg.eigh(S)

    # Re-arrange the eigenvectors by ascending eigenvalues
    rank = np.argsort(eig_vals)[-k:][::-1]
    Uk = eig_vecs[:, rank]

    # Compress the data by projecting
    Z = X_centered @ Uk

    return Uk, x_avg, Z


def decode(Uk, x_avg, Z):
    return Z @ Uk.T + x_avg
