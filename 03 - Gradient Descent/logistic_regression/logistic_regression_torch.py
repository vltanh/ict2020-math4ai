import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
torch.manual_seed(seed=0)


def logistic_regression(X, y, W_init, lr, max_nsteps):
    N, D = X.shape

    # Assign bias values
    X_ = torch.ones(N, D+1)
    X_[:, 1:] = torch.tensor(X)

    y = torch.FloatTensor(y).unsqueeze(1)

    # Setup
    model = nn.Linear(D+1, 1, bias=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Log history
    history = []

    step = 1
    while True:
        h = model(X_)
        loss = criterion(h, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stop condition
        history.append((model.weight.t().detach(), loss.item()))
        step += 1

        if step > max_nsteps:
            break

    return model.weight.t().detach(), history


def test():
    from sklearn.datasets import make_blobs

    # Data
    N, C = 1000, 2
    X, y = make_blobs(n_samples=N, centers=C, n_features=2)

    # Hyperparameters
    lr = 0.1
    max_nsteps = 10000

    # Solve Softmax Regression by Gradient Descent
    W, h = logistic_regression(X, y, None, lr, max_nsteps)
    Wh, Lh = list(zip(*h))

    # Visualize Result
    print('Result:')
    print('Final loss:', Lh[-1])

    sns.lineplot(x=range(len(Lh)), y=Lh)
    plt.savefig('logreg_torch')
    plt.close()


if __name__ == '__main__':
    import traceback
    try:
        test()
    except Exception as e:
        traceback.print_exc()
