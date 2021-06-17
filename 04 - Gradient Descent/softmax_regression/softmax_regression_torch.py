import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
torch.manual_seed(seed=0)


def softmax_regression(X, y, W_init, lr, max_nsteps):
    N, D = X.shape
    C = y.max() + 1

    # Assign bias values
    X_ = torch.ones(N, D+1)
    X_[:, 1:] = torch.tensor(X)

    y = torch.tensor(y)

    # Setup
    model = nn.Linear(D+1, C, bias=False)
    criterion = nn.CrossEntropyLoss()
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
