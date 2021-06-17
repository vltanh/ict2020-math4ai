import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(seed=0)


def linear_regression(X, y, W_init, lr, max_nsteps):
    N, D = X.shape

    # Assign bias values
    X_ = torch.ones(N, D+1)
    X_[:, 1:] = torch.tensor(X)

    y = torch.FloatTensor(y).unsqueeze(1)

    # Setup
    model = nn.Linear(D+1, 1, bias=False)
    criterion = nn.MSELoss()
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
