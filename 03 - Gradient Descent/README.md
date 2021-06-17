# Gradient Descent for Linear/Logistic/Softmax Regression

## Info

|Name|Student ID|Mail|
|---|---|---|
|Vũ Lê Thế Anh|20C13002|anh.vu2020@ict.jvn.edu.vn|

## Requirements

Install the necessary libraries in `requirements.txt`

```
pip install -r requirements.txt
```

If you want to run the PyTorch version, install additionally the latest version of PyTorch.

## Folder structure

Each regression problem `<p>` (linear, logistic, or softmax) comes with a folder `<p>_regression` containing 4 files:
- `<p>_regression.py` and `<p>_regression_torch.py`: implementation of the gradient descent steps to update the weights of the model (also known as training the model), using numpy and pytorch respectively;
- `main.py`: run the model training on a list of toy datasets from sklearn to test the algorithm;
- `visualize.py`: run the model training and visualize the training progress on a 2D synthesized dataset.