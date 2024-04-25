import torch
import matplotlib.pyplot as plt


def split_data(data, ratio):
    n = len(data)
    indices = torch.randperm(n)
    bound_index = int(n * ratio)

    xtrain = data[indices[:bound_index]]
    xtest = data[indices[bound_index:]]

    return xtrain, xtest


def mse_error(predict, target):
    diff = torch.subtract(predict, target)
    diff_squared = torch.pow(diff, 2)
    return torch.mean(diff_squared)


def visualize_data(data, labels, feature=0):
    plt.scatter(data[:, feature], labels)
    plt.xlabel(f"feature $x_{feature}$")
    plt.ylabel("label $y$")
    plt.title(f"$x_{feature}$ vs $y$")

