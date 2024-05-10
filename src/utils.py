import torch
import matplotlib.pyplot as plt
import math
import numpy as np


def normalize(data, mean, std):
    return (data - mean) / std


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


def nearest_2_pow(n):
    a = int(math.log2(n))

    if 2 ** a == n:
        return n

    return 2 ** (a + 1)


def fft(p):
    n = len(p)
    n = nearest_2_pow(n)

    if n == 1:
        return p

    if len(p) < n:
        np.resize(p, n)

    p_e = p[0::2]
    p_o = p[1::2]

    y_e = fft(p_e)
    y_o = fft(p_o)

    omega = np.exp(2j * np.pi / n)
    y = np.zeros(n, dtype=complex)

    for i in range(n//2 - 1):
        y[i] = y_e[i] + (omega**i) * y_o[i]
        y[i + n//2] = y_e[i] - (omega ** i) * y_o[i]
    return y


def poly_multiply(p1, p2):
    # Calculate the minimum required length of the result
    min_length = len(p1) + len(p2) - 1

    # Pass directly to fft; the fft function will handle resizing
    fft_p1 = fft(p1)
    fft_p2 = fft(p2)

    # Element-wise multiplication of the FFT results
    fft_result = fft_p1 * fft_p2

    # Inverse FFT to get the coefficients of the result
    result = fft(fft_result)

    # Only take the real part and round to nearest integer if necessary
    # Ensure the output is of the appropriate length by trimming any additional zeros added during FFT processing
    result = np.round(result.real[:min_length])

    return result


def from_onehot(x):
    return torch.argmax(x, dim=1)
