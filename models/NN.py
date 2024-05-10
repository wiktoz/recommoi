import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class NN:
    def __init__(self, layers, activation_fns):
        self.n_layers = len(layers)
        self.loss = None
        self.loss_function = None
        self.learning_rate = None

        self.w = {}
        self.b = {}
        self.activations = {}

        for i in range(1, self.n_layers):
            self.w[i] = torch.randn(layers[i - 1], layers[i]) / torch.sqrt(torch.tensor(layers[i - 1]))
            self.b[i] = torch.zeros(layers[i])
            self.activations[i] = activation_fns[i - 1]

    def forward(self, x):
        # w(x) + b
        a = {}

        # activations: z = f(a)
        z = [x, None, None, None, None, None]

        for i in range(1, self.n_layers):
            a[i] = torch.matmul(z[i-1], self.w[i]) + self.b[i]
            z[i] = self.activations[i].forward(a[i])

        return z, a

    def backward(self, z, a, y_true):
        y_pred = z[self.n_layers - 1]
        delta = self.loss_function.gradient(y_true, y_pred) * self.activations[self.n_layers - 1].gradient(y_pred)
        dw = torch.matmul(z[self.n_layers - 2].T, delta)

        self.w[self.n_layers - 1] -= self.learning_rate * dw
        self.b[self.n_layers - 1] -= self.learning_rate * torch.mean(delta, 0)

        for l in reversed(range(1, self.n_layers - 1)):
            delta = torch.matmul(delta, self.w[l + 1].T) * self.activations[l].gradient(a[l])
            dw = torch.matmul(z[l - 1].T, delta)
            self.w[l] -= self.learning_rate * dw
            self.b[l] -= self.learning_rate * torch.mean(delta, 0)

    def fit(self, data, epochs, loss_fn, lr=0.001, batch_size=128):
        self.loss_function = loss_fn
        self.learning_rate = lr

        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

        for i in range(epochs):
            for x_batch, y_batch in data_loader:
                y_true = F.one_hot(y_batch, num_classes=10)
                z, a = self.forward(x_batch.view(len(x_batch), 784))
                self.backward(z, a, y_true)

            if (i + 1) % 10 == 0:
                print("Completed", i+1)

    def predict(self, x):
        """
        :param x: (array) Containing parameters
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """
        _, a = self.forward(x)
        return a[self.n_layers - 1]
