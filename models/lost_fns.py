import torch


class MSE:
    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        return torch.mean(((y_true - y_pred)**2))

    @staticmethod
    def gradient(y_true, y_pred):
        return (2/(len(y_pred)*len(y_pred[0])))*(y_pred - y_true)
