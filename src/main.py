import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import random_split
import torch.nn.functional as F

from models.NN import NN
from models.activation_fns import ReLU
from models.lost_fns import MSE


def main():
    mnist_dataset = MNIST(root='../data/', download=True, train=True, transform=transforms.ToTensor())

    train_data, validation_data = random_split(mnist_dataset, [50000, 10000])
    img_width = 28
    img_height = 28
    classes = 10

    dimensions = (img_width * img_height, 100, 50, 20, classes)
    activations = (ReLU, ReLU, ReLU, ReLU)
    epochs = 1000
    batch_size = 128
    learning_rate = 1e-2

    x_train = torch.stack([torch.flatten(x) for x, y in train_data])
    y_train = torch.Tensor([y for x, y in train_data]).to(torch.int64)
    y_train_onehot = F.one_hot(y_train, num_classes=classes)

    x_validation = torch.stack([torch.flatten(x) for x, y in validation_data])
    y_validation = torch.Tensor([y for x, y in validation_data]).to(torch.int64)
    y_validation_onehot = F.one_hot(y_validation, num_classes=classes)

    nn = NN(dimensions, activations)
    nn.fit(data=train_data, epochs=epochs, loss_fn=MSE, lr=learning_rate, batch_size=batch_size)

    prediction_onehot = nn.predict(x_train)
    prediction = torch.argmax(prediction_onehot, dim=1)

    v_prediction_onehot = nn.predict(x_validation)
    v_prediction = torch.argmax(v_prediction_onehot, dim=1)

    print("------------------------")
    print("TRAIN")
    print("Acc:", (torch.sum(prediction == y_train) / len(prediction)).item(), "%")
    print("Loss:", (MSE.loss(prediction_onehot, y_train_onehot).item()))
    print()
    print("VALIDATION")
    print("Acc:", (torch.sum(v_prediction == y_validation) / len(v_prediction)).item(), "%")
    print("Loss:", (MSE.loss(v_prediction_onehot, y_validation_onehot).item()))
    print("------------------------")


if __name__ == "__main__":
    main()


"""
ratings_file = pd.read_csv('../data/ratings.csv')

    # user_ratings = np.stack(ratings_file.groupby(['userId'])['rating'].apply(np.array).values, axis=0)
    ratings = torch.from_numpy(ratings_file['rating'].values)

    # user_ratings = normalize(user_ratings, user_ratings.mean(), user_ratings.std())
    print(user_ratings)
    # users_rating_mean = torch()
    global_rating_mean = torch.mean(ratings)

    user_ids = ratings_file.userId.unique()
    movies_ids = ratings_file.movieId.unique()

    users_num = len(user_ids)
    movies_num = len(movies_ids)

    user_movie_matrix = torch.zeros((movies_num, users_num))
    print(user_movie_matrix)"""