import pandas as pd
import torch


def main():
    movies = pd.read_csv('../data/movies.csv')
    ratings = pd.read_csv('../data/ratings.csv')

    ratings_data = torch.from_numpy(ratings.values)

    print(ratings.columns)
    print(ratings_data)


if __name__ == "__main__":
    main()
