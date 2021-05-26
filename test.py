import os
import sys
import pandas as pd
from mvircmdalg1 import *

def read_data_file(fileName: str):
    """
    load data from the file.

    Args:
        fileName: ratings csv data file base name

    Returns:
        pandas dataframe
        if file not found return empty dataframe
    """
    fullName = "data/" + fileName + ".csv"
    if not os.path.exists(fullName):
        print("Input file({}) NOT found! ".format(fullName))
        return pd.DataFrame()
    return pd.read_csv(fullName)

# calculate two movie's distance
def dist_between_movies(movieARating, movieBRating):
    comm = 0
    if len(movieARating) < len(movieBRating):
        tmp = movieARating
        movieARating = movieBRating
        movieBRating = tmp
    for k, v in movieARating.items():
        v2 = movieBRating.get(k, 10 + v)
        comm += (1 - abs(v2 - v)/10)
    return (len(movieARating) - comm)/len(movieARating)


movie_users_dist_dict = {}
for idx in range(data.shape[0]):
    if data['rating'][idx] > 3:
        if movie_users_dist_dict.get(data['movieId'][idx]) is None:
            movie_users_dist_dict[data['movieId'][idx]] = dict()
        movie_users_dist_dict[data['movieId'][idx]][data['userId'][idx]] = data['rating'][idx]


dists_to_movie_1 = {}
movie1Rating = movie_users_dist_dict.get(1)
for movie, ratings in movie_users_dist_dict.items():
    dists_to_movie_1[movie] = dist_between_movies(movie1Rating, ratings)
    print("dist between 1 and {} is {}".format(movie, dists_to_movie_1[movie]))

dists_to_movie_1 = dict(sorted(dists_to_movie_1.items(), key=lambda x: x[0]))
dists_to_movie_1

