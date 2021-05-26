import os
import sys
import pandas as pd

user_movies_dict = {}
movie_users_dict = {}
user_movies_dict_full = {}

def pre_process_rating_data(data):
    for idx in range(data.shape[0]):
        if user_movies_dict_full.get(data['userId'][idx]) is None:
            user_movies_dict_full[data['userId'][idx]] = set()
        user_movies_dict_full[data['userId'][idx]].add(data['movieId'][idx])
        if data['rating'][idx] > 3:
            if user_movies_dict.get(data['userId'][idx]) is None:
                user_movies_dict[data['userId'][idx]] = []
            if movie_users_dict.get(data['movieId'][idx]) is None:
                movie_users_dict[data['movieId'][idx]] = []
            user_movies_dict[data['userId'][idx]].append(data['movieId'][idx])
            movie_users_dict[data['movieId'][idx]].append(data['userId'][idx])

