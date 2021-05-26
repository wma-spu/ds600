import os
import pandas as pd
import time
import progressbar
from datetime import datetime
import nltk
from collections import defaultdict
from surprise import SVD
from surprise import SlopeOne
from surprise import Dataset
from surprise import Reader

class MovieLen:
    ratingsDF = pd.DataFrame()
    moviesDF = pd.DataFrame()
    usersSet = set()
    user_movies_dict = {}
    movie_users_dict = {}
    user_movies_dict_full = {}
    movie_users_rating_dict = {}
    surprise_top_n_dict = None
    surprise_own_top_n_dict = None

    def __init__(self):
        print("Loading data and initializing...")
        self.ratingsDF = self.read_csv_to_dataframe('ratings')
        self.moviesDF = self.read_csv_to_dataframe('movies')
        self.usersSet = set(self.ratingsDF['userId'])
        self.pre_process_rating_data()


    def read_csv_to_dataframe(self, fileName: str):
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


    def search_movies_from_part_name(self, partName: str):
        return self.moviesDF[self.moviesDF['title'].str.contains(partName, case=False)]


    def is_valid_movieId(self, movieId):
        return (movieId in self.moviesDF['movieId'])


    def is_valid_userId(self, userId):
        return (userId in self.usersSet)


    def pre_process_rating_data(self):
        for idx in range(self.ratingsDF.shape[0]):
            if self.user_movies_dict_full.get(self.ratingsDF['userId'][idx]) is None:
                self.user_movies_dict_full[self.ratingsDF['userId'][idx]] = set()
            self.user_movies_dict_full[self.ratingsDF['userId'][idx]].add(self.ratingsDF['movieId'][idx])
            if self.ratingsDF['rating'][idx] > 3:
                if self.user_movies_dict.get(self.ratingsDF['userId'][idx]) is None:
                    self.user_movies_dict[self.ratingsDF['userId'][idx]] = []
                if self.movie_users_dict.get(self.ratingsDF['movieId'][idx]) is None:
                    self.movie_users_dict[self.ratingsDF['movieId'][idx]] = []
                if self.movie_users_rating_dict.get(self.ratingsDF['movieId'][idx]) is None:
                    self.movie_users_rating_dict[self.ratingsDF['movieId'][idx]] = dict()
                self.movie_users_rating_dict[self.ratingsDF['movieId'][idx]][self.ratingsDF['userId'][idx]] = self.ratingsDF['rating'][idx]
                self.user_movies_dict[self.ratingsDF['userId'][idx]].append(self.ratingsDF['movieId'][idx])
                self.movie_users_dict[self.ratingsDF['movieId'][idx]].append(self.ratingsDF['userId'][idx])


    def get_candidates_with_freqency_for_movies(self, movieIds):
        movies = []
        for movie in movieIds:
            users = self.movie_users_dict.get(movie)
            for user in users:
                movies.extend(self.user_movies_dict.get(user))
        return dict(sorted(nltk.FreqDist(movies).items(), key = lambda x : x[1], reverse=True))

    def get_latest_n_movies_for_user(self, userA, n):
        sub = self.ratingsDF[(self.ratingsDF['userId'] == userA) & (self.ratingsDF['rating'] > 3)]
        sub = sub.sort_values(by='timestamp', ascending=False)
        return list(sub.head(n)['movieId'])

    def get_candidates_with_freqency_for_user(self, userA):
        latestMovies = self.get_latest_n_movies_for_user(userA, 20)
        return self.get_candidates_with_freqency_for_movies(latestMovies)
        
    def get_all_movies_user_watched(self, userA):
        return self.user_movies_dict_full.get(userA)

    def get_movie_title(self, movieId):
        return list(self.moviesDF[self.moviesDF['movieId'] == movieId]['title'])[0]

    def dist_between_movies(self, movieARating, movieBRating):
        comm = 0
        if (movieARating is None) or (movieBRating is None):
            return 1
        set1 = set(movieARating.keys())
        set2 = set(movieBRating.keys())
        factor = len(set1.intersection(set2))
        for k in set1.intersection(set2):
            v = movieARating.get(k)
            v2 = movieBRating.get(k)
            comm += (1 - abs(v2 - v)/factor)
        return (len(set1.union(set2)) - comm)/len(set1.union(set2))


    def get_nearest_neighbors_with_dist_for_movies(self, movieIds):
        movies = set()
        dist_to_movies = {}
        for movie in list(self.moviesDF['movieId']):
            movie1Rating = self.movie_users_rating_dict.get(movie)
            dist = []
            for movie2 in movieIds:
                movie2Rating = self.movie_users_rating_dict.get(movie2)
                dist.append(self.dist_between_movies(movie1Rating, movie2Rating))
            dist.sort()
            dist_to_movies[movie] = sum(dist[:min(len(dist), 5)])
        return dict(sorted(dist_to_movies.items(), key = lambda x : x[1]))


    def get_nearest_neighbors_with_dist_for_user(self, userA):
        latestMovies = self.get_latest_n_movies_for_user(userA, 20)
        return self.get_nearest_neighbors_with_dist_for_movies(latestMovies)

    def init_supprise_alg(self):
        if self.surprise_top_n_dict is not None:
            return
        print("First time loading and initializing surprise algorithm......")
        reader = Reader(rating_scale=(0.5, 5))
        surpriseData = Dataset.load_from_df(self.ratingsDF[['userId', 'movieId', 'rating']], reader)
        trainset = surpriseData.build_full_trainset()
        algo = SVD()
        algo.fit(trainset)
        testset = trainset.build_anti_testset()
        predictions = algo.test(testset)
        self.surprise_top_n_dict = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            self.surprise_top_n_dict[uid].append((iid, est))
        for uid, user_ratings in self.surprise_top_n_dict.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            self.surprise_top_n_dict[uid] = user_ratings


    def get_top_n_recommendations_with_predict_rating_for_user(self, userA):
        return dict(self.surprise_top_n_dict.get(userA))

    def init_supprise_own_alg(self):
        if self.surprise_own_top_n_dict is not None:
            return
        from KNNSP import KNNSP
        print("First time loading and initializing surprise KNNSP algorithm......")
        reader = Reader(rating_scale=(0.5, 5))
        surpriseData = Dataset.load_from_df(self.ratingsDF[['userId', 'movieId', 'rating']], reader)
        trainset = surpriseData.build_full_trainset()
        algo = KNNSP()
        algo.fit(trainset)
        testset = trainset.build_anti_testset()
        predictions = algo.test(testset)
        self.surprise_own_top_n_dict = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            self.surprise_own_top_n_dict[uid].append((iid, est))
        for uid, user_ratings in self.surprise_own_top_n_dict.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            self.surprise_own_top_n_dict[uid] = user_ratings


    def get_top_n_recommendations_with_predict_own_rating_for_user(self, userA):
        return dict(self.surprise_own_top_n_dict.get(userA))
        
        




