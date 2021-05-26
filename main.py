#!/usr/bin/env python

"""
User interactive interface for Movie Recommendation Tool.
"""

import os
import sys
import pandas as pd

from ds600Proj import MovieLen

def welcome_user():
    """
    display welcome message to the application user and show available commands info

    Args:
        None

    Returns:
        user name
    """
    global ds600
    while True:
        try:
            user = int(input("> Welcome to Movie Recommendation Tool! Please input your id[1, 610]?\n"))
            if user > 0 and user <= 610:
                break
        except:
            continue
    
    print("\n> Hi user {}! Here are your options:".format(user))
    print(
        """
        - search:   find movieIds that title contains the input text
        - man1:     input several favorite movies id, get recommended movies using alg1
        - man2:     input several favorite movies id, get recommended movies using alg2
        - alg1:     based on user's history taste, get recommended movies using alg1 top support
        - alg2:     based on user's history taste, get recommended movies using alg2 min distance
        - alg3:     using scikit surprise built-in SVD recommender alg to provide top n movies
        - alg4:     using scikit surprise user own KNNSP recommender alg to provide top n movies
        - user:     switch current user
        - comp:     compare the two recommendations from alg1 and alg2
        - exit:     quit app
        """
    )
    return user

def input_command():
    """
    give prompt and let user input a command

    Args:
        None

    Returns:
        command name
    """
    cmdList = ["search", "man1", "man2", "alg1", "alg2", "alg3", "alg4", "user", "comp", "exit"]
    while True:
        cmd = input("Please input command {} \n > ".format(cmdList))
        if cmd in cmdList:
            return cmd

def input_movie_ids():
    """
    let user input movie ids manually, 

    Returns:
        movie ids set
    """
    global ds600
    while True:
        try:
            moviesStr = input("Please input the several movie ids, seperated by comma[,]: ")
            movies = set()
            for item in moviesStr.split(','):
                movieId = int(item)
                if ds600.is_valid_movieId(movieId):
                    movies.add(movieId)
            if len(movies) == 0:
                continue
            print("Your valid input are: {}". format(str(movies)))
            return movies
        except:
            continue

def search_movie_titles():
    global ds600
    text = input("Please input movie title to search in dataset: \n")
    df = ds600.search_movies_from_part_name(text)
    print(df.head(10))

def print_recommendations_limit_count(num, candidates, excludes):
    global ds600
    excludes_set = set()
    if excludes is not None:
        excludes_set = set(excludes)
    for k, v in candidates.items():
        if k not in excludes_set:
            if num > 0:
                num -= 1
                print('{:6d}'.format(k), v, ds600.get_movie_title(k))
            else:
                break

def compare_candidates():
    global ds600, candidates, excludes, user

    while True:
        try:
            N = int(input("Please input top N members to compare [1, 1000]: "))
            if N < 1 or N > 1000:
                continue
            num = N
            algs = input("Please input two numbers to compare outputs of algs[eg. 1,2 1,3]: ")
            if algs not in {'1,2', '1,3', '1,4', '2,3', '2,4', '3,4', 'break'}:
                continue
            if algs == 'break':
                break
            a, b = algs.strip().split(',')
            set1 = set()
            set2 = set()
            for k, v in candidates.get(int(a)).items():
                if k in excludes:
                    continue
                set1.add(k)
                num -= 1
                if num == 0:
                    break
            num = N
            for k, v in candidates.get(int(b)).items():
                if k in excludes:
                    continue
                set2.add(k)
                num -= 1
                if num == 0:
                    break
            print("--------------------------------------------------------------------")
            print("Common movies: ", set1.intersection(set2))
            print("--------------------------------")
            print("In Alg{} recommendations, not in Alg{}'s: ".format(a, b), set1.difference(set2))
            print("--------------------------------")
            print("In Alg{} recommendations, not in Alg{}'s: ".format(b, a), set2.difference(set1))
            print("--------------------------------")
            print("Common/Diff Ratio: ", len(set1.intersection(set2))/(len(set1.difference(set2)) + len(set2.difference(set1))))
            print("--------------------------------------------------------------------")
            return
        except Exception as err:
            print(err)
            continue
    

def main():
    """
    main function
    """
    global ds600, candidates, excludes, user
    movies = set()
    ds600 = MovieLen()
    user = welcome_user()
    candidates = {}
    excludes = {}
    while True:
        cmd = input_command()
        if cmd == "man1":
            movies = input_movie_ids()
            candidates[1] = ds600.get_candidates_with_freqency_for_movies(movies)
            print_recommendations_limit_count(20, candidates[1], movies)
        elif cmd == "man2":
            movies = input_movie_ids()
            candidates[1] = ds600.get_candidates_with_freqency_for_movies(movies)
            candidates[2] = ds600.get_nearest_neighbors_with_dist_for_movies(movies)
            excludes = movies
            print_recommendations_limit_count(20, candidates[2], movies)
        elif cmd == "search":
            search_movie_titles()
        elif cmd == "alg1":
            candidates[1] = ds600.get_candidates_with_freqency_for_user(user)
            excludes = ds600.get_all_movies_user_watched(user)
            print_recommendations_limit_count(20, candidates[1], excludes)
        elif cmd == "alg2":
            excludes = ds600.get_all_movies_user_watched(user)
            candidates[2] = ds600.get_nearest_neighbors_with_dist_for_user(user)
            print_recommendations_limit_count(20, candidates[2], excludes)
        elif cmd == "alg3":
            ds600.init_supprise_alg()
            excludes = ds600.get_all_movies_user_watched(user)
            candidates[3] = ds600.get_top_n_recommendations_with_predict_rating_for_user(user)
            print_recommendations_limit_count(20, candidates[3], excludes)
        elif cmd == "alg4":
            ds600.init_supprise_own_alg()
            excludes = ds600.get_all_movies_user_watched(user)
            candidates[4] = ds600.get_top_n_recommendations_with_predict_own_rating_for_user(user)
            print_recommendations_limit_count(20, candidates[4], excludes)
        elif cmd == "user":
            user = welcome_user()
        elif cmd == "comp":
            compare_candidates()
        else:
            exit(0)


if __name__ == "__main__":
    main()