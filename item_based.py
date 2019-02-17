#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : item_base.py
# @Author: Jingjie JIN, Jinlong ZHANG, Yirong CHEN
# @Date  : 2018/5/25

import random
import math
import operator
import numpy as np
from scipy.linalg.misc import norm
import csv


class CF:
    def __init__(self, train, test):
        self.trainFile = train
        self.testFile = test
        self.load_data()

    # load the training and test dataset
    def load_data(self):
        self.trainData = {}
        self.testData = {}
        for line in open(self.trainFile, 'r'):
            userId, itemId, rating, timestamp = line.strip().split()
            self.trainData.setdefault(int(userId), {})
            self.trainData[int(userId)][int(itemId)] = int(rating)
        for line in open(self.testFile, 'r'):
            userId, itemId, rating, timestamp = line.strip().split()
            self.testData.setdefault(int(userId), {})
            self.testData[int(userId)][int(itemId)] = int(rating)
        
                    
    # Jaccard similarity
    def movieSimilarity_Jaccard(self):
        train = self.trainData
        self.movieSim = {}
        self.movie_popular = {}
        for user, movies in train.items():
            for movie in movies.keys():
                self.movie_popular.setdefault(movie, 0)
                self.movie_popular[movie] += 1
        
        self.movie_count = len(self.movie_popular)
        
        for user, movies in train.items():
            for m1 in movies:
                for m2 in movies.keys():
                    if m1 == m2:
                        continue
                    self.movieSim.setdefault(m1, {})
                    self.movieSim[m1].setdefault(m2, 0)
                    self.movieSim[m1][m2] += 1
                    
        for m1, related_movies in self.movieSim.items():
            for m2, count in related_movies.items():
                # notice handling 0 vector where number of user for one movie is 0
                if self.movie_popular[m1] == 0 or self.movie_popular[m2] == 0:
                    self.movieSim[m1][m2] = 0
                else:
                    self.movieSim[m1][m2] = \
                        1.0* count / math.sqrt(self.movie_popular[m1]* self.movie_popular[m2])
    
    # Cosine-based similarity
    def movieSimilarity_cos(self):       
        train = self.trainData
        self.movieSim = {}
        movieRating = {}
                
        for user, movies in train.items():
            for m1 in movies:
                for m2 in movies.keys():
                    if m1 == m2:
                        continue
                    self.movieSim.setdefault(m1, {})
                    self.movieSim[m1].setdefault(m2, 0)
        
        for user, movies in train.items():
            for m1 in movies:
                for m2 in movies.keys():
                    if m1 == m2:
                        continue
                    movieRating.setdefault(m1,{})
                    movieRating[m1].setdefault(m2,[[],[],[]])
                    movieRating[m1][m2][0].append(movies[m1])
                    movieRating[m1][m2][1].append(movies[m2])
                    movieRating[m1][m2][2].append(user)
               
        
        
        for m1, related_movies in self.movieSim.items():
            for m2, _ in related_movies.items():
                self.movieSim[m1][m2] = \
                    sum(np.multiply(movieRating[m1][m2][0],movieRating[m1][m2][1])) \
                    / (norm(movieRating[m1][m2][0]) * norm(movieRating[m1][m2][1]))
            
    
    # Pearson Correlation Coefficient similarity (but did not use in this project)
    def movieSimilarity_pearson(self):
        train = self.trainData
        self.movieSim = {}
        movieRating = {}

        for user, movies in train.items():
            for m1 in movies:
                for m2 in movies.keys():
                    if m1 == m2:
                        continue
                    self.movieSim.setdefault(m1, {})
                    self.movieSim[m1].setdefault(m2, 0)
        
        for user, movies in train.items():
            for m1 in movies:
                for m2 in movies.keys():
                    if m1 == m2:
                        continue
                    movieRating.setdefault(m1,{})
                    movieRating[m1].setdefault(m2,[[],[],[]])
                    movieRating[m1][m2][0].append(movies[m1])
                    movieRating[m1][m2][1].append(movies[m2])
                    movieRating[m1][m2][2].append(user)        
        
        for m1, related_movies in self.movieSim.items():
            for m2, _ in related_movies.items():
                numerator = 0
                denominator_1 = 0
                denominator_2 = 0
                for user in movieRating[m1][m2][2]:
                    user_m1 = train[user][m1]
                    user_m2 = train[user][m2]
                    mean_m1 = np.mean(movieRating[m1][m2][0])
                    mean_m2 = np.mean(movieRating[m1][m2][1])
                    numerator += (user_m1 - mean_m1)*(user_m2 - mean_m2)
                    denominator_1 += (user_m1 - mean_m1)**2
                    denominator_2 += (user_m2 - mean_m2)**2
                if denominator_1 != 0 and denominator_2!=0:
                    self.movieSim[m1][m2] = \
                    numerator / (math.sqrt(denominator_1)*math.sqrt(denominator_2))
     
    # Euclidean Distance similarity
    def movieSimilarity_euclidean(self):
        train = self.trainData
        self.movieSim = {}
        movieRating = {}

        for user, movies in train.items():
            for m1 in movies:
                for m2 in movies.keys():
                    if m1 == m2:
                        continue
                    self.movieSim.setdefault(m1, {})
                    self.movieSim[m1].setdefault(m2, 0)
        
        for user, movies in train.items():
            for m1 in movies:
                for m2 in movies.keys():
                    if m1 == m2:
                        continue
                    movieRating.setdefault(m1,{})
                    movieRating[m1].setdefault(m2,[[],[],[]])
                    movieRating[m1][m2][0].append(movies[m1])
                    movieRating[m1][m2][1].append(movies[m2])
                    movieRating[m1][m2][2].append(user)
        
        for m1, related_movies in self.movieSim.items():
            for m2, _ in related_movies.items():
                dis = norm(np.array(movieRating[m1][m2][0]) - np.array(movieRating[m1][m2][1]))
                self.movieSim[m1][m2] = 1 / (1 + dis)
    
    # Jaccard Coefficient similarity (did not use in this project)
    def movieSimilarity_Tanimoto(self):
        train = self.trainData
        self.movieSim = {}
        movieRating = {}

        for user, movies in train.items():
            for m1 in movies:
                for m2 in movies.keys():
                    if m1 == m2:
                        continue
                    self.movieSim.setdefault(m1, {})
                    self.movieSim[m1].setdefault(m2, 0)
        
        for user, movies in train.items():
            for m1 in movies:
                for m2 in movies.keys():
                    if m1 == m2:
                        continue
                    movieRating.setdefault(m1,{})
                    movieRating[m1].setdefault(m2,[[],[],[]])
                    movieRating[m1][m2][0].append(movies[m1])
                    movieRating[m1][m2][1].append(movies[m2])
                    movieRating[m1][m2][2].append(user)
        
        for m1, related_movies in self.movieSim.items():
            for m2, _ in related_movies.items():
                self.movieSim[m1][m2] = \
                    sum(np.multiply(movieRating[m1][m2][0],movieRating[m1][m2][1])) \
                    / (norm(movieRating[m1][m2][0]) + norm(movieRating[m1][m2][1]) - sum(np.multiply(movieRating[m1][m2][0],movieRating[m1][m2][1])))
                
                    
    # for a target user, find top-K similary movies, recommend top-N (100) movies
    def itemBasedRecommendation(self, user, K=20, N=100):
        train = self.trainData
        rank = {}
        watched_movies = train.get(user, {})
        related_movies = set()
         
        for movie, rating in watched_movies.items():
            for related_movie, w in sorted(self.movieSim[movie].items(), key=operator.itemgetter(1), reverse=True)[:K]:
                if related_movie in watched_movies:
                    continue
                related_movies.add(related_movie)
        
        for m1 in related_movies:
            numerator = 0
            denominator = 0
            for m2, rating in watched_movies.items():
                if m1 == m2:
                    continue
                else:
                    if m1 in self.movieSim.keys():
                        if m2 in self.movieSim[m1].keys():
                            #denominator += 1
                            denominator += abs(self.movieSim[m1][m2])
                            numerator += self.movieSim[m1][m2] * rating
            rank[m1] = numerator / denominator
        
        
        return dict(sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[:N])         

    
    def rmse(self, K, N):
        test = self.testData        
        mse = 0
        num = 0
        for user in test.keys():
            error = 0
            number = 0
            test_movies = test.get(user, {})
            recommend_movies = self.itemBasedRecommendation(user,K,N)
            
            for m1 in recommend_movies.keys():
                if m1 in test_movies:
                    number += 1
                    error += (recommend_movies[m1] - test_movies[m1])**2
                    #error += abs(recommend_movies[m1] - test_movies[m1])
            if number != 0:
                mse += math.sqrt(error/number)
                #print(number)
                #mse += error/number
                num += 1
        print(K, end='     ')
        print(mse/num)
        

def print_sample(method,cf):
    print("Using %s"%method)
    for k in [5, 10, 20, 50, 100]:
        # N = 100
        cf.rmse(k,100)

def testItemBasedCF(method):
    train = 'ml-100k/u1.base'
    test = 'ml-100k/u1.test'
    cf = CF(train, test)
    if method == 'movieSimilarity_Jaccard':
        cf.movieSimilarity_Jaccard()
        print_sample("movieSimilarity_Jaccard",cf)
    elif method == 'movieSimilarity_cos':
        cf.movieSimilarity_cos()
        print_sample("movieSimilarity_cos",cf)
#    elif method == 'movieSimilarity_pearson':
#        cf.movieSimilarity_pearson()
#        print_sample("movieSimilarity_pearson",cf)
    elif method == 'movieSimilarity_euclidean':
        cf.movieSimilarity_euclidean()
        print_sample("movieSimilarity_euclidean",cf)
    elif method == 'movieSimilarity_Tanimoto':
        cf.movieSimilarity_Tanimoto()
        print_sample('movieSimilarity_Tanimoto',cf)
    

if __name__ == "__main__":
    print("Using itemCF:")
    testItemBasedCF('movieSimilarity_Jaccard')
#    testItemBasedCF('movieSimilarity_cos')
#    testItemBasedCF('movieSimilarity_euclidean')

