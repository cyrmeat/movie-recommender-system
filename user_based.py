#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : item_base.py
# @Author: Jingjie JIN, Jinlong ZHANG, Yirong CHEN
# @Date  : 2018/5/25

import random
import math
import operator
import numpy


class UserBasedCF:
    def __init__(self, train, test):
        self.trainFile = train
        self.testFile = test
        self.load_data()

    # load the training and test dataset
    def load_data(self):
        self.trainData = {}
        self.testData = {}
        self.movieList=[]
        for line in open(self.trainFile, 'r'):
            userId, itemId, rating, timestamp = line.strip().split()
            if itemId not in self.movieList:
                   self.movieList.append(itemId)
            
            self.trainData.setdefault(userId, {})
            self.trainData[userId][itemId] = rating
        for line in open(self.testFile, 'r'):
            userId, itemId, rating, timestamp = line.strip().split()
            self.testData.setdefault(userId, {})
            self.testData[userId][itemId] = rating
        
## calcate the user related matrix
    def user_ralation(self):
        train = self.trainData
        self.user_ralation = {}
        self.user_movie_count={}
        movie_users = {}
        for user, movies in train.items():
            for movie in movies.keys():
                movie_users.setdefault(movie, set())
                movie_users[movie].add(user)
        

        for movie, users in movie_users.items():
            for user1 in users:
                self.user_movie_count.setdefault(user1, 0)
                self.user_movie_count[user1] += 1
                for user2 in users:
                    if user1 != user2:
                        self.user_ralation.setdefault(user1, {})
                        self.user_ralation[user1].setdefault(user2, 0)
                        self.user_ralation[user1][user2] += 1
    ##calculate the user vector
    def user_vector(self):
        train = self.trainData
        movie_feature=self.movieList
        total_user = len(list(train.keys()))
        self.user_vector = {}
        for user,movie_rating_pair in train.items():
            temp_user_vector=[0 for _ in range(len(movie_feature))]
            for movie, rating in movie_rating_pair.items():
                movie_feature_index=movie_feature.index(movie)
                temp_user_vector[movie_feature_index]=float(rating)
            self.user_vector[user]=temp_user_vector
    
    def userSimilarity_jaccard_distance(self):
        trainData=self.trainData.copy()
        self.userSim= self.user_ralation.copy()
        for user1, related_users in self.userSim.items():
            user1_movie=set(trainData[user1].keys())
            for user2, count in related_users.items():
                user2_movie=set(trainData[user2].keys())
                user1_and_user2=user1_movie&user2_movie
                user1_or_user2=user1_movie|user2_movie
                self.userSim[user1][user2] = len(user1_and_user2)/len(user1_or_user2)

        
        ##calculate the Euclidean_distance
    def userSimilarity_Euclidean_distance(self):

        user_vector=self.user_vector.copy()
        self.userSim= self.user_ralation.copy()
        i=0
        for user1, related_users in self.userSim.items():
            i+=1
            j=0
            #print(i)
            for user2, count in related_users.items():
                #print(j)
                j+=1
                
                user1_vector=numpy.array(user_vector[user1])
                user2_vector=numpy.array(user_vector[user2])
                euclidean_distance=numpy.linalg.norm(user1_vector - user2_vector)
                self.userSim[user1][user2] = 1/(1+euclidean_distance)
    ##calculate the Cosine similarity
    def userSimilarity_Cosine_similarity(self):
        user_vector=self.user_vector.copy()
        self.userSim= self.user_ralation.copy()
        movie_feature=self.movieList
        user_vector0=[0 for _ in range(len(movie_feature))]
        ## calculate the user similarity
        i=0
        for user1, related_users in self.userSim.items():
            i+=1
            j=0
            #print(i)
            for user2, count in related_users.items():
                #print(j)
                j+=1
                
                user1_vector=numpy.array(user_vector[user1])
                user2_vector=numpy.array(user_vector[user2])
                numerator = sum(user1_vector*user2_vector)
                user1_norm2=numpy.linalg.norm(user1_vector - user_vector0)
                user2_norm2=numpy.linalg.norm(user2_vector - user_vector0)
                denominator=user1_norm2*user2_norm2
                #print(numerator/denominator)
                self.userSim[user1][user2] = numerator/denominator
                
    
    # for a target user, find top-K similary users, recommend top-N (10) movies
    def userBasedRecommendation(self, user, K=20, N=100):
        train = self.trainData
        rank = {}
        movie_time={}
 
        ranked_movies = train.get(user, {})
        #print(ranked_movies)
        for similar_users, rating in sorted(self.userSim[user].items(), key=operator.itemgetter(1), reverse = True)[:K]:
            
            for movie in train[similar_users]:
          
                if movie in ranked_movies:
                    continue
                rank.setdefault(movie, 0)
                movie_time.setdefault(movie,0)

                rank[movie] += float(train[similar_users][movie])
                movie_time[movie]+=1
        for movie in rank.keys():

            rank[movie]=rank[movie]/movie_time[movie]
            

        return dict(sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[:N])
    


   ## calculate the mse 
    def MSE(self,num_of_similar_user=20):
       ##取出test data = {user1:{movie1:rating1,movie2:rating2,movie3:rating3},user:{movie1:rating1}}
       test_data=self.testData
       mse_dict={}
       for user,movie_rating_pair in test_data.items():
           sum_error=0
           sum_movie=0
           movie_rating_test=test_data[user]
           recomend_movie_dict=self.userBasedRecommendation(user, N=num_of_similar_user)
           for movie in recomend_movie_dict.keys():
               if movie in movie_rating_test.keys():
                   prediction=float(recomend_movie_dict[movie])
                   test_rating=float(movie_rating_test[movie])
                   error = abs(prediction-test_rating)
                   sum_error+=error**2
                   sum_movie+=1
           if sum_movie>0:
               mse_dict[user]=math.sqrt(sum_error/sum_movie)
       all_mse=0
       sum_user=0
       for user in mse_dict.keys():
           all_mse+=mse_dict[user]
           sum_user+=1
       mse=all_mse/sum_user
       return mse 
           
           
         

def testUserBasedCF():
    train = 'ml-100k/u1.base'
    test = 'ml-100k/u1.test'
    cf = UserBasedCF(train, test)    
    cf.user_ralation()
    cf.user_vector()
    #cf.userSimilarity_jaccard_distance()
    cf.userSimilarity_jaccard_distance()
    for num_of_similar_user in [5,10,20,50,100]:
        mse=cf.MSE(num_of_similar_user)
        print("jaccard_distance : {}".format(mse))
    cf = UserBasedCF(train, test)    
    cf.user_ralation()
    cf.user_vector()
    #cf.userSimilarity_Euclidean_distance()
    cf.userSimilarity_Euclidean_distance()
    for num_of_similar_user in [5,10,20,50,100]:
        mse=cf.MSE(num_of_similar_user)
        print("Euclidean_distance : {}".format(mse))
    cf = UserBasedCF(train, test)    
    cf.user_ralation()
    cf.user_vector()
    #cf.userSimilarity_Cosine_distance()
    cf.userSimilarity_Cosine_similarity()
    for num_of_similar_user in [5,10,20,50,100]:
        mse=cf.MSE(num_of_similar_user)
        print("Cosine_similarity : {}".format(mse))    


if __name__ == "__main__":
    testUserBasedCF()
        
        
            
         
        
