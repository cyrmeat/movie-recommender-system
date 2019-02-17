# COMP9417 Machine Learning and Data Mining

Assignment2: Topic 9: Recommender system using collaborative filtering 
=====
Introduction
-----
Recommender system is widely used in E-commerce websites that predicts appropriate books, movies, music, websites and news for users. Collaborative Filtering (CF) is one of the leading recommending approaches, which analyses existing rating data to compute the similarity between users or items and generates the recommendation list. In our assignment, we focus on user-based and item-based collaborative filtering method to predict on users’ movie ratings.   

Goals
-----
The goals of this assignment are:   
(1) Implementing basic function of user-based and item-based CF algorithm to predict users’ ratings on movies.    
(2) Improving user-based and item-based CF using Top-K users/movies that use the most similar k users/movies to do rating predictions.   
(3) Proving that generally top-K item-based CF performs better than user-based CF in terms of accuracy (RMSE).

Dataset
-----
In this assignment, we use ‘u.data’ which in the compress folder ‘ml-100k’ downloaded from ‘older datasets: MoiveLens 100K Dataset’. Download link is http://files.grouplens.org/datasets/movielens/ml-100k/u.data    

Organized result of running 'user_based.py'
-----
		k=5	k=10	k=20	k=50	k=100
jaccard		1.082	1.022	0.972	0.948	0.993
Euclidean	1.079	1.025	0.963	1.017	1.139
Cosine		1.274 	1.089	1.003	0.88	0.954

Organized results of running 'item_based.py'
-----
		k=5	k=10	k=20	k=50	k=100
jaccard		0.919	0.923	0.898	0.872	0.868
Euclidean	1.013	0.938	0.930	1.040	1.054
Cosine		0.890	0.925	0.935	0.945	0.920

Instructions for running python files:
-----
1.Getting dataset from google drive, link is 'https://drive.google.com/open?id=13P2zHwG0encbNiktksY34Q8HP9CP2GcD'.
2.Putting folder 'ml-100k', file 'user-based.py' and 'item-based.py' under a same folder.
3.For 'user_based.py', just run the whole file.
4.For 'item_based.py', since it is a bit slow to generate matrixs, we recommend that run different similarity function separately. Leaving line '287', line '288' or line '289' uncommented to test 'Jaccard', 'Cosine' or 'Euclidean'.


Authors
-----
Jinlong ZHANG
Yirong CHEN
Jinjie JIN

2018 S1 COMP9417

