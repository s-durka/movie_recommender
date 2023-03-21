'''
Movie recommender based on the kNN classification algorithm
Author: Stanisław Durka
dataset: 
http://grouplens.org/datasets/movielens/100k/
(movielens 100k dataset: Rating prediction dataset (rating scale 1-5))
'''

import pandas as pd
from scipy.sparse import csr_array
import numpy as np
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix

FILMS = 1682
USERS = 943

NEIGHBORS = 5 # how many neighbors as the parameter for kNN

test = pd.read_csv('test.csv', sep=' ', names=['user id', 'item id', 'rating', 'timestamp'])
df = pd.read_csv('./ml-100k/u.data', sep='\t', names=['user id', 'item id', 'rating', 'timestamp'])
df.drop(['timestamp'], axis=1,inplace=True)

matrix = np.zeros([USERS,FILMS])

# iterate rows of dataframe
for i in range (df.shape[0]): 
    user = df['user id'][i]
    item = df['item id'][i]
    rating = df['rating'][i]
    matrix[user-1][item-1] = rating

matrix_sparse = csr_array(matrix)

neigh = NearestNeighbors(n_neighbors=NEIGHBORS+1,metric='cosine') # using the cosine metric in order to only compare non-zero ratings 
neigh.fit(matrix_sparse)

'''
    @n - how many films to recommend;
    @neighbors - sklearn.NearestNeighbors fitted to our data;
    @X - sparse matrix of our data (movie reviews);
    The movies recommended to our user are ranked based on the following value:
    V(item) = sum_i{(rating of user_i) * (distance to user_i)}
'''
def recommend_films(user_id, n, knn, X):
    user = user_id - 1 # user indexes in array start with 0
    k_nearest = knn.kneighbors(X.getrow(user))

    ratings_dict = {} # <movie_number-1>, rating
    weights = k_nearest[0] # cosine distances 
    users = k_nearest[1] # user id's - 1
    

    prev_row = csr_array((0,0))
    for i in range(1, users.size): # users[0] is the user_id of our user minus 1
        print(users[0,i])
        row = X.getrow(users[0,i])
        # print("r1: ", row.count_nonzero())
        row = row*weights[0,i]
        # print("r2: ", row.count_nonzero())
        # print(row.shape)
        if (i > 1):
            row = row + prev_row
        prev_row = row


    '''
        now, "row" is a sparse vector of reviews and their corresponding values
        what's left is to delete items that our user has already watched, and
        sort the vector by the value of reviews
    '''
    # 'TODO:' this part uses dense arrays, ideally it should use sparse vectors like the rest of the program
    my_reviews = X.getrow(users[0,1]).todense()[0]
    # print("my reviews:\n" ,my_reviews[:10])
    total_scores = row.todense()[0]
    total_scores = np.array(list(zip(np.arange(1, total_scores.size+1), total_scores))) # total_scores[i][j], **i == ITEM ID**! (because of range(1,size+1))
    total_scores = total_scores[my_reviews == 0.]
    total_scores = total_scores[total_scores[:,1] > 0]
    #convert back to tuple:
    total_scores = [tuple(i) for i in total_scores]

    dtype= [('uid', int), ('score', float)]
    total_scores = np.array(total_scores, dtype=dtype)
    total_scores = np.sort(total_scores, order='score')[::-1]
    # print("total scores:\n", total_scores[:10])
    return total_scores[:n]['uid']

# example usage:
USER_ID = 1
TOP_N = 5

result = recommend_films(USER_ID, TOP_N, neigh, matrix_sparse)
print("Hi user", USER_ID, "! Here are top", TOP_N, "films you might like:", result)
