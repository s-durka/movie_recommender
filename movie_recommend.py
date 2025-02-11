'''
Movie recommender based on the kNN classification algorithm
dataset: 
http://grouplens.org/datasets/movielens/100k/
(movielens 100k dataset: Rating prediction dataset (rating scale 1-5))
'''

import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.neighbors import NearestNeighbors

FILMS = 1682
USERS = 943
NEIGHBORS = 5  # how many neighbors as the parameter for kNN

def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', names=['user id', 'item id', 'rating', 'timestamp'])
    df.drop(['timestamp'], axis=1, inplace=True)
    return df

def create_matrix(df, users, films):
    matrix = np.zeros((users, films))
    for _, row in df.iterrows():
        user, item, rating = row['user id'], row['item id'], row['rating']
        matrix[user-1][item-1] = rating
    return csr_matrix(matrix)

def recommend_films(user_id, n, knn, ratings):
    user = user_id - 1  # user indexes in array start with 0
    k_nearest = knn.kneighbors(ratings.getrow(user))

    weights = k_nearest[0]  # cosine distances 
    users = k_nearest[1]  # user id's - 1

    prev_row = csr_matrix((1, ratings.shape[1]))
    for i in range(1, users.size):  # users[0] is the user_id of our user minus 1
        row = ratings.getrow(users[0, i]) * weights[0, i]
        prev_row += row

    my_reviews = ratings.getrow(user)
    total_scores = prev_row.multiply(my_reviews == 0).tocoo()
    total_scores = [(i + 1, total_scores.data[idx]) for idx, i in enumerate(total_scores.col) if total_scores.data[idx] > 0]

    # convert total_scores back to tuple:
    total_scores = [tuple(i) for i in total_scores]
    dtype = [('uid', int), ('score', float)]
    total_scores = np.array(total_scores, dtype=dtype)
    total_scores = np.sort(total_scores, order='score')[::-1]

    return total_scores[:n]['uid']

def load_movie_names(file_path):
    # Load the u.item file
    df = pd.read_csv(file_path, sep='|', header=None, encoding='latin-1',
                     names=['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'] + [f'genre_{i}' for i in range(19)])
    
    # Create a dictionary mapping movie_id to movie_title
    movie_dict = pd.Series(df.movie_title.values, index=df.movie_id).to_dict()
    return movie_dict

def main():
    df = load_data('./ml-100k/u.data')
    movie_dict = load_movie_names('./ml-100k/u.item')

    matrix_sparse = create_matrix(df, USERS, FILMS)

    neigh = NearestNeighbors(n_neighbors=NEIGHBORS + 1, metric='cosine')
    neigh.fit(matrix_sparse)

    USER_ID = 1
    TOP_N = 5

    result = recommend_films(USER_ID, TOP_N, neigh, matrix_sparse)
    result = [movie_dict[i] for i in result]
    print(f"Hi, User {USER_ID}. Based on your other reviews, here are {TOP_N} films you might like:\n{ ',\n'.join(map(str, result)) }")

if __name__ == "__main__":
    main()
