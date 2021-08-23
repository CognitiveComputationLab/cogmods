# IMPORTS
import ccobra
import pandas as pd
import numpy as np


# DATA PROCESSING

# Ratings
rcols = ['userId','movieId','rating']
ml_ratings_training = pd.read_csv('../data/final_py_data_training.csv', usecols=rcols)

# Movies, genres
mcols = ['movieId','title','genres']
ml_movie_genre = pd.read_csv('../data/final_py_movie_genre.csv', usecols=mcols)

# Genre vectors
list_genres_set = []
for i in ml_movie_genre.genres:
    temp = i.split('|')
    temp.sort()
    list_genres_set.append(temp)
ml_movie_genre.insert(loc=3, column='genres_list', value=list_genres_set)
ml_movie_genre.drop(labels='genres', axis=1, inplace=True)

# Get full list of every genre
all_genres = []
for i in ml_movie_genre.genres_list:
    for j in i:
        if j not in all_genres:
            all_genres.append(j)
all_genres.sort()
# Put 'no genres listed' at the end
all_genres.remove('(no genres listed)')
all_genres.append('(no genres listed)')

# Create binary set of genres
n_movies = ml_movie_genre.shape[0]
n_genres = len(all_genres)
bin_genres_set = [list(np.zeros(n_genres, dtype=int)) for i in range(n_movies)]
# Fill in binary vectors of genres
for i in range(n_movies):
    for j in list_genres_set[i]:
        bin_genres_set[i][all_genres.index(j)] = 1
# Add the binary generes vector to movies df
ml_movie_genre.insert(loc=3, column='genres_vector', value=bin_genres_set)


# Create Content similarity matrix (cosine)
def create_sim_matrix(list_set):
    # computing cosine similarty of a list of vectors
    n = len(list_set)
    sim_mat = np.ones((n,n))
    for i in range(n):
        for j in range(n):
            # If diagnoal
            if i == j:
                continue
            # If under diagnoal (same as above)
            elif i > j:
                sim_mat[i][j] = sim_mat[j][i]
            # Compute above diagnoal
            else:
                sim_mat[i][j] = round(
                    np.dot(list_set[i], list_set[j]) / (np.linalg.norm(list_set[i]) * np.linalg.norm(list_set[j])), 5)   
    return sim_mat

# Create unique genre vector combos
all_movie_vecs = ml_movie_genre.genres_vector.tolist()
unq_movie_vecs = []
for i in all_movie_vecs:
    if i not in unq_movie_vecs:
        unq_movie_vecs.append(i)
unq_movie_vecs.sort(reverse=True)

# Add similarity vector id to movie df
ml_movie_genre['sim_vec_id'] = ml_movie_genre['genres_vector'].map(lambda x: unq_movie_vecs.index(x))

# Create df of the similarity matrix of genres
sim_genres_matrix = pd.DataFrame(create_sim_matrix(unq_movie_vecs))

# Tuple lists of ordered vector similarity (id, score) per id
sim_df_lists = []
for i in range(sim_genres_matrix.shape[0]):
    sim_series = sim_genres_matrix.iloc[i].sort_values(ascending=False)
    sim_list = []
    for ind, val in sim_series.iteritems():
        sim_list.append((ind, val))
    sim_df_lists.append(pd.DataFrame(sim_list, columns=['ids','scores']).set_index('ids')['scores'])

# Merge movie-genres to ratings
ml_ratings_genre = ml_ratings_training.merge(ml_movie_genre, left_on='movieId', right_on='movieId', how='left')


# START OF CCOBRA PROGRAM
class Content_model(ccobra.CCobraModel):
    def __init__(self, name='Content'):
        super(Content_model, self).__init__(name, ["recommendation"], ["single-choice"])
        self.sim_k = 35

    def predict(self, item, **kwargs):
        user_id_int = int(item.identifier)
        movie_id_int = int(eval(item.task[0][0]))

        # Find id for similarity matrix
        sim_id = int(ml_movie_genre[ml_movie_genre['movieId']==movie_id_int]['sim_vec_id'].values[0])
    
        # Create df of user training ratings
        user_ratings = pd.DataFrame(ml_ratings_genre[ml_ratings_genre['userId']==user_id_int], 
                                    columns=['userId', 'movieId', 'rating', 'sim_vec_id'])
        val = user_ratings['sim_vec_id'].map(lambda x: sim_df_lists[sim_id].loc[x])
        user_ratings.insert(column='sim_score', value=val, loc=4)
        w_score = user_ratings['rating'] * user_ratings['sim_score']
        user_ratings.insert(column='w_score', value=w_score, loc=5)
            
        # Sort to get top-N
        user_ratings.sort_values(by=['sim_score','rating'], axis=0, inplace=True, ascending=[0,0])
        
        # Return weighted prediction, or average if no similar genres
        sum_sim_sc = sum(user_ratings[0:self.sim_k].sim_score)
        sum_w_sc = sum(user_ratings[0:self.sim_k].w_score)
        # Error if all sim scores are zero, return user average rating
        if sum_sim_sc == 0:
            pred_rating = round(np.mean(user_ratings.rating), 2)
        else:
            pred_rating = round(sum_w_sc / sum_sim_sc, 2)

        return pred_rating


