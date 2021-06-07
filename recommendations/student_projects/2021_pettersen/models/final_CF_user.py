# IMPORTS
import ccobra
import pandas as pd
from surprise import Dataset, Reader
from surprise import KNNWithMeans

# Ratings
rcols = ['userId','movieId','rating']
ml_ratings_training = pd.read_csv('../data/final_py_data_training.csv', usecols=rcols)

# Convert to Surprise Ratings
reader = Reader(rating_scale=(0.5, 5))
surprise_training = Dataset.load_from_df(ml_ratings_training, reader=reader).build_full_trainset()

# Train algorithm
u_min_k = 5
u_max_k = 20
sim_options_user = {'name': 'pearson', 'user_based': True}
algo_user = KNNWithMeans(k = u_max_k, min_k = u_min_k, sim_options = sim_options_user)
algo_user.fit(surprise_training)


class user_CF_model(ccobra.CCobraModel):
    def __init__(self, name='User_CF'):
        super(user_CF_model, self).__init__(name, ["recommendation"], ["single-choice"])
        
    def predict(self, item, **kwargs):

        user_id = item.identifier
        movie_id = int(eval(item.task[0][0]))
        # Prediction form
        predict_form = [[user_id, movie_id, 1]]
        predict_result = algo_user.test(predict_form)

        return round(predict_result[0].est, 3)