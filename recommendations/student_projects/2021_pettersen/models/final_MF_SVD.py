# IMPORTS
import ccobra
import pandas as pd
from surprise import Dataset, Reader
from surprise import SVD

# Ratings
rcols = ['userId','movieId','rating']
ml_ratings_training = pd.read_csv('../data/final_py_data_training.csv', usecols=rcols)

# Convert to Surprise Ratings
reader = Reader(rating_scale=(0.5, 5))
surprise_training = Dataset.load_from_df(ml_ratings_training, reader=reader).build_full_trainset()

# Train algorithm
algo_svd = SVD(n_factors= 60, n_epochs= 25, lr_all= .007, reg_all= 0.07)
algo_svd.fit(surprise_training)


class svd_MF_model(ccobra.CCobraModel):
    def __init__(self, name='SVD_MF'):
        super(svd_MF_model, self).__init__(name, ["recommendation"], ["single-choice"])
        
    def predict(self, item, **kwargs):

        user_id = item.identifier
        movie_id = int(eval(item.task[0][0]))
        # Prediction form
        predict_form = [[user_id, movie_id, 1]]
        predict_result = algo_svd.test(predict_form)

        return round(predict_result[0].est, 3)