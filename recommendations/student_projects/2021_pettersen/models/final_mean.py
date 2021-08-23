# IMPORTS
import ccobra
import pandas as pd

# RATING DATA
ml_ratings_training = pd.read_csv('../data/final_py_data_training.csv')


class Mean_model(ccobra.CCobraModel):
    def __init__(self, name='Mean'):
        super(Mean_model, self).__init__(name, ["recommendation"], ["single-choice"])

        self.mean_rating = 0

    def pre_train(self, dataset, **kwargs):
        self.mean_rating = round(ml_ratings_training.rating.mean(),2)

        # To double check the mean value
        #print("Global mean:", self.mean_rating)

    def predict(self, item, **kwargs):

        return self.mean_rating