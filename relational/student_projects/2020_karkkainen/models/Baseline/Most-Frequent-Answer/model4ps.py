""" Implements a most frequent answer model.

"""

import numpy as np
import ccobra
import pandas as pd


def createDict():
    data = pd.read_csv('4ps.csv')
    keys = data['Task-ID'].tolist()
    values = data['most_frequent_response'].tolist()
    return dict(zip(keys, values))



class MostFreqModel(ccobra.CCobraModel):
    """ Model producing the most frequent answer as a response.

    """

    def __init__(self, name='MostFrequentAnswer'):
        """ Initializes the random model.

        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the CCOBRA
            framework as a means for identifying the model.

        """

        self.answers = createDict()

        super(MostFreqModel, self).__init__(
            name, ["spatial-relational"], ["verify", "single-choice"])

    def predict(self, item, **kwargs):
        """ Predicts the most frequent answer for the given task ID """

        return self.answers[kwargs['Task-ID']]
