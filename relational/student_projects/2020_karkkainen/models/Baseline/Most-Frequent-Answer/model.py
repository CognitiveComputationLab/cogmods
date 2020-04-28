""" Implements a most frequent answer model.

"""

import numpy as np
import ccobra
import pandas as pd


def createDict(single_choice):
    if single_choice:
        data = pd.read_csv('single-choice.csv')
        keys = data['Task-ID'].tolist()
        values = data['most_frequent_response'].tolist()
        return dict(zip(keys, values))
    else:
        data = pd.read_csv('verification.csv')
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
        self.single_choice_answers = createDict(True)
        self.verify_answers = createDict(False)

        super(MostFreqModel, self).__init__(
            name, ["spatial-relational"], ["verify", "single-choice"])

    def predict(self, item, **kwargs):
        """ Predicts the most frequent answer for the given task ID """
        
        if item.response_type == 'verify':
            return self.verify_answers[kwargs['Task-ID']]
        else:
            return [self.single_choice_answers[kwargs['Task-ID']], item.task[-1][-1], item.task[0][1]]
