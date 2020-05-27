""" Transitive-Inference model implementation.
"""
import ccobra
import random
import numpy as np

class CorrectReply(ccobra.CCobraModel):
    """ TransitivityInt CCOBRA implementation.
    """
    def __init__(self, name='CorrectReply'):
        """ Initializes the TransitivityInt model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        super().__init__(name, ['spatial-relational'], ['single-choice'])

    def predict(self, item, **kwargs):
        """ Predicts weighted responses to a given syllogism.
        """ 
        # Generate and return the current prediction
        first, second = item.choices[0][0][0], item.choices[1][0][0]
        return int(self.correctReply((first, second)))

    def predictS(self, pair):
        return int(self.correctReply(pair))

    def adaptS(self, pair):
        return
