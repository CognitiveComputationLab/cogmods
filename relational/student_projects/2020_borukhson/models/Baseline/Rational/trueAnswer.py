""" Transitive-Inference model implementation.
"""
import ccobra
import random
import numpy as np

class CorrectReply(ccobra.CCobraModel):

    def rewardedStimulus(self, stim, pair):
        return min([int(a) for a in pair]) == int(stim)    
    def sortedPair(self, pair):
        return str(min([int(a) for a in pair])), str(max([int(a) for a in pair]))
    def correctReply(self, pair):
        return str(min([int(a) for a in pair]))
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
