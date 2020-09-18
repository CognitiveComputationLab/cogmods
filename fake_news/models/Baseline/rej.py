""" News Item Processing model implementation.
"""
import ccobra
import random
import math

class Rej(ccobra.CCobraModel):
    """ TransitivityInt CCOBRA implementation.
    """
    def __init__(self, name='AlwaysReject', commands = []):
        """ Initializes the TransitivityInt model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        self.parameter = {}
        super().__init__(name, ['misinformation'], ['single-choice'])

    def predictS(self, item, **kwargs):
        if len(kwargs.keys()) == 1:
            kwargs = kwargs['kwargs']
        return 0
    def predict(self, item, **kwargs):
        return 'Reject'

    def adapt(self, item, target, **kwargs):
        pass

    def adaptS(self, itemPair):
        pass
        