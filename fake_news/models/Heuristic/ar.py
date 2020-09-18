""" News Item Processing model implementation.
"""
import ccobra
import random
import math

class RA(ccobra.CCobraModel):
    """ TransitivityInt CCOBRA implementation.
    """
    def __init__(self, name='Actual-Recognition', commands = []):
        """ Initializes the TransitivityInt model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        self.parameter = {}
        self.parameter['fam'] = 1
        super().__init__(name, ['misinformation'], ['single-choice'], commands)

    def predictS(self, item):
        if item.feature('Familiarity') > self.parameter['fam']:
            return item.binCorrectCategorization()
        return item.binIncorrectCategorization()

    def adapt(self, item, target, **kwargs):
        pass

    def adaptS(self, itemPair):
        pass
        
