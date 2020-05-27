""" Implements a random uniform model.

"""

import numpy as np
import ccobra

class RandomModel(ccobra.CCobraModel):
    """ Model producing randomly generated responses.

    """

    def __init__(self, name='RandomModel'):
        """ Initializes the random model.

        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the CCOBRA
            framework as a means for identifying the model.

        """
        self.a = 0

        super().__init__(
            name, ["spatial-relational"], ["verify", "single-choice"])

    def predict(self, item, **kwargs):
        """ Predicts weighted responses to a given syllogism.

        Parameters
        ----------
        task : str
            task to produce a response for.

        """
        choices = [True, False]
        if item.response_type == 'single-choice':
            choices = [item.choices[1][0][0],item.choices[0][0][0]]
        return int(choices[np.random.randint(0, len(choices))])

    def predictS(self, item):
        return 0.5

    def adaptS(self, item):
        return
