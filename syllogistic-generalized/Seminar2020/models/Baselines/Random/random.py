""" Random model generating predictions based on a uniform distribution.

"""

import ccobra
import numpy as np


class RandomModel(ccobra.CCobraModel):
    def __init__(self, name='Random'):
        super(RandomModel, self).__init__(name, ['syllogistic-generalized'], ['single-choice'])

    def predict(self, item, **kwargs):
        """ Generate random response prediction.

        """

        pred_idx = np.random.randint(len(item.choices))
        return item.choices[pred_idx]
