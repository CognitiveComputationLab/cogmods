# Heuristical model for the compare benchmark. The heuristic is based on the hypothesis made by the authors in the original paper.
import numpy as np

import ccobra

class HeuristicModel(ccobra.CCobraModel):
    def __init__(self, name='Heuristic'):
        super(HeuristicModel, self).__init__(
            name, ['moral'], ['single-choice'])



    def predict(self, item, **kwargs):
        if item.task[0][0] == 'PW-RT':
            prediction = np.random.randint(1, 3)
        if item.task[0][0] == 'OB-PW':
            prediction = 1
        else:
            prediction = 2
        return prediction
