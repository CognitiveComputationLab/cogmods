# Deontological model for the trolley problem. In this case the deontological rule followed is 'Thou shalt not kill'.
# Thus in each case the moral choice is not taking action.

import ccobra

class DeontologyModel(ccobra.CCobraModel):
    def __init__(self, name='NeverKill'):
        super(DeontologyModel, self).__init__(
            name, ['moral'], ['single-choice'])



    def predict(self, item, **kwargs):
        prediction = 0
        return prediction
