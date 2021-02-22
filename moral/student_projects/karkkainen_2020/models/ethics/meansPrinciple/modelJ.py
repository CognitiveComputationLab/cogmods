# Deontological model for the trolley problem. The deontological principle followed is the means principle.
# Only in the 'Switch' takin action is seen as moral. In the other cases the large person is used as a means, and fails the principle.
import ccobra

class DeontologyModel(ccobra.CCobraModel):
    def __init__(self, name='MeansPrinciple'):
        super(DeontologyModel, self).__init__(
            name, ['moral'], ['single-choice'])



    def predict(self, item, **kwargs):
        if item.task[0][0] == 'Switch':
            prediction = 1
        else:
            prediction = 0
        return prediction
