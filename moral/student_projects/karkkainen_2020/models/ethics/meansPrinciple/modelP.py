# Deontological model for the permissibility benchmark. In each case a person is used as a means to save other people, making each of the situations impermissible.

import ccobra

class DeontologyModel(ccobra.CCobraModel):
    def __init__(self, name='MeansPrinciple'):
        super(DeontologyModel, self).__init__(
            name, ['moral'], ['single-choice'])



    def predict(self, item, **kwargs):
        prediction = 'IMPERMISSIBLE'
        return prediction
