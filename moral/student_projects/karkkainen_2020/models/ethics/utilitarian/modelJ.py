# Utilitarian model for the trolley problem. In each case, the utilitarian thinks pulling the lever / pushing the man is the right thing to do, as it saves the most lives.

import ccobra

class UtilitarianModel(ccobra.CCobraModel):
    def __init__(self, name='Utilitarian'):
        super(UtilitarianModel, self).__init__(
            name, ['moral'], ['single-choice'])



    def predict(self, item, **kwargs):
        prediction = 1
        return prediction
