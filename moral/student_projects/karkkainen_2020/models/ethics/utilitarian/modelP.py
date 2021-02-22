# Utilitarian model for the permissibility benchmark. In each case, the utilitarian thinks taking the action as the utility is higher when more lives are saved.

import ccobra

class UtilitarianModel(ccobra.CCobraModel):
    def __init__(self, name='Utilitarian'):
        super(UtilitarianModel, self).__init__(
            name, ['moral'], ['single-choice'])



    def predict(self, item, **kwargs):
        prediction = 'PERMISSIBLE'
        return prediction
