import numpy as np

import ccobra

class RandomModel(ccobra.CCobraModel):
    def __init__(self, name='RandomModel'):
        super(RandomModel, self).__init__(
            name, ['moral'], ['single-choice'])



    def predict(self, item, **kwargs):
        prediction = np.random.randint(1, 4)

        return prediction
