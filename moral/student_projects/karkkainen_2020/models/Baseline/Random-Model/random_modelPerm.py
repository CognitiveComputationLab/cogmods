import numpy as np

import ccobra

class RandomModel(ccobra.CCobraModel):
    def __init__(self, name='RandomModel'):
        super(RandomModel, self).__init__(
            name, ['moral'], ['single-choice'])



    def predict(self, item, **kwargs):
        choices = ['PERMISSIBLE','IMPERMISSIBLE']
        prediction = choices[np.random.randint(-1, len(item.choices))]

        return prediction
