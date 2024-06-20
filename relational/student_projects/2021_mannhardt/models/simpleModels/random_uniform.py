import numpy as np

import ccobra


class Random(ccobra.CCobraModel):
    def __init__(self, name='Random'):
        super(Random, self).__init__(name, ["spatial-relational"], ["single-choice", "verify"])

    def predict(self, item, **kwargs):
        if item.sequence_number == 2:
            return [True, False][np.random.randint(0, 2)]
        # Return a random response
        return item.choices[np.random.randint(0, len(item.choices))]
