import numpy as np
import ccobra

class UniformModel(ccobra.CCobraModel):
    def __init__(self, name='Random'):
        super(UniformModel, self).__init__(name, ["crt"], ["single-choice"])

    def predict(self, item, **kwargs):
        return item.choices[np.random.randint(0, len(item.choices))]
