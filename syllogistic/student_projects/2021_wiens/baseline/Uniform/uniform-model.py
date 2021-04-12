import ccobra
import numpy as np


class UniformModel(ccobra.CCobraModel):

    def __init__(self, name='UniformModel', k=1):
        super(UniformModel, self).__init__(name, ["syllogistic"], ["verify"])

    def predict(self, item, **kwargs):
        choices = [True, False]
        return choices[np.random.randint(0, len(choices))]
