import numpy as np

import ccobra

mostFrequentAnswer = {
    "OB-PW" : 1,
    "PW-RT" : 1,
    "RT-OB" : 2
    }

class MFAModel(ccobra.CCobraModel):
    def __init__(self, name='MFA'):
        super(MFAModel, self).__init__(
            name, ['moral'], ['single-choice'])



    def predict(self, item, **kwargs):
        prediction = mostFrequentAnswer[item.task[0][0]]

        return prediction
