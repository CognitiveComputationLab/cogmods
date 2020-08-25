import numpy as np
import ccobra

class CRT_false(ccobra.CCobraModel):
    def __init__(self, name='AlwaysFalse'):
        super(CRT_false, self).__init__(name, ["crt"], ["single-choice"])

    def predict(self, item, **kwargs):
        # item.choices[0] = false, item.choices[1] = true
        return item.choices[0]
