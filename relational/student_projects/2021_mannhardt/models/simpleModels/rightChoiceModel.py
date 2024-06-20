import numpy as np

import ccobra
import TC

class RightChoice(ccobra.CCobraModel):
    def __init__(self, name='RightChoice'):
        super(RightChoice, self).__init__(name, ["spatial-relational"], ["single-choice", "verify"])

    def predict(self, item, **kwargs):
        seq = item.sequence_number
        response = ""
        if seq == 1:
            tc = TC.TransitiveClosure(item.task)
            tc.create_graph()
            choice1 = item.choices[0][0][0]
            choice2 = item.choices[1][0][0]
            if tc.is_valid_model(choice1):
                response = choice1
            elif tc.is_valid_model(choice2):
                response = choice2
        elif seq == 2:
            tc = TC.TransitiveClosure(model=item.task[0][0])
            counterFactCons = tc.check_counterfact(item.choices[0][0])
            return counterFactCons
        elif seq == 3:
            response = item.choices[1][0][0]
        return response
