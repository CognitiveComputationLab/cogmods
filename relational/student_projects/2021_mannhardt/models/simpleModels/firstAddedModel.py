import numpy as np

import ccobra
import TC

class FirstAddedObject(ccobra.CCobraModel):
    def __init__(self, name='FirstAddedObject'):
        super(FirstAddedObject, self).__init__(name, ["spatial-relational"], ["single-choice", "verify"])

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
            initialModel = item.task[0][0]
            retModel = ""
            # C on left side
            if initialModel[0] == "C":
                retModel = initialModel[2]+initialModel[:2]
            # C on right side
            elif initialModel[2] == "C":
                retModel = initialModel[1:]+initialModel[0]
            return retModel
        return response
