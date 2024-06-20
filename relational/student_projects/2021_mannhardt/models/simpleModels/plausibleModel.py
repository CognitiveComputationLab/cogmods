import numpy as np

import ccobra
import TC

class PlausibleModel(ccobra.CCobraModel):
    def __init__(self, name='PlausibleModel'):
        super(PlausibleModel, self).__init__(name, ["spatial-relational"], ["single-choice", "verify"])

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
            response = counterFactCons
        elif seq == 3:
            # return plausible model
            pls = kwargs['event'].split("_")[-3]
            if pls.endswith("le"):
                response = item.choices[0][0][0]
            elif pls.endswith("ri"):
                response = item.choices[1][0][0]
        return response