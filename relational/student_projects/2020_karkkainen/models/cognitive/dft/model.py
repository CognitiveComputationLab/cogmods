"""DFT Model.

Our  model based on the DFT Architecture. Since the DFT architecture always
returns the same result and the CCOBRA values are predictable, we are able to
implement it simply with a lookup table.

Authors:
    Rabea Turon
    Yvan Satyawan <ys88@saturn.uni-freiburg.de>
"""
import ccobra
from lookupTables.single_choice import SingleChoice
from lookupTables.verification import Verification


class DynamicArchitectureModel(ccobra.CCobraModel):

    def __init__(self, name='NeuralDynamicArchitecture'):
        super(DynamicArchitectureModel, self)\
            .__init__(name, ['spatial-relational'], ['single-choice', 'verify'])

        self.single_choice = SingleChoice()
        self.verification = Verification()

    def predict(self, item, **kwargs):

        response_type = item.response_type
        choice = None

        if response_type == 'verify':
            choice = self.verification.lookup(item.task_id)

        elif response_type == 'single-choice':
            direction = self.single_choice.lookup((item.task[0][0],
                                                   item.task[1][0]))
            choice = item.choices[direction]  # select the item with the index

        return choice
