import ccobra
from mmodalsentential.assertion_parser import ccobra_to_assertion
from mmodalsentential.reasoner import necessary


class MentalModel(ccobra.CCobraModel):
    def __init__(self, name='MentalModel System 2'):
        super(MentalModel, self).__init__(
            name, ['modal'], ['verify'])

    def predict(self, item, **kwargs):
        task = ccobra_to_assertion(item.task[0])
        choices = ccobra_to_assertion(item.choices[0][0])
        prediction = necessary(task, choices, 2)

        # print(task)
        # print(choices)
        # print(prediction)
        # print()

        return prediction