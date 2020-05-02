import ccobra
from mmodalsentential.assertion_parser import ccobra_to_assertion
from mmodalsentential.reasoner import necessary, possible

class MentalModel(ccobra.CCobraModel):
    def __init__(self, name='MentalModel System 1'):
        super(MentalModel, self).__init__(
            name, ['modal'], ['verify'])

    def predict(self, item, **kwargs):
        task = ccobra_to_assertion(item.task[0])
        choices = ccobra_to_assertion(item.choices[0][0])
        return necessary(task, choices)

    def pre_train(self, dataset):
        pass


    def adapt(self, item, response, **kwargs):
        pass
