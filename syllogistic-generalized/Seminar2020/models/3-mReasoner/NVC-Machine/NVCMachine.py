import ccobra

class NVCMachine(ccobra.CCobraModel):
    def __init__(self):

        super(NVCMachine, self).__init__( \
            'NVC', ['syllogistic-generalized'], ['single-choice'])

    def predict(self, item, **kwargs):
        return [['NVC']]
