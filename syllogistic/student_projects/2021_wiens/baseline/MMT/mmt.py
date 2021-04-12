import random
import ccobra


class MentalModel(ccobra.CCobraModel):
    """ This model randomly decides if reasoners search for alternative models
    """
    def __init__(self, name='Mental Model'):
        super(MentalModel, self).__init__(
            name, ['syllogistic'], ['verify'])

        self.prediction = {"AE1": [["Eac", "Eca"]],
                           "AE2": [["Eac", "Eca"], ["Oac", "Oca"], ["Oac", "NVC"]],
                           "EA1": [["Eac", "Eca"], ["Oac", "Oca"], ["NVC", "Oca"]],
                           "EA2": [["Eac", "Eca"]]}

    def predict(self, item, **kwargs):
        enc_task = ccobra.syllogistic.encode_task(item.task)
        response = random.choice(self.prediction[enc_task])
        enc_conclusion = ccobra.syllogistic.encode_response(item.choices[0], item.task)
        if enc_conclusion in response:
            return True
        return False
