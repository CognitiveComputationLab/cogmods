import ccobra

class Matching(ccobra.CCobraModel):
    """
    This model returns True if the mood of the prediction matches the mood of the conclusion
    for the syllogisms AE1, AE2, EA1, EA2
    """
    def __init__(self, name='Matching'):
        super(Matching, self).__init__(
            name, ['syllogistic'], ['verify'])

    def predict(self, item, **kwargs):
        enc_conclusion = ccobra.syllogistic.encode_response(item.choices[0], item.task)
        if enc_conclusion in ["Eac", "Eca"]:
            return True
        return False
