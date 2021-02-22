# My own model, it uses the mode response of a archetype. Each archetype is formed by a gender.
# For each person, the response is determined by the mode response of their corresponding archetype.

import ccobra

archetypes = {'munder': [1, 2, 2],
    'mover': [1, 1, 2],
    'wunder': [1, 2, 2],
    'wover': [1, 1, 2]}

dilemmas = {'OB-PW': 0,
            'PW-RT': 1,
            'RT-OB': 2}

class ArchetypeModel(ccobra.CCobraModel):
    def __init__(self, name='Archetypes'):
        super(ArchetypeModel, self).__init__(
            name, ['moral'], ['single-choice'])



    def predict(self, item, **kwargs):
        if kwargs['Alter'] < 41:
            responses = archetypes[kwargs['Geschlecht']+'under']
        else: 
            responses = archetypes[kwargs['Geschlecht']+'over']
 
        prediction = responses[dilemmas[item.task[0][0]]]

        return prediction
