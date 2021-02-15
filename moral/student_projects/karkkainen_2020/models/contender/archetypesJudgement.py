# My own model, it uses the mode response of a archetype. Each archetype is formed by the combination of their continent and gender.
# For each person, the response is determined by the mode response of their corresponding archetype.
import ccobra

archetypes = {'EuropeMen': [0, 1, 1],
 'EuropeWomen': [0, 1, 1],
 'AsiaMen': [0, 1, 1],
 'AsiaWomen': [0, 1, 1],
 'AmericasMen': [1, 1, 1],
 'AmericasWomen': [1, 1, 1],
 'Oc.Men': [0, 1, 1],
 'Oc.Women': [0, 1, 1]}

dilemmas = {'Footbridge': 0,
            'Loop': 1,
            'Switch': 2}

class ArchetypeModel(ccobra.CCobraModel):
    def __init__(self, name='Archetypes'):
        super(ArchetypeModel, self).__init__(
            name, ['moral'], ['single-choice'])



    def predict(self, item, **kwargs):

        responses = archetypes[kwargs['Continent'] + kwargs['survey.gender']]
 
        prediction = responses[dilemmas[item.task[0][0]]]

        return prediction
