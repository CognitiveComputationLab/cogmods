"""
Modal Logic Solver, originally provided by Guerth 2019

:parameter
type = t =  K = none
            S4 = reflexive, transitive
            B = reflexive, symmetric
            T = reflexive
"""
import ccobra
from Modal_Logic.ccobra_adapter import ccobra_to_assertion
from Modal_Logic.solver import does_a_follow_from_b


class MentalModel(ccobra.CCobraModel):
    def __init__(self, name='Modal Logic', t='K'):
        name = f"Modal Logic {t}"
        super(MentalModel, self).__init__(
            name, ['modal'], ['verify'])
        self.type = t
        self.last_response = None

    def predict(self, item, **kwargs):
        task = ccobra_to_assertion(item.task[0])
        choices = ccobra_to_assertion(item.choices[0][0])
        if self.type == 'K':
            r = does_a_follow_from_b(task, choices)
        elif self.type == 'S4':
            r = does_a_follow_from_b(task, choices, ['reflexive', 'transitive'])
        elif self.type == 'B':
            r = does_a_follow_from_b(task, choices, ['reflexive', 'symmetric'])
        elif self.type == 'T':
            r = does_a_follow_from_b(task, choices, ['reflexive'])
        else:
            raise Exception

        self.last_response = r
        return r


