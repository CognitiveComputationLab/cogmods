""" Random model generating predictions based on a uniform distribution.

"""

import ccobra
import numpy as np
import copy
from bidict import bidict


class PreferredModel(ccobra.CCobraModel):
    def __init__(self, name='Preferred Model 2010'):
        super(PreferredModel, self).__init__(name, ['syllogistic-generalized'], ['single-choice'])

        self.axioms = {
        'All': self.axiom_all,
        'Most': self.axiom_most,
        'Few': self.axiom_few,
        'Most not': self.axiom_few,
        'Some not': self.axiom_some_not,
        'Some': self.axiom_some,
        'No': self.axiom_no
        }


    def sharing_rows(self, model, x, y, invert_row = None):
        copy_model = copy.copy(model)
        if invert_row:
            copy_model[:, invert_row] = np.invert(copy_model[:, invert_row])

        count = 0
        for row in copy_model:
            if row[x] and row[y]:
                count += 1
        return count

    def axiom_all(self, model, x, y):
        return  (model[:, x].sum() != 0 and
        model[:, y].sum() != 0 and
        model[:, x].sum() == self.sharing_rows(model, x, y))# and
        #model[:, y].sum() == self.sharing_rows(model, x, y))

    def axiom_some(self, model, x, y):
        return  (self.sharing_rows(model, x, y) != 0 and
                self.sharing_rows(model, x, y) == self.sharing_rows(model, x, y, invert_row=y))

    def axiom_no(self, model, x, y):
        return  (self.sharing_rows(model, x, y) == 0 and
                model[:, x].sum != 0 and
                model[:, y].sum != 0)

    def axiom_some_not(self, model, x, y):
        return  (0 != self.sharing_rows(model, x, y, invert_row=y) and
                self.sharing_rows(model, x, y) == self.sharing_rows(model, x, y, invert_row=y))

    def axiom_most(self, model, x, y):
        return  (self.sharing_rows(model, x, y, invert_row=y) < self.sharing_rows(model, x, y) and
                0 != self.sharing_rows(model, x, y, invert_row=y))

    def axiom_few(self, model, x, y):
        return  (self.sharing_rows(model, x, y, invert_row=y) > self.sharing_rows(model, x, y) and
                0 != self.sharing_rows(model, x, y))

    def fill_first_premise(self, model, quantifier, instance_a, instance_b):
        if quantifier == 'All':
            model[0, instance_a] = 1
            model[0, instance_b] = 1
        elif quantifier == 'Most':
            model[0:3, instance_a] = 1
            model[0:2, instance_b] = 1
        elif quantifier == 'Few' or quantifier == 'Most not':
            model[0:3, instance_a] = 1
            model[0, instance_b] = 1
        elif quantifier == 'Some' or quantifier == 'Some not':
            model[0:2, instance_a] = 1
            model[0, instance_b] = 1
            model[2, instance_b] = 1
        else:
            model[0, instance_a] = 1
            model[1, instance_b] = 1

        return model

    def fill_second_premise(self, model, quantifier, instance_b, instance_c, instance_a):
        if quantifier == 'All':
            for i in range(model.shape[0]):
                if model[i, instance_b] == 1:
                    model[i, instance_c] = 1

        num_a = model[:, instance_a].sum()

        if quantifier == 'Most':
            for i in range(num_a, model.shape[0]):
                if model[:, instance_b].sum() == 3:
                    break
                if model[i, instance_b] == 0:
                    model[i, instance_b] = 1

            #for i in range(model.shape[0]-1, -1, -1):
            for i in range(model.shape[0]):
                if model[:, instance_c].sum() == 2:
                    break
                if model[i, instance_b] == 1:
                    model[i, instance_c] = 1

        if quantifier == 'Few' or quantifier == 'Most not':
            for i in range(num_a, model.shape[0]):
                if model[:, instance_b].sum() == 3:
                    break
                if model[i, instance_b] == 0:
                    model[i, instance_b] = 1

            #for i in range(model.shape[0]-1, -1, -1):
            for i in range(model.shape[0]):
                if model[:, instance_c].sum() == 1:
                    break
                if model[i, instance_b] == 1:
                    model[i, instance_c] = 1

        if quantifier == 'Some' or quantifier == 'Some not':
            for i in range(num_a, model.shape[0]):
                if model[:, instance_b].sum() == 2:
                    break
                if model[i, instance_b] == 0:
                    model[i, instance_b] = 1

            #for i in range(model.shape[0]-1, -1, -1):
            for i in range(model.shape[0]):
                if model[i, instance_b] == 1:
                    model[i, instance_c] = 1
                    break

        if quantifier == 'No':
            
            for i in range(model.shape[0]):
                if model[i, instance_b] == 0:
                    model[i, instance_c] = 1
                    break

        return model


    def get_conclusion_direction(self, choices, syl_A, syl_C):
        choice = choices[0][0]
        if choice[1] == syl_A:
            return (syl_A, syl_C)
        else:
            return (syl_C, syl_A)

    def predict(self, item, **kwargs):
        """
        Generate random response prediction.
        """
        model = np.zeros((6, 3), dtype=bool)

        syl = ccobra.syllogistic_generalized.GeneralizedSyllogism(item)
        instances = bidict({syl.A: 0, syl.B: 1, syl.C: 2})

        # Fill model with first premise
        q1, inst1_1_str, inst1_2_str = syl.p1
        inst1_1 = instances[inst1_1_str]
        inst1_2 = instances[inst1_2_str]
        model = self.fill_first_premise(model, q1, inst1_1, inst1_2)

        # Fill model with second premise
        q2, inst2_1_str, inst2_2_str, = syl.p2
        inst2_1 = instances[inst2_1_str]
        inst2_2 = instances[inst2_2_str]
        model = self.fill_second_premise(model, q2, inst2_1, inst2_2, 0)

        print(syl.task)
        print(model.astype(int))

        # Conclusion A-C or C-A
        inst_a, inst_c = self.get_conclusion_direction(item.choices, syl.A,
                                                      syl.C)

        # Draw conclusion from list of choices
        for choice in item.choices:
            quantifier = choice[0][0]
            if self.axioms[quantifier](model, instances[inst_a], instances[inst_c]):
                print('Prediction', quantifier, inst_a, inst_c)
                return [[quantifier, inst_a, inst_c]]

        print('No conclusion possible')
        return [['NVC']]

        


if __name__ == "__main__":
    # execute only if run as a script
    pm = PreferredModel()

    identifier = "TEST"
    sequence_number = 1
    domain = "syllogistic-generalized"
    resp_type = "single-choice"

    tasks = [
        "All;A;B/Few;B;C",
        "All;A;B/Most;B;C",
        "Few;A;B/All;B;C",
        "Few;A;B/Few;B;C",
        "Few;A;B/Most;B;C",
        "Few;A;B/No;B;C",
        "Few;A;B/Some not;B;C",
        "Few;A;B/Some;B;C",
        "Most;A;B/All;B;C",
        "Most;A;B/Few;B;C",
        "Most;A;B/Most;B;C",
        "Most;A;B/No;B;C",
        "Most;A;B/Some not;B;C",
        "Most;A;B/Some;B;C",
        "No;A;B/Few;B;C",
        "No;A;B/Most;B;C",
        "Some not;A;B/Few;B;C",
        "Some not;A;B/Most;B;C",
        "Some;A;B/Few;B;C",
        "Some;A;B/Most;B;C"
        # "All;A;B/Few;B;C"
        # ,"All;A;B/Few;B;C"
        # "Some;fencers;engineers/Some;campers;fencers"
        # ,"Most not;waiters;poets/Some;cashiers;waiters"
    ]
    choices = [
        "All;A;C|Some;A;C|No;A;C|Some not;A;C|Most;A;C|Few;A;C"
        # "All;C;A|Some;C;A|No;C;A|Some not;C;A|Most;C;A|Few;C;A"
        # "All;engineers;campers|All;campers;engineers|Some;engineers;campers
        # |Some;campers;engineers|No;engineers;campers|No;campers;engineers|Some not;engineers;campers|Some not;campers;engineers|Most;engineers;campers|Most;campers;engineers|Most not;engineers;campers|Most not;campers;engineers|NVC"
        # ,"All;poets;cashiers|All;cashiers;poets|Some;poets;cashiers|Some
        # ;cashiers;poets|No;poets;cashiers|No;cashiers;poets|Some not;poets;cashiers|Some not;cashiers;poets|Most;poets;cashiers|Most;cashiers;poets|Most not;poets;cashiers|Most not;cashiers;poets|NVC"
    ]

    # single_choice = "All;A;C|Some;A;C|No;A;C|Some not;A;C|Most;A;C|Few;A;C"
    single_choice = "All;C;A|Some;C;A|No;C;A|Some not;C;A|Most;C;A|Few;C;A"

    idx = 0;
    for task in tasks:
        test_item = ccobra.Item(identifier, domain, task, resp_type,
                                single_choice, sequence_number)
        result = pm.predict(test_item)
        print("--------------------------------------")
        idx += 1

