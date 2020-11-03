""" Random model generating predictions based on a uniform distribution.

"""

import ccobra
import numpy as np
import copy
from bidict import bidict


class PreferredModel(ccobra.CCobraModel):
    def __init__(self, name='Preferred Model 2020'):
        super(PreferredModel, self).__init__(name, ['syllogistic-generalized'], ['single-choice'])

        self.axioms = {
        'All': self.axiom_all,
        'Most': self.axiom_most,
        #'Few': self.axiom_few,
        'Some': self.axiom_some,
        'Some not': self.axiom_some_not,
        #'Most not': self.axiom_few,
        #'No': self.axiom_no
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
        model[:, x].sum() == self.sharing_rows(model, x, y)) #and
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

    def fill_instance_c(self, model, max_num, instance_c, instance_b):
        for i in range(model.shape[0]-1, -1, -1):
                if model[:, instance_c].sum() == max_num:
                    break
                if model[i, instance_b] == 1:
                    model[i, instance_c] = 1
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

            model = self.fill_instance_c(model, 2, instance_c, instance_b)

        if quantifier == 'Few' or quantifier == 'Most not':
            for i in range(num_a, model.shape[0]):
                if model[:, instance_b].sum() == 3:
                    break
                if model[i, instance_b] == 0:
                    model[i, instance_b] = 1

            model = self.fill_instance_c(model, 1, instance_c, instance_b)

        if quantifier == 'Some' or quantifier == 'Some not':
            for i in range(num_a, model.shape[0]):
                if model[:, instance_b].sum() >= 2:
                    break
                if model[i, instance_b] == 0:
                    model[i, instance_b] = 1

            model = self.fill_instance_c(model, 1, instance_c, instance_b)

        if quantifier == 'No':
            for i in range(model.shape[0]):
                if model[i, instance_b] == 0:
                    model[i, instance_c] = 1
                    break

        return model

    def predict(self, item, **kwargs):
        """
        Generate random response prediction.
        """
        model = np.zeros((6, 3), dtype=bool)

        syl = ccobra.syllogistic_generalized.GeneralizedSyllogism(item)

        if syl.figure == 2:
            prs = item.task_str.split("/")
            new_task = prs[1] + "/" + prs[0]
            new_item = ccobra.Item(item.identifier, item.domain, new_task,
                                   item.response_type, item.choices_str,
                                   item.sequence_number)
            syl = ccobra.syllogistic_generalized.GeneralizedSyllogism(new_item)

        instances = bidict({syl.A: 0, syl.B: 1, syl.C: 2})

        print(instances)

        # Fill first premise
        q1, i0, i1 = syl.p1
        model = self.fill_first_premise(model, q1, instances[i0], instances[i1])

        # Fill model with second premise
        q2, i2, i3 = syl.p2
        model = self.fill_second_premise(model, q2, 1, 2, 0)

        print(syl.task)
        print(model.astype(int))

        for quantifier in self.axioms:
            print("Conclusion", instances[i0], instances[i1])
            if self.axioms[quantifier](model, 0, 2):
                print('Prediction', quantifier, syl.A, syl.C)
                return [[quantifier, syl.A, syl.C]]

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
        "All;A;B/Few;B;C"
        ,"Some;fencers;engineers/Some;campers;fencers"
        ,"Most not;waiters;poets/Some;cashiers;waiters"
    ]
    choices = [
        "All;A;C|Some;A;C|No;A;C|Some not;A;C|Most;A;C|Few;A;C"
        ,"All;engineers;campers|All;campers;engineers|Some;engineers;campers|Some;campers;engineers|No;engineers;campers|No;campers;engineers|Some not;engineers;campers|Some not;campers;engineers|Most;engineers;campers|Most;campers;engineers|Most not;engineers;campers|Most not;campers;engineers|NVC"
        ,"All;poets;cashiers|All;cashiers;poets|Some;poets;cashiers|Some;cashiers;poets|No;poets;cashiers|No;cashiers;poets|Some not;poets;cashiers|Some not;cashiers;poets|Most;poets;cashiers|Most;cashiers;poets|Most not;poets;cashiers|Most not;cashiers;poets|NVC"
    ]

    idx = 0;
    for task in tasks:
        test_item = ccobra.Item(identifier, domain, task, resp_type,
                                choices[idx], sequence_number)
        result = pm.predict(test_item)
        print("--------------------------------------")
        idx += 1

