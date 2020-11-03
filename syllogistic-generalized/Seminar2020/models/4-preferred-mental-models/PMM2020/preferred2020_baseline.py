""" Random model generating predictions based on a uniform distribution.

"""

import ccobra
import numpy as np
import copy
from bidict import bidict


class PreferredModel(ccobra.CCobraModel):
    def __init__(self, name='PMM 2020 Baseline'):
        super(PreferredModel, self).__init__(name, ['syllogistic-generalized'], ['single-choice'])

        self.axioms = {
        'All': self.axiom_all,
        'Most': self.axiom_most,
        'Most not': self.axiom_few,
        'Some': self.axiom_some,
        'Some not': self.axiom_some_not,
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

        # For each quantifier
        num_a = model[:, instance_a].sum()

        if quantifier == 'Most':
            # Fill instance B ...
            for i in range(num_a, model.shape[0]):
                if model[:, instance_b].sum() == 3:
                    break
                if model[i, instance_b] == 0 and model[i, instance_a] == 0:
                    model[i, instance_b] = 1
            # Then instance C
            model = self.fill_instance_c(model, 2, instance_c, instance_b)

        if quantifier == 'Most not':
            # Fill instance B ...
            for i in range(num_a, model.shape[0]):
                if model[:, instance_b].sum() == 3:
                    break
                if model[i, instance_b] == 0 and model[i, instance_a] == 0:
                    model[i, instance_b] = 1

            model = self.fill_instance_c(model, 1, instance_c, instance_b)

        if quantifier == 'Some' or quantifier == 'Some not':
            # Fill instance B ...
            for i in range(num_a, model.shape[0]):
                if model[:, instance_b].sum() >= 2:
                    break
                if model[i, instance_b] == 0:
                    model[i, instance_b] = 1

            model = self.fill_instance_c(model, 1, instance_c, instance_b)

        if quantifier == 'No':
            for i in reversed(range(model.shape[0])):
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

        # Fill first premise
        q1, i0, i1 = syl.p1
        model = self.fill_first_premise(model, q1, instances[i0], instances[i1])

        # Fill model with second premise
        q2, i2, i3 = syl.p2
        model = self.fill_second_premise(model, q2, 1, 2, 0)

        print(syl.task)
        if syl.task[0][0] == 'Some' and syl.task[1][0] == 'Some':
            print(model.astype(int))

        for quantifier in self.axioms:
            if self.axioms[quantifier](model, 0, 2):
                print('Prediction1', quantifier, syl.A, syl.C)
                return [[quantifier, syl.A, syl.C]]

        print('No conclusion possible')
        return [['NVC']]

