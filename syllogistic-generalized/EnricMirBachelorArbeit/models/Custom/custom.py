""" Implementation of the most-frequent answer (MFA) model which predicts responses based on the
most-frequently selected choice from the available background (training) data.

"""

import ccobra
import numpy as np
import copy
from bidict import bidict
import random

'''
Mental Model to guess generalized syllogisms answers. Some Definitions:
Task (encoded as [Quantifier1][Quantifier2][Figure])
All: A
Most: T
Most not: D
Some: I
Some not: O
Few: B
No: E

'''

class CustomModel(ccobra.CCobraModel):

    # Conversion Functions
    @staticmethod
    def addall(s, elements):
        for e in elements:
            if not (e in s):
                s.append(e)
    def conversion_predict(self, item, **kwargs):
       reverse_first_premise = True if random.random() < self.params["reverse_first_premise"] else False
       reverse_second_premise = True if random.random() < self.params["reverse_second_premise"] else False
       proposition1 = item.task[0]
       proposition2 = item.task[1]
       premises1 = [proposition1]
       premises2 = [proposition2]

       if reverse_first_premise and random.random() < self.params[proposition1[0]]:
           premises1.append([proposition1[0], proposition1[2], proposition1[1]])
       if reverse_second_premise and random.random() < self.params[proposition2[0]]:
           premises2.append([proposition2[0], proposition2[2], proposition2[1]])

       if item.task[0][1] == item.task[1][1]:
           a = item.task[0][2]
           b = item.task[0][1]
           c = item.task[1][2]
       elif item.task[0][1] == item.task[1][2]:
           a = item.task[0][2]
           b = item.task[0][1]
           c = item.task[1][1]
       elif item.task[0][2] == item.task[1][1]:
           a = item.task[0][1]
           b = item.task[0][2]
           c = item.task[1][2]
       else:
           a = item.task[0][1]
           b = item.task[0][2]
           c = item.task[1][1]

       predictions = []

       for p1 in premises1:
           for p2 in premises2:
               if p1 == ["All", a, b]:
                   if p2 == ["All", b, c]:
                       self.addall(predictions, [["All", a, c], ["Some", a, c], ["Some", c, a]])
                   elif p2 in [["No", b, c], ["No", c, b]]:
                       self.addall(predictions, [["No", a, c], ["No", c, a], ["Some not", a, c], ["Some not", c, a]])
                   elif p2 in [["Some not", c, b], ["Few", c, b], ["Most", c, b], ["Few not", c, b], ["Most not", c, b]]:
                       self.addall(predictions, [["Some not", c, a]])

               elif p1 == ["All", b, a]:
                   if p2 == ["All", c, b]:
                       self.addall(predictions, [["All", a, c], ["Some", a, c], ["Some", c, a]])
                   elif p2 in [["All", b, c], ["Some", c, b], ["Some", b, c]]:
                       self.addall(predictions, [["Some", a, c], ["Some", c, a]])
                   elif p2 in [["No", c, b], ["No", b, c], ["Some not", b, c]]:
                       self.addall(predictions, [["Some not", a, c]])
                   elif p2 in [["Few", b, c], ["Most", b, c], ["Few not", b, c], ["Most not", b, c]]:
                       self.addall(predictions, [["Some", a, c], ["Some", c, a], ["Some not", a, c]])
                   elif p2 in [["Few", c, b], ["Most not", c, b]]:
                       self.addall(predictions, [["Few", c, a], ["Some", a, c], ["Some", c, a], ["Most not", c, a],
                                                 ["Some not", c, a]])
                   elif p2 in [["Most", c, b], ["Few not", c, b]]:
                       self.addall(predictions, [["Most", c, a], ["Some", a, c], ["Some", c, a], ["Few not", c, a],
                                                 ["Some not", c, a]])

               elif p1 == ["Some", a, b]:
                   if p2 == ["All", b, c]:
                       self.addall(predictions, [["Some", a, c], ["Some", c, a]])
                   elif p2 in [["No", b, c], ["No", c, b]]:
                       self.addall(predictions, [["Some not", a, c]])

               elif p1 == ["Some", b, a]:
                   if p2 == ["All", b, c]:
                       self.addall(predictions, [["Some", a, c], ["Some", c, a]])
                   elif p2 in [["No", c, b], ["No", b, c]]:
                       self.addall(predictions, [["Some not", a, c]])

               elif p1[0] == "No":
                   if p2 == ["All", c, b]:
                       self.addall(predictions, [["No", c, a], ["No", a, c], ["Some not", a, c], ["Some not", c, a]])
                   elif p2 == ["All", b, c] or p2[0] in ["Some", "Few", "Most", "Most not", "Few not"]:
                       self.addall(predictions, [["Some not", c, a]])

               elif p1 == ["Some not", a, b]:
                   if p2 == ["All", c, b]:
                       self.addall(predictions, [["Some not", a, c]])

               elif p1 == ["Some not", b, a]:
                   if p2 == ["All", b, c]:
                       self.addall(predictions, [["Some not", c, a]])

               elif p1 in [["Few", a, b], ["Most Not", a, b]]:
                   if p2 == ['All', b, c]:
                       self.addall(predictions, [["Few", a, c], ["Some", a, c], ["Some", c, a], ["Some not", a, c],
                                                 ["Most not", a, c]])
                   elif p2 == ['All', c, b] or p2[0] == 'No':
                       self.addall(predictions, [["Some not", a, c]])

               elif p1 in [["Few", b, a], ["Most not", b, a]]:
                   if p2 == ["All", b, c]:
                       self.addall(predictions, [["Some", a, c], ["Some", c, a], ["Some not", c, a]])
                   elif p2 in [["Most", b, c], ["Few not", b, c]]:
                       self.addall(predictions, [["Some not", c, a]])

               elif p1 in [["Most", a, b], ["Few not", a, b]]:
                   if p2 == ['All', b, c]:
                       self.addall(predictions, [["Most", a, c], ["Some", a, c], ["Some", c, a], ["Few not", a, c],
                                                 ["Some not", a, c]])
                   elif p2 == ['All', c, b] or p2[0] == "No":
                       self.addall(predictions, [["Some not", a, c]])

               elif p1 == [["Most", b, a], ["Few not", b, a]]:
                   if p2 == ["All", b, c]:
                       self.addall(predictions, [["Some", a, c], ["Some", c, a], ["Some not", c, a]])
                   elif p2 in [["Most", b, c], ["Few not", b, c]]:
                       self.addall(predictions, [["Some", a, c], ["Some", c, a]])
                   elif p2 in [["Most not", b, c], ["Few", b, c]]:
                       self.addall(predictions, [["Some not", a, c]])

       for p in predictions:
           if item.task[0][0] in p[0] or item.task[1][0] in p[0]:
               return p

       for p in predictions:
           if p[0] == "Some":
               return p

       # NVC
       if [["NVC"]] in item.choices:
           return ["NVC"]
       else:
           return random.choices(item.choices)

    # Matching Functions
    def get_conclusion_mood(self, item):
        """computes the most conservative moods of a task."""
        most_conservative_rank = max(self.mood_to_rank[item.task[0][0]], self.mood_to_rank[item.task[1][0]])
        conclusion_mood = self.rank_to_mood[most_conservative_rank]
        return conclusion_mood

    def get_conclusion_terms(self, item):
        """extracts the two elements of the premises that are used for the conclusion, aka. removes the "connection"."""
        elements = [item.task[0][1], item.task[0][2], item.task[1][1], item.task[1][2]]
        connecting_element = None
        valid = True
        for i in range(1, 3):
            for j in range(1, 3):
                if item.task[0][i] == item.task[1][j]:
                    connecting_element = item.task[1][j]
                    for removals in range(2):
                        elements.remove(connecting_element)
        if not connecting_element:
            print("Found no connecting element in task {}".format(item.task))
            valid = False
        return elements, valid

    def build_conclusion(self, conclusion_mood, elements):
        """uses the given mood and elements to build all possible conclusions according to our Matching hypothesis"""
        possible_conclusions = []
        for mood in conclusion_mood:
            possible_conclusions.append([mood, elements[0], elements[1]])
            possible_conclusions.append([mood, elements[1], elements[0]])
        return possible_conclusions

    def matching_predict(self, item, **kwargs):
        """Predict the responses based on the extension of the students of the Matching hypothesis to generalized quantifiers"""
        elements, is_valid = self.get_conclusion_terms(item)
        conclusion_mood = self.get_conclusion_mood(item)
        possible_conclusions = self.build_conclusion(conclusion_mood, elements)
        conclusion_list = []
        for poss in item.choices:
            conclusion_list.append(poss[0])
        for computed_conclusion in possible_conclusions:
            if computed_conclusion not in conclusion_list:
                possible_conclusions.remove(computed_conclusion)
        if len(possible_conclusions) == 0:
            return ['NVC']
        return random.choice(possible_conclusions)

    # PMM Functions
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

    def pmm_predict(self, item, **kwargs):
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

        for quantifier in self.axioms:
            if self.axioms[quantifier](model, 0, 2):
                if [[quantifier, syl.A, syl.C]] in item.choices:
                    return [[quantifier, syl.A, syl.C]]
                elif [[quantifier, syl.C, syl.A]] in item.choices:
                    return [[quantifier, syl.C, syl.A]]

        if [['NVC']] in item.choices:
            return [['NVC']]

    # Custom functions
    def encode_item(self, item):
        return ccobra.syllogistic_generalized.GeneralizedSyllogism(item)

    def rank_encoded_task(self, q1, q2, fig):
        if q1 == q2:
            return 0
        difficulty = 1
        if q1 not in ['A', 'E']:
            difficulty += 1
        if q2 not in ['A', 'E']:
            difficulty += 1
        if fig == '2':
            difficulty += 1
        if fig == '3':
            difficulty +=2
        return difficulty

    def __init__(self, name='Custom'):
        super(CustomModel, self).__init__(name, ['syllogistic-generalized'], ['single-choice'])

        # Conversion
        self.params = {
           "reverse_first_premise": 0.2,
           "reverse_second_premise": 0.2,
           "All": 0.4,
           "No": 0,
           "Some": 0,
           "Some not": 0.4,
           "Most": 0.4,
           "Few": 0.4,
           "Most not": 0.4,
           "Few not": 0.4
        }

        # PartNeg Rules
        self.nvc_answered = False


        # Matching
        self.mood_to_rank = {'No': 6, 'Most not': 5, 'Some not': 4, 'Some': 3, 'Few': 2, 'Most': 1, 'All': 0}
        self.rank_to_mood = {6: ['No'], 5: ['Most not'], 4: ['Some not'], 3: ['Some'], 2: ['Few'], 1: ['Most'], 0: ['All']}

        # PMM
        self.axioms = {
            'All': self.axiom_all,
            'No': self.axiom_no,
            'Some not': self.axiom_some_not,
            'Some': self.axiom_some,
            'Few': self.axiom_few,
            'Most': self.axiom_most,
            'Most not': self.axiom_few
        }

        # List ranking a question's difficulty and most frequent answer
        quantifiers = ['A', 'T', 'D', 'I', 'O', 'B', 'E']
        combinations = [x+y+z for x in quantifiers for y in quantifiers for z in ['1', '2', '3', '4']]
        self.rank = {}
        for entry in combinations:
            self.rank[entry] = self.rank_encoded_task(entry[0], entry[1], entry[2])


    def pre_train(self, dataset, **kwargs):
        """ No custom pretrained since we're using other models

        """

    def predict(self, item, **kwargs):
        """ Generate prediction based on difficulty and using other models

        """

        #Using the rules for NVC
        syl = ccobra.syllogistic_generalized.GeneralizedSyllogism(item)

        if self.nvc_answered and item.task[0][0] != 'All' and item.task[1][0] != 'All':
            return syl.decode_response('NVC')
        syl = self.encode_item(item)
        enc_choices = [syl.encode_response(x) for x in item.choices]
        difficulty = self.rank[syl.encoded_task]

        if difficulty == 0:
            # Use conversion
            return self.conversion_predict(item)
        if difficulty == 1 or difficulty == 2:
            # Use matching (3 since it delivered best results)
            return self.matching_predict(item)
        elif difficulty >= 3:
            # Use pmm
            return self.pmm_predict(item)
        else:
            # Return a random answer
            return syl.decode_response(np.random.choice(enc_choices))

    def adapt(self, item, truth, **kwargs):
        """ Just used to check if the participant has already answered nvc at least once

        """
        syl = ccobra.syllogistic_generalized.GeneralizedSyllogism(item)
        task_enc = syl.encoded_task
        true = syl.encode_response(truth)
        if true == "NVC":
            self.nvc_answered = True
