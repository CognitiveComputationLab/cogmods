import copy
import random


class Mental_Model():
    """ Mental model theory implementation based on P. N. JOHNSON-LAIRD 1999
    """

    def __init__(self):
        self.alternative_model_prop = 1

    def get_premises(self, syllogism):
        """takes a syllogism in string representation (eg AE1) and returns its premises
        """
        if syllogism[2] == "1":
            return [syllogism[0] + "a" + "b", syllogism[1] + "b" + "c"]
        elif syllogism[2] == "2":
            return [syllogism[0] + "b" + "a", syllogism[1] + "c" + "b"]
        elif syllogism[2] == "3":
            return [syllogism[0] + "a" + "b", syllogism[1] + "c" + "b"]
        elif syllogism[2] == "4":
            return [syllogism[0] + "b" + "a", syllogism[1] + "b" + "c"]

    def construct_mental_model(self, enc_task):
        """Returns initial mental model and set of terms that are exhaustivly represented in the model

        >>> mm = Mental_Model()
        >>> model, exhausted = mm.construct_mental_model("AE1")
        >>> model, sorted(list(exhausted))
        ([['a', 'b', '-c'], ['a', 'b', '-c'], ['c'], ['c']], ['a', 'b', 'c'])
        >>> model, exhausted = mm.construct_mental_model("AE2")
        >>> model, sorted(list(exhausted))
        ([['a', 'b'], ['a', 'b'], ['-b', 'c'], ['-b', 'c']], ['b', 'c'])
        >>> model, exhausted = mm.construct_mental_model("EA1")
        >>> model, sorted(list(exhausted))
        ([['a', '-b'], ['a', '-b'], ['b', 'c'], ['b', 'c']], ['a', 'b'])
        >>> model, exhausted = mm.construct_mental_model("EA2")
        >>> model, sorted(list(exhausted))
        ([['-a', 'b', 'c'], ['-a', 'b', 'c'], ['a'], ['a']], ['a', 'b', 'c'])
        """
        premises = self.get_premises(enc_task)
        mental_model, exhausted = self.first_premisse_encoding(premises[0])
        mental_model, exhausted = self.combine_second_premise(premises[1], mental_model, exhausted)
        return mental_model, exhausted

    def first_premisse_encoding(self, premise):
        """encodes first premise into a mental model"""
        subj = premise[1]
        obj = premise[2]

        if premise[0] == "A":
            return [[subj, obj], [subj, obj]], set(subj)
        elif premise[0] == "I":
            return [[subj, obj], [subj], [obj]], set()
        elif premise[0] == "E":
            return [[subj, "-" + obj], [subj, "-" + obj], [obj], [obj]], set([subj, obj])
        elif premise[0] == "O":
            return [[subj, "-" + obj], [subj, "-" + obj], [obj], [obj]], set()

    def combine_second_premise(self, premise, mental_model, exhausted):
        """takes a mental model and expands it with a new premise"""
        subj = premise[1]
        if premise[0] == "A":
            for i in range(len(mental_model)):
                if "b" in mental_model[i]:
                    mental_model[i].append("c")
            exhausted.add(subj)
        elif premise[0] == "I":
            for i in range(len(mental_model)):
                if "b" in mental_model[i]:
                    mental_model[i].append("c")
                    break
        elif premise[0] == "E":
            exhausted.update(("b", "c"))
            if subj == "b":
                for i in range(len(mental_model)):
                    if "b" in mental_model[i]:
                        mental_model[i].append("-c")
                mental_model.extend([["c"], ["c"]])
            else:
                mental_model.extend([["-b", "c"], ["-b", "c"]])
        elif premise[0] == "O":
            if subj == "b":
                for i in range(len(mental_model)):
                    if "b" in mental_model[i]:
                        mental_model[i].append("-c")
                mental_model.extend([["c"], ["c"]])
            else:
                mental_model.extend([["-b", "c"], ["-b", "c"]])
        return [sorted(individual, key=lambda x: x[-1]) for individual in mental_model], exhausted

    def negative_model(self, mental_model):
        """returns True if there is a negative token in the mental model representation"""
        for element in mental_model:
            for token in element:
                if token[0] == '-':
                    return True
        return False

    def find_conclusion(self, mental_model, exhausted):
        """takes a mental model and concludes what follows for the end terms in the model
        >>> mm = Mental_Model() 
        >>> mm.find_conclusion([['a', 'b', '-c'], ['a', 'b', '-c'], ['c'], ['c']], {'a', 'b', 'c'})
        ['Eac', 'Eca']
        >>> mm.find_conclusion([['a', 'b'], ['a', 'b'], ['-b', 'c'], ['-b', 'c']], {'b', 'c'})
        ['Eac', 'Eca']
        >>> mm.find_conclusion([['a', 'b'], ['a', 'b'], ['a', '-b', 'c'], ['-b', 'c']], {'b', 'c'})
        ['Oac', 'Oca']
        >>> mm.find_conclusion([['a', 'b'], ['a', 'b'], ['a', '-b', 'c'], ['a', '-b', 'c']], {'b', 'c'})
        ['Oac', 'NVC']
        >>> mm.find_conclusion([['a', '-b'], ['a', '-b'], ['b', 'c'], ['b', 'c']], {'a', 'b'})
        ['Eac', 'Eca']
        >>> mm.find_conclusion([['a', '-b', 'c'], ['a', '-b'], ['b', 'c'], ['b', 'c']], {'a', 'b'})
        ['Oac', 'Oca']
        >>> mm.find_conclusion([['a', '-b', 'c'], ['a', '-b', 'c'], ['b', 'c'], ['b', 'c']], {'a', 'b'})
        ['NVC', 'Oca']
        >>> mm.find_conclusion([['-a', 'b', 'c'], ['-a', 'b', 'c'], ['a'], ['a']], {'a', 'b', 'c'})
        ['Eac', 'Eca']
        """
        result = []
        negative_token = self.negative_model(mental_model)

        for x in ['a', 'c']:
            y = 'a' if x == 'c' else 'c'
            end = x + y
            conclusions = set(["E" + end]) if negative_token else set(["A" + end])
            for model in mental_model:
                if x in model:
                    if not negative_token:
                        if y not in model and 'A' + end in conclusions:
                            conclusions.remove('A' + end)
                        if y in model:
                            conclusions.add('I' + end)
                    else:
                        if (y in model or len(exhausted) < 2) and 'E' + end in conclusions:
                            conclusions.remove('E' + end)
                        if y not in model:
                            conclusions.add('O' + end)

            if 'A' + end in conclusions and 'I' + end in conclusions:
                conclusions.remove('I' + end)
            if 'E' + end in conclusions and 'O' + end in conclusions:
                conclusions.remove('O' + end)

            if len(conclusions) > 0:
                result.extend(list(conclusions))
            else:
                result.append('NVC')
        return result

    def breaking(self, mental_model, exhausted):
        """rule for alternative model construction
        finds an individual with two end tokens and a mid term b that is not exhausted
        breaks the individual apart

        >>> mm = Mental_Model()
        >>> mm.breaking([['a', 'b', 'c']], {'a', 'c'})
        [['a', 'b'], ['b', 'c']]
        """
        new_model = []
        if 'b' in exhausted:
            return mental_model
        for i in range(len(mental_model)):
            if len(mental_model[i]) == 3:
                new_model.append([mental_model[i][0], mental_model[i][1]])
                new_model.append([mental_model[i][1], mental_model[i][2]])
            else:
                new_model.append(mental_model[i])
        return new_model

    def add_pos_instances(self, mental_model, exhausted, conclusion):
        """ rule for alternative model construction
        tries to refute A and I conclusions by adding new individuals to the model

        >>> mm = Mental_Model()
        >>> mm.add_pos_instances([['a', 'b', 'c'], ['a', 'b', 'c']], {'a', 'b'}, 'Aac')
        [['a', 'b', 'c'], ['a', 'b', 'c'], ['c']]
        """
        if conclusion[0] != 'A':
            return mental_model
        new_model = copy.deepcopy(mental_model)
        for token in ['a', 'c']:
            if token not in exhausted and [token] not in mental_model:
                new_model.append([token])
                return new_model
        return mental_model

    def find_movable_objects(self, mental_model, exhausted, conclusion):
        """ finds exhausted end terms that are not connected"""
        first, second = None, None
        if ('a' not in exhausted and 'b' not in exhausted) or conclusion[0] != 'O':
            return
        for first_model in mental_model:
            if 'a' in first_model and len(first_model) < 3:
                first = first_model.copy()
        for second_model in mental_model:
            if 'c' in second_model and len(second_model) < 3:
                second = second_model.copy()
        if first is None or second is None:
            return
        if ('b' in second and '-b' in first) or ('-b' in second and 'b' in first):
            return
        return (first, second)

    def join_sub_model(self, mental_model, from_model, to_model):
        """ joins two individuals and returns new model"""
        new_model = [from_model + to_model]
        for model in mental_model:
            new_model.append(model)
        new_model.remove(from_model)
        new_model.remove(to_model)
        return new_model

    def move(self, mental_model, exhausted, conclusion):
        """ rule for alternative model construction
        searches for moveable individuals in mental model and joins them
        until there are no such individuals left

        >>> mm = Mental_Model()
        >>> mm.move([['a', '-b'], ['a', '-b'], ['b', '-c'], ['b', '-c'], ['c'], ['c']], {'a', 'b', 'c'}, 'Oac')
        [['a', '-b', 'c'], ['a', '-b', 'c'], ['b', '-c'], ['b', '-c']]

        """
        new_model = copy.deepcopy(mental_model)
        moveable_objects = self.find_movable_objects(mental_model, exhausted, conclusion)
        if moveable_objects is None:
            return mental_model
        while moveable_objects:
            new_model = self.join_sub_model(new_model, moveable_objects[0], moveable_objects[1])
            moveable_objects = self.find_movable_objects(new_model, exhausted, conclusion)
        return new_model

    def add_neg_instances(self, mental_model, exhausted, conclusion):
        """ rule for alternative model construction
        tries to refute E and O conclusions by adding end terms to individuals

        >>> mm = Mental_Model()
        >>> mm.add_neg_instances([['a', 'b'], ['a', 'b'], ['-b', 'c'], ['-b', 'c']], {'a'}, 'Oac')
        [['a', 'b', 'c'], ['a', 'b', 'c'], ['-b', 'c'], ['-b', 'c']]
        >>> mm.add_neg_instances([['a', 'b'], ['a', 'b'], ['-b', 'c'], ['-b', 'c']], {'a'}, 'Eac')
        [['a', 'b', 'c'], ['a', 'b'], ['-b', 'c'], ['-b', 'c']]
        """
        new_model = copy.deepcopy(mental_model)
        for element in [item for item in ['a', 'c'] if item not in exhausted]:
            inds = []
            token = 'a' if element == 'c' else 'c'
            for i, model in enumerate(mental_model):
                if element not in model and '-' + element not in model:
                    if token in model or '-' + token in model:
                        inds.append(i)
            if len(inds) < 1:
                continue
            if conclusion[0] == 'E':
                if element == 'a':
                    new_model[inds[0]] = ['a'] + new_model[inds[0]]
                else:
                    new_model[inds[0]].append(element)
            else:
                for index in inds:
                    if element == 'a':
                        new_model[index] = ['a'] + new_model[index]
                    else:
                        new_model[index].append(element)
        return new_model

    def search_alternative_model(self, mental_model, exhausted, conclusion):
        """ adds new individals, moves individuals and breaks individuals apart in order to find an
        alternative mental model representation"""
        new_model = []
        if conclusion == "NVC":
            return mental_model
        current_model = copy.deepcopy(new_model)
        if self.negative_model(mental_model):
            while not sorted(new_model) == sorted(current_model):
                current_model = copy.deepcopy(new_model)
                new_model = self.breaking(new_model, exhausted)
            new_model = self.move(new_model, exhausted, conclusion)
        else:
            new_model = self.breaking(mental_model, exhausted)

        if sorted(new_model) == sorted(current_model):
            if not self.negative_model(mental_model):
                new_model = self.add_pos_instances(mental_model, exhausted, conclusion)
            else:
                new_model = self.add_neg_instances(mental_model, exhausted, conclusion)
        return new_model

    def predict(self, syllogism, verify_target=None, amp=None):
        """ constructs mental model and draws first conclusions
        tries to find alternative models for some participants
        """
        amp = self.alternative_model_prop if amp is None else amp
        initial_model, exhausted = self.construct_mental_model(syllogism)
        conclusions = self.find_conclusion(initial_model, exhausted)
        if verify_target in conclusions:
            return conclusions
        # search for alternative model
        current_model = initial_model
        search = True
        if random.random() < amp:
            while search:
                for conclusion in conclusions:
                    new_model = self.search_alternative_model(current_model, exhausted, conclusion)
                    if sorted(new_model) == sorted(current_model):
                        new_conclusions = conclusions
                        search = False
                        break
                    new_conclusions = self.find_conclusion(new_model, exhausted)
                    if verify_target in new_conclusions:
                        return new_conclusions
                current_model = new_model
                conclusions = new_conclusions
        return conclusions
