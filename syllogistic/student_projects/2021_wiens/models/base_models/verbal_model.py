import copy


class Verbal_Model():
    """
    Verbal Models from Polk 1995
    A Verbal Model is a list of Individuals sorted by their access time.
    Every Term is represented by a tupel. The first entry is the name of the term, the second entry is its time stamp
    """

    def __init__(self):
        # Individual-differences default parameters
        self.Idp1 = "b"  # Some
        self.Idp2 = "a"  # Some not
        self.Idp3 = "a"
        self.Idp4 = "a"  # Some
        self.Idp5 = "b"
        self.Idp6 = "a"  # Some not
        self.Idp10 = "b"
        self.Idp11 = "b"
        self.Idp12 = "b"
        self.Idp13 = "c"
        self.Idp14 = "b"
        self.Idp15 = "c"
        self.Idp16 = "c"
        self.Idp17 = "c"
        self.Idp18 = "b"
        self.Idp19 = "c"
        self.Idp20 = "b"
        self.Idp21 = "c"

        self.time = 0

        self.parameters = [self.Idp3, self.Idp5, self.Idp10, self.Idp11, self.Idp12, self.Idp13, self.Idp14, self.Idp15,
                           self.Idp16, self.Idp17, self.Idp18, self.Idp19, self.Idp20, self.Idp21]

    def get_individual_diffrence_parameter(self):
        """ Only parameter that directly involve A and E premises are included to reduce complexity.
        """
        possibilities = []
        for idp3 in ["a", "b"]:
            for idp5 in ["a", "b"]:
                for i in ["VR1", "VR2", "VR3"]:
                    if i == "VR1":
                        val = ["a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a"]
                    if i == "VR2":
                        val = ["a", "b", "b", "a", "a", "a", "a", "a", "a", "a", "a", "a"]
                    if i == "VR3":
                        val = ["b", "b", "b", "c", "b", "c", "c", "c", "b", "c", "b", "c"]
                        possibilities.append([idp3, idp5] + val)
        return possibilities

    def set_individual_diffrence_parameter(self, parameterization):
        self.Idp3 = parameterization[0]
        self.Idp5 = parameterization[1]
        self.Idp10 = parameterization[2]
        self.Idp11 = parameterization[3]
        self.Idp12 = parameterization[4]
        self.Idp13 = parameterization[5]
        self.Idp14 = parameterization[6]
        self.Idp15 = parameterization[7]
        self.Idp16 = parameterization[8]
        self.Idp17 = parameterization[9]
        self.Idp18 = parameterization[10]
        self.Idp19 = parameterization[11]
        self.Idp20 = parameterization[12]
        self.Idp21 = parameterization[13]

    def get_time(self):
        self.time = self.time + 1
        return self.time

    def get_premises(self, syllogism):
        """takes a syllogism in string representation and returns its premises"""
        if syllogism[2] == "1":
            return [syllogism[0] + "a" + "b", syllogism[1] + "b" + "c"]
        elif syllogism[2] == "2":
            return [syllogism[0] + "b" + "a", syllogism[1] + "c" + "b"]
        elif syllogism[2] == "3":
            return [syllogism[0] + "a" + "b", syllogism[1] + "c" + "b"]
        elif syllogism[2] == "4":
            return [syllogism[0] + "b" + "a", syllogism[1] + "b" + "c"]

    def inital_verbal_model(self, syllogism):
        """builds initial verbal model based on premise information"""
        premises = self.get_premises(syllogism)
        t = self.time
        verbal_model, identifiers = self.initial_premise_encoding(premises[0], t)
        verbal_model, identifiers = self.encode(verbal_model, identifiers, premises[1])
        return verbal_model, identifiers

    def initial_premise_encoding(self, premise, t=0):
        """encodes first premise to a verbal model"""
        verbal_model = []
        subj = premise[1]
        obj = premise[2]
        identifiers = set(subj)

        if premise[0] == "A":
            verbal_model.append([(subj, t), (obj, t)])
        elif premise[0] == "I":
            verbal_model.extend([[(subj, t)], [(subj, t), (obj, t)]])
        elif premise[0] == "E":
            verbal_model.append([(subj, t), ("-" + obj, t)])
        elif premise[0] == "O":
            verbal_model.extend([[(subj, t), ("-" + obj, t)], [(subj, t)]])
        return verbal_model, identifiers

    def find_subj_without_obj(self, verbal_model, subj, obj):
        """returns indexes of all individuals with a given subject but without the specified object"""
        index = []
        contains_obj, contains_subj = False, False
        for i in range(len(verbal_model)):
            for element in verbal_model[i]:
                if element[0] == subj:
                    contains_subj = True
                if element[0] == obj:
                    contains_obj = True

            if contains_subj and not contains_obj:
                index.append(i)
            contains_obj, contains_subj = False, False
        return index

    def find_subj_with_obj(self, verbal_model, subj, obj):
        """retruns  indexes of all individuals with a given subject and object"""
        index = []
        contains_subj, contains_obj = False, False
        for i in range(len(verbal_model)):
            for element in verbal_model[i]:
                if element[0] == subj:
                    contains_subj = True
                if element[0] == obj:
                    contains_obj = True
            if contains_obj and contains_subj:
                index.append(i)
            contains_subj, contains_obj = False, False
        return index

    def find_term(self, verbal_model, term):
        """returns all indexes of the individuals that contain a specified term"""
        index = []
        for i in range(len(verbal_model)):
            for element in verbal_model[i]:
                if element[0] == term:
                    index.append(i)
        return index

    def subj_in_model(self, verbal_model, subj):
        """returns true if the specified subject is a member of the verbal model"""
        for individual in verbal_model:
            for element in individual:
                if element[0] == subj:
                    return True
        return False

    def change_most_recent_indivicuals(self, verbal_model, mr_i, obj, t):
        """changes individual by adding an obj to the individual.
        The changed individual is appended to the end of the verbal model"""
        if not self.find_term(verbal_model, obj):
            model = verbal_model.pop(mr_i)
            verbal_model.extend([model + [(obj, t)]])
        return verbal_model

    def encode(self, verbal_model, identifiers, premise, subj=None):
        """expands a verbal model by encoding new premisses
        """
        # extend model with new premise
        subj = premise[1] if subj is None else subj
        positive_subj = subj if len(subj) == 1 else subj[1:]
        obj = "-" + premise[2] if premise[0] == "E" or premise[0] == "O" else premise[2]
        t = self.get_time()

        if premise[0] == "A" or premise[0] == "E":
            identifiers.add(positive_subj)
            atomic_semantic = self.Idp3 if premise[0] == "A" else self.Idp5

            changable_index = self.find_subj_without_obj(verbal_model, subj, obj)

            # append obj to all individuals with subj but without obj
            copy_model = copy.deepcopy(verbal_model)
            for i in changable_index:
                model = copy_model[i]
                verbal_model.remove(model)
                verbal_model.append(model + [(obj, t)])

            # no subj found, extend model
            if not self.subj_in_model(verbal_model, subj) or atomic_semantic == "b":
                verbal_model.append([(subj, t), (obj, t)])

        elif premise[0] == "I" or premise[0] == "O":
            identifiers.add(positive_subj)
            atomic_semantic = self.Idp4 if premise[0] == "I" else self.Idp6

            changable_index = []

            if atomic_semantic == "a" or atomic_semantic == "b":
                changable_index = self.find_term(verbal_model, subj)
                if len(changable_index) > 1:
                    mr_i = changable_index[-1]
                    verbal_model = self.change_most_recent_indivicuals(verbal_model, mr_i, obj, t)

            if len(changable_index) < 1 and (atomic_semantic == "a" or atomic_semantic == "c"):
                changable_index = self.find_subj_with_obj(verbal_model, subj, obj)
                if len(changable_index) > 1:
                    mr_i = changable_index[-1]
                    verbal_model = self.change_most_recent_indivicuals(verbal_model, mr_i, obj, t)

            # append new individual if search failed or if atomic_semantic or atomic_semantic has value d
            if len(changable_index) < 1 or atomic_semantic == "d":
                verbal_model.extend([[(subj, t), (obj, t)], [(subj, t)]])
            # only change most recently accessed Individual
            else:
                mr_i = changable_index[-1]
                mr = verbal_model[mr_i]

                compound_semantic = self.Idp1 if premise[0] == "I" else self.Idp2
                not_obj = obj if obj.startswith("-") else obj[1:]

                # compound semantics of premises
                if compound_semantic == "a":
                    new_indv = [element for element in mr if element[0] != obj]
                    verbal_model.append(new_indv)
                elif compound_semantic == "b":
                    verbal_model.append([(subj, t)])
                elif compound_semantic == "c":
                    verbal_model.append([(subj, t), (not_obj, t)])
                elif compound_semantic == "d":
                    new_indv = [element if element != obj else not_obj for element in mr]
                    new_indv2 = [element for element in mr if element[0] != obj]
                    verbal_model.extend([new_indv, new_indv2])

        return verbal_model, identifiers

    def conclude(self, verbal_model, identifiers):
        """draws all valid conclusions from a verbal model
        >>> vr = Verbal_Model()
        >>> vr.conclude([], set())
        ['NVC']
        >>> vr.conclude([[('a', 2), ('c', 2)]], set('a'))
        ['Aac']
        >>> vr.conclude([[('a', 2), ('-c', 3)]], set('a'))
        ['Eac']
        >>> vr.conclude([[('a', 2), ('-c', 3)]], set('c'))
        ['NVC']
        >>> vr.conclude([[('a', 2), ('-c', 3)], [('a', 2)]], set('a'))
        ['Oac']
        """
        # remove timing information
        vm = []
        result = set()
        for element in verbal_model:
            vm.append(list(zip(*element))[0])

        for x in identifiers:
            if x == "b" or x.startswith("-"):
                continue
            y = 'a' if x == 'c' else 'c'
            # x is not in any individual of the model
            if len(self.find_term(verbal_model, x)) < 1:
                continue
            conclusions = {"A" + x + y, "E" + x + y}
            for model in vm:
                if x in model:
                    if y in model:
                        conclusions.add("I" + x + y)
                    elif y not in model:
                        conclusions.discard("A" + x + y)
                    if "-" + y in model:
                        conclusions.add("O" + x + y)
                    elif "-" + y not in model:
                        conclusions.discard("E" + x + y)

            # remove weaker conclusions
            if "E" + x + y in conclusions and "O" + x + y in conclusions:
                conclusions.remove("O" + x + y)
            if "A" + x + y in conclusions and "I" + x + y in conclusions:
                conclusions.remove("I" + x + y)

            result = result.union(conclusions)

        if len(result) < 1:
            result.add("NVC")

        return list(result)

    def get_indirect_knowledge(self, reference_property, target_proposition):
        """finds new premises to be encoded based on the individual-differences parameters the
        reference property and the target proposition"""
        tp_obj, tp_subj = target_proposition[2], target_proposition[1]
        premise = target_proposition[0]

        # refernce property is y
        if reference_property == tp_obj and premise == "A":
            if self.Idp10 == "b":
                return "A" + tp_obj + tp_subj
            elif self.Idp10 == "c":
                return "O" + tp_obj + tp_subj
        elif reference_property == tp_obj and premise == "I":
            if self.Idp11 == "b":
                return "I" + tp_obj + tp_subj
        elif reference_property == tp_obj and premise == "E":
            if self.Idp12 == "b":
                return "E" + tp_obj + tp_subj
        elif reference_property == tp_obj and premise == "O":
            if self.Idp13 == "b":
                return "I" + tp_obj + tp_subj
            elif self.Idp13 == "c":
                return "O" + tp_obj + tp_subj

        # refernce property is -x
        if reference_property == "-" + tp_subj and premise == "A":
            if self.Idp14 == "b":
                return "E" + tp_subj + tp_obj
        elif reference_property == "-" + tp_subj and premise == "I":
            if self.Idp15 == "c":
                return "I" + tp_subj + tp_obj
        elif reference_property == "-" + tp_subj and premise == "E":
            if self.Idp16 == "c":
                return "E" + tp_subj + tp_obj
        elif reference_property == "-" + tp_subj and premise == "O":
            if self.Idp17 == "c":
                return "O" + tp_subj + tp_obj

        # refernce property is -y
        if reference_property == "-" + tp_obj and premise == "A":
            if self.Idp18 == "b":
                return "E" + tp_obj + tp_subj
        elif reference_property == "-" + tp_obj and premise == "I":
            if self.Idp19 == "c":
                return "I" + tp_obj + tp_subj
        elif reference_property == "-" + tp_obj and premise == "E":
            if self.Idp20 == "b":
                return "A" + tp_obj + tp_subj
        elif reference_property == "-" + tp_obj and premise == "O":
            if self.Idp21 == "c":
                return "O" + tp_obj + tp_subj
        return False

    def reencode(self, premises, verbal_model, identifiers, conclusion):
        """reencodes verbal model. only stops if new conclusions are found or
        all indirect knowledge was extracted"""
        sorted_inv = verbal_model[::-1]
        most_recent_terms = []
        # sort terms by their acess time
        for indv in sorted_inv:
            terms = sorted(indv, key=lambda x: x[1], reverse=True)
            # extract terms
            terms = list(zip(*terms))[0]
            for term in terms:
                if term not in most_recent_terms:
                    most_recent_terms.append(term)

        for term in most_recent_terms:
            pos_term = term if len(term) == 1 else term[1]
            for target_proposition in premises:
                if pos_term in target_proposition and not term.startswith("-"):
                    verbal_model, identifiers = self.encode(verbal_model, identifiers, target_proposition, term)
                    new_conclusions = self.conclude(verbal_model, identifiers)
                    if conclusion != new_conclusions:
                        return new_conclusions

                if pos_term in target_proposition:
                    indirect_knowledge = self.get_indirect_knowledge(term, target_proposition)
                    if indirect_knowledge:
                        verbal_model, identifiers = self.encode(verbal_model, identifiers, indirect_knowledge, term)
                        new_conclusions = self.conclude(verbal_model, identifiers)
                        if conclusion != new_conclusions:
                            return new_conclusions
        return ["NVC"]

    def predict(self, syllogism, reencoding=True):
        """builds initial verbal model and draws a tentative conclusion
        if nothing followed reencodes the model"""
        verbal_model, identifiers = self.inital_verbal_model(syllogism)
        conclusions = self.conclude(verbal_model, identifiers)
        if conclusions != ["NVC"] and reencoding:
            return conclusions
        premises = self.get_premises(syllogism)
        new_conclusions = self.reencode(premises, verbal_model, identifiers, conclusions)
        return new_conclusions
