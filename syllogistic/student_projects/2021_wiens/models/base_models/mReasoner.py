import random
import math
import copy
import itertools


class mReasoner():
    """ mReasoner implementation based on Khemlani, S. and Johnson-Laird, P. N. (2013).
    Some functions are directly translated from the source.
    For original code see http://www.modeltheory.org/models/mreasoner/
    """

    def __init__(self):
        self.sigma = 0  # counterexample search
        self.lam = 4  # size
        self.epsilon = 0  # encoding
        self.omega = 1  # weaken

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

    def get_individuals(self, subj, obj, mood):
        """returns all individuals that must be a part of a mental model,
        all individuals that directly follow from the premise (canonical)
        and all individuals that hold in the premise (combinations)"""
        individuals = []
        if mood == "A":
            individuals.append([subj, obj])
            canonical_individuals = [[subj, obj]]
            combinations = [[subj, obj], ["-" + subj, obj], ["-" + subj, "-" + obj]]
        elif mood == "I":
            individuals.append([subj, obj])
            canonical_individuals = [[subj, obj], [subj]]
            combinations = [[subj, obj], ["-" + subj, obj], [subj, "-" + obj], ["-" + subj, "-" + obj]]
        elif mood == "E":
            individuals.extend([[subj, "-" + obj], ["-" + subj, obj]])
            canonical_individuals = [[subj, "-" + obj], ["-" + subj, obj]]
            combinations = [["-" + subj, obj], [subj, "-" + obj], ["-" + subj, "-" + obj]]
        elif mood == "O":
            individuals.append([subj, "-" + obj])
            canonical_individuals = [[subj, "-" + obj], [subj, obj], [obj]]
            combinations = [[subj, obj], ["-" + subj, obj], [subj, "-" + obj], ["-" + subj, "-" + obj]]
        return individuals, canonical_individuals, combinations

    def start_model(self, premise, size):
        """encodes the first premise of the model by randomly drawing individuals
        """
        subj = premise[1]
        obj = premise[2]
        mood = premise[0]
        individuals, canonical_individuals, combinations = self.get_individuals(subj, obj, mood)

        i = len(individuals)
        while i < size:
            if random.random() < self.epsilon:
                individuals.append(random.choice(canonical_individuals))
            else:
                individuals.append(random.choice(combinations))
            i += 1
        # make sure that obj is part of O premise model
        if mood == "O" and len([x for x in individuals if obj in x]) <= 0:
            individuals.pop()
            individuals.append(random.choice([[obj], [subj, obj]]))
        return individuals

    def combine(self, premise, model):
        """expands the mental model by the second premise"""
        mood = premise[0]
        subj = premise[1]
        obj = premise[2]
        j = 0
        for i in range(len(model)):
            if mood == "A":
                if "b" in model[i]:
                    model[i] = model[i] + ["c"]
            elif mood == "I":
                if subj == "b" and "b" in model[i]:
                    model[i] = model[i] + ["c"]
                    j += 1
                    if j == 2:
                        break
                elif subj == "c" and "b" in model[i]:
                    model[i] = model[i] + ["c"]
                    model.extend([["c"]])
                    break
            elif mood == "E":
                if subj == "b" and "b" in model[i]:
                    model[i] = model[i] + ["-c"]
                else:
                    model.extend([[subj, "-" + obj], [subj, "-" + obj], [subj, "-" + obj], [subj, "-" + obj]])
                    break
            elif mood == "O":
                if subj == "b" and "b" in model[i]:
                    model[i] = model[i] + ["-c"]
                    j += 1
                    if j == 2:
                        break
                elif subj == "c":
                    model.extend([["-b", "c"], ["c"]])
                    break
        return [sorted(row, key=lambda e: e[-1]) for row in model]

    def generate_size(self):
        """uses truncated poisson density to generate raondom integers"""
        size = 0
        candidate_size = 0
        while not size:
            candidate_size = self.poisson_random_number(self.lam)
            if not (candidate_size >= 0 and candidate_size <= 1):
                size = candidate_size
        return size

    def poisson_random_number(self, lam):
        u = random.random()
        p = 0
        i = 0
        while True:
            p += self.trunc_poisson_density(i, lam)
            if u < p:
                return i
            i += 1

    def trunc_poisson_density(self, n, lam):
        if n > 34:
            n = 34
        if lam > 0:
            return math.pow(lam, n) * math.exp(-lam) / math.factorial(n)
        elif lam == 0:
            if n == 0:
                return 1
            else:
                return 0
        elif lam < 0:
            raise Exception

    def build_model(self, premises):
        """returns initial mental model based on premises"""
        intensions = self.get_premises(premises)
        first = intensions[0]
        second = intensions[1]
        capacity = self.generate_size()
        model = self.start_model(first, capacity)
        model = self.combine(second, model)
        return model

    def form_initial_conclusion(self, syllogism):
        """based on the dominant mood and the figure of the premise diffrent conclusions are preferred
        >>> m = mReasoner()
        >>> m.form_initial_conclusion("AE1")
        ['Eac']
        >>> m.form_initial_conclusion("AE2")
        ['Eac', 'Eca']
        >>> m.form_initial_conclusion("EA1")
        ['Eac']
        >>> m.form_initial_conclusion("EA2")
        ['Eac', 'Eca']
        """
        dominant_mood = self.dominant_mood(syllogism)
        figure = syllogism[2]
        if figure == "1":
            term_order = ["ac"]
        elif figure == "2":
            if "O" in syllogism:
                term_order = ["ca"]
            else:
                term_order = ["ac", "ca"]
        elif figure == "3":
            if syllogism[0] == syllogism[1] or syllogism[0] == "O":
                term_order = ["ac"]
            elif syllogism[1] == "O":
                term_order = ["ca"]
            else:
                term_order = ["ac", "ca"]
        elif figure == "4":
            if syllogism[0] == syllogism[1]:
                term_order = ["ac"]
            elif dominant_mood == "O" and syllogism[0] == "O":
                term_order = ["ca"]
            elif dominant_mood == "O" and syllogism[1] == "O":
                term_order = ["ac"]
            elif syllogism[:1] == "IE" or syllogism[:1] == "EI":
                term_order = ["ac"]
            else:
                term_order = ["ac", "ca"]

        conclusions = []
        for order in term_order:
            conclusions.append(dominant_mood + order)
        return conclusions

    def dominant_mood(self, intension):
        if "O" in intension:
            dominant_mood = "O"
        elif "E" in intension:
            dominant_mood = "E"
        elif "I" in intension:
            dominant_mood = "I"
        else:
            dominant_mood = "A"
        return dominant_mood

    def check_conclusions(self, conclusions, model):
        """checks which conclusions hold in the mental model and returns them
        >>> m = mReasoner()
        >>> m.check_conclusions(['Eac'], [['a', 'b'], ['b'], ['b', 'c']])
        ['Eac']
        >>> m.check_conclusions(['Eca', 'Eac'], [['a', 'b'], ['b'], ['b', 'c']])
        ['Eca', 'Eac']
        >>> m.check_conclusions(['Oac'], [['a', 'b', 'c'], ['a', 'b'], ['b', 'c']])
        ['Oac']
        """
        valid = []
        for conclusion in conclusions:
            mood_holds = {"A": True, "I": False, "E": True, "O": False}
            subj = conclusion[1]
            obj = conclusion[2]
            individuals_with_subj = [ind for ind in model if subj in ind]
            for individual in individuals_with_subj:
                # subj without obj
                if obj not in individual:
                    mood_holds["A"] = False
                    mood_holds["O"] = True
                # subj with obj
                if obj in individual:
                    mood_holds["I"] = True
                    mood_holds["E"] = False
            # All and No Conclusions False if no Subj in model
            if len(individuals_with_subj) < 1:
                mood_holds["E"] = False
                mood_holds["A"] = False
            if mood_holds[conclusion[0]]:
                valid.append(conclusion)
            # No conclusion valid
            if len(valid) < 1:
                valid.append("NVC")
        return valid

    def valid_counter_example(self, syllogism, conclusion):
        """
        >>> m = mReasoner()
        >>> l = []
        >>> for syl in ["AE1", "AE2", "EA1", "EA2"]:
        ...    counter = m.valid_counter_example(syl, ['Eac'])
        ...    valid = True if counter is not None else False
        ...    l.append(valid)
        >>> l
        [False, True, True, True]
        """
        possibilities = self.possible_models(syllogism)
        premises = self.get_premises(syllogism)
        counter = self.find_counterexample(possibilities, conclusion[0], premises)
        return counter

    def possible_models(self, syllogism):
        """create all possibible models of size 2 based on the mood of the first premise"""
        all_possible_models = []
        premises = self.get_premises(syllogism)
        first_premise = premises[0]
        subj, obj, mood = first_premise[1], first_premise[2], first_premise[0]
        _, canonical_individuals, combinations = self.get_individuals(subj, obj, mood)
        possible_individuals = canonical_individuals + combinations

        # remove duplicates
        possible_individuals = [list(i) for i in set(map(tuple, possible_individuals))]

        subj, obj = premises[1][1], premises[1][2]
        mood = premises[1][0]
        # create all possible individual combinations of size 2
        comb = list(itertools.combinations_with_replacement(possible_individuals, 2))
        for possibility in comb:
            all_possible_models.append([sorted(row, key=lambda e: e[-1]) for row in possibility])

        return all_possible_models

    def find_counterexample(self, possibilities, conclusion, premises):
        """  finds a model that holds in both premises but refutes the initial conclusion"""
        test_models = []
        for possible_model in possibilities:
            new_models = self.add_end_term(possible_model)
            for model in new_models:
                if model not in test_models:
                    test_models.append(model)
        for model in test_models:
            if premises[0] in self.check_conclusions([premises[0]], model):
                if premises[1] in self.check_conclusions([premises[1]], model):
                    if conclusion not in self.check_conclusions([conclusion], model):
                        return model

    def add_end_term(self, model):
        """iterates over the model and finds all possible individuals with missing end terms
        returns all possibilities of a model with added end terms"""
        new_models = []
        new_model = copy.deepcopy(model)
        for element in ["a", "c"]:
            rows = []
            token = 'a' if element == 'c' else 'c'
            for i, indv in enumerate(model):
                if element not in indv and '-' + element not in indv:
                    if token in indv or '-' + token in indv:
                        rows.append(i)
            if len(rows) < 1:
                continue
            # collection of added end terms to one possibible individual
            for i in rows:
                if element == 'a':
                    new_model[i] = ['a'] + new_model[i]
                else:
                    new_model[i].append(element)
                new_models.append(new_model)
                new_model = copy.deepcopy(model)
            # try to add end terms to all possible individuals
            if len(rows) > 1:
                for i in rows:
                    if element == 'a':
                        new_model[i] = ['a'] + new_model[i]
                    else:
                        new_model[i].append(element)
                new_models.append(new_model)
        return new_models

    def system2(self, syllogism, model, conclusion, verify_target=None, weaken=None):
        """tries to refute an initial conclusion by searching for a counterexample.
        if such a counterexample is found weakens the initial conclusion and tries to
        refute it again with a counterexample"""
        # remove duplicates
        omega = self.omega if weaken is None else weaken
        weaken = random.random()
        new_model = self.valid_counter_example(syllogism, [conclusion])
        # counterexample found
        weaker_conclusion = False
        if new_model is not None:
            # weaken
            if weaken < omega:
                if conclusion[0] == "A":
                    weaker_conclusion = "I" + conclusion[1:]
                elif conclusion[0] == "E":
                    weaker_conclusion = "O" + conclusion[1:]
        # weaker conclusion exists test if it holds in alternative model
        if weaker_conclusion:
            weaker_conclusion = [weaker_conclusion]
            valid_conclusions = self.check_conclusions(weaker_conclusion, new_model)
            # belief bias: return weaker conclusion if it holds
            if verify_target in valid_conclusions:
                return valid_conclusions
            # weaker conclusion holds, try to refute it
            if valid_conclusions == ["NVC"]:
                return valid_conclusions
            new_model = self.valid_counter_example(syllogism, weaker_conclusion)
            # counterexample found for weaker conclusion
            if new_model is not None:
                return ["NVC"]
            else:
                return weaker_conclusion
        # counterexample found but do not weaken conclusion
        if not weaker_conclusion and new_model is not None:
            return ["NVC"]
        return [conclusion]

    def predict(self, syllogism, system2=None, weaken=None, verify_target=None):
        """
        >>> m = mReasoner()
        >>> m.predict("AE1", system2=0, weaken=0)
        ['Eac']
        >>> m.predict("AE1", system2=1, weaken=0)
        ['Eac']
        >>> m.predict("AE2", system2=1, weaken=0)
        ['NVC']
        >>> m.predict("AE2", system2=1, weaken=1)
        ['Oac']
        >>> m.predict("EA1", system2=1, weaken=1)
        ['NVC']
        >>> m.predict("EA2", system2=1, weaken=1)
        ['Eca']
        """
        sigma = self.sigma if system2 is None else system2
        omega = self.omega if weaken is None else weaken
        model = self.build_model(syllogism)
        initial_conclusions = self.form_initial_conclusion(syllogism)
        valid_conclusions = self.check_conclusions(initial_conclusions, model)
        # selective processing: accept conclusion if matches with the target regardless of sigma
        if verify_target in valid_conclusions:
            return valid_conclusions
        if random.random() < sigma:
            conclusions = []
            for conclusion in valid_conclusions:
                if conclusion == "NVC":
                    continue
                conclusion = self.system2(syllogism, model, conclusion, verify_target=verify_target, weaken=omega)
                if conclusion != ["NVC"]:
                    conclusions.extend(conclusion)
            valid_conclusions = conclusions
            if len(valid_conclusions) < 1:
                return ["NVC"]
        return valid_conclusions
