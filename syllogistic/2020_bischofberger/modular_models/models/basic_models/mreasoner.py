import math
import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from modular_models.models.basic_models.interface import SyllogisticReasoningModel
from modular_models.util import sylutil


class MReasoner(SyllogisticReasoningModel):
    def __init__(self):
        SyllogisticReasoningModel.__init__(self)

        # Size of encoded models
        self.params["lambda"] = 4.0

        # Deviation from canoncality in model encoding
        self.params["epsilon"] = 0.0

        # The probability that counterexamples are searched for (= sigma in Khemlani 2016)
        self.params["System 2"] = 1.0

        # The probability that a conclusion is weakened when a counterexample is found (rather than returning NVC)
        self.params["Weaken"] = 1.0

        # Same grid as Khemlani and Johnson-Laird 2016
        self.param_grid["lambda"] = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        self.param_grid["epsilon"] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.param_grid["System 2"] = [0.0, 1.0]
        self.param_grid["Weaken"] = [0.0, 1.0]

    def draw_individual(self, mood, subj, pred, complete):
        """ Stochastically yields an individual for building the MM of a syllogistic proposition """

        if mood == "A":
            if complete:
                # All A are B -> possible inds: [a b], [-a b], [-a -b], impossible ind: [a -b]
                return random.choice([[subj, pred],
                                      ["-"+subj, pred],
                                      ["-"+subj, "-"+pred]])
            return [subj, pred]
        elif mood == "I":
            if complete:
                # Some A are B -> possible inds: [a b], [-a b], [a, -b], [-a -b]
                return random.choice([[subj, pred],
                                      ["-"+subj, pred],
                                      [subj, "-"+pred],
                                      ["-"+subj, "-"+pred]])
            return random.choice([[subj, pred],
                                      [subj]])
        elif mood == "E":
            if complete:
                # No A are B -> possible inds: [-a b], [a, -b], [-a -b], impossible ind: [a b]
                return random.choice([["-"+subj, pred],
                                      [subj, "-"+pred],
                                      ["-"+subj, "-"+pred]])
            return random.choice([["-"+subj, pred],
                                  [subj, "-"+pred]])
        elif mood == "O":
            if complete:
                # Some A are not B -> possible inds: [a b], [-a b], [a, -b], [-a -b]
                return random.choice([[subj, pred],
                                      ["-"+subj, pred],
                                      [subj, "-"+pred],
                                      ["-"+subj, "-"+pred]])
            return random.choice([[subj, pred],
                                  [subj, "-"+pred],
                                  [pred]])

    def encode(self, syllogism, size=4, deviation=0.0):
        if size < 2:
            size = 2

        mm = []
        (subj0, pred0), (subj1, pred1) = sylutil.term_order(syllogism[2])

        ### Encode first premise ###
        # individuals that must be present in the model of the 1st premise
        required_inds = {"A": [[subj0, pred0]], "I": [[subj0, pred0]], "E": [[subj0, "-"+pred0], ["-"+subj0, pred0]],
                         "O": [[subj0, "-"+pred0]]}[syllogism[0]]

        appendix = []

        # O needs an additional requirement to make sure "b" is present (otherwise e.g. [a -b] [-a -b] would be allowed)
        while not all([ind in appendix for ind in required_inds]) or \
                syllogism[0] == "O" and not any([prop == pred0 for ind in appendix for prop in ind]):
            appendix = []
            for i in range(size):
                appendix.append(self.draw_individual(syllogism[0], subj0, pred0, random.random() < deviation))
        mm.extend(appendix)

        i_b = [i for i, row in enumerate(mm) if "b" in row]

        ### Encode second premise ###
        if syllogism[1] == "A":
            for i in i_b:
                mm[i].append("c")
        elif syllogism[1] == "I":
            if subj1 == "b":
                for n, i in enumerate(i_b):
                    mm[i].append("c")
                    if n == 1:
                        break
            elif subj1 == "c":
                mm[i_b[0]].append("c")
                mm.extend([["c"]])
        elif syllogism[1] == "E":
            if subj1 == "b":
                for i in i_b:
                    mm[i].append("-c")
                mm.append(["c"])
            elif subj1 == "c":
                mm.extend([[subj1, "-"+pred1], [subj1, "-"+pred1], [subj1, "-"+pred1], [subj1, "-"+pred1]])
        elif syllogism[1] == "O":
            if subj1 == "b":
                for n, i in enumerate(i_b):
                    mm[i].append("-c")
                    # augment max 2 individuals
                    if n == 1:
                        break
                mm.append(["c"])
            elif subj1 == "c":
                mm.extend([["-b", "c"], ["c"]])

        return [sorted(row, key=lambda e: e[-1]) for row in mm]

    def heuristic(self, syllogism):
        """ Heuristic to draw conclusion from a syllogism's premises' intensions.

        >>> m = MReasoner()
        >>> # Test cases obtained by entering syllogisms into MReasoner with sigma=0
        >>> [m.heuristic("AA1"), m.heuristic("AA2"), m.heuristic("AA3"), m.heuristic("AA4")]
        [['Aac'], ['Aac', 'Aca'], ['Aac'], ['Aac']]
        >>> [m.heuristic("AI1"), m.heuristic("AI2"), m.heuristic("AI3"), m.heuristic("AI4")]
        [['Iac'], ['Iac', 'Ica'], ['Iac', 'Ica'], ['Iac', 'Ica']]
        >>> [m.heuristic("AE1"), m.heuristic("AE2"), m.heuristic("AE3"), m.heuristic("AE4")]
        [['Eac'], ['Eac', 'Eca'], ['Eac', 'Eca'], ['Eac', 'Eca']]
        >>> [m.heuristic("AO1"), m.heuristic("AO2"), m.heuristic("AO3"), m.heuristic("AO4")]
        [['Oac'], ['Oca'], ['Oca'], ['Oac']]

        >>> [m.heuristic("IA1"), m.heuristic("IA2"), m.heuristic("IA3"), m.heuristic("IA4")]
        [['Iac'], ['Iac', 'Ica'], ['Iac', 'Ica'], ['Iac', 'Ica']]
        >>> [m.heuristic("II1"), m.heuristic("II2"), m.heuristic("II3"), m.heuristic("II4")]
        [['Iac'], ['Iac', 'Ica'], ['Iac'], ['Iac']]
        >>> [m.heuristic("IE1"), m.heuristic("IE2"), m.heuristic("IE3"), m.heuristic("IE4")]
        [['Eac'], ['Eac', 'Eca'], ['Eac', 'Eca'], ['Eac']]
        >>> [m.heuristic("IO1"), m.heuristic("IO2"), m.heuristic("IO3"), m.heuristic("IO4")]
        [['Oac'], ['Oca'], ['Oca'], ['Oac']]

        >>> [m.heuristic("EA1"), m.heuristic("EA2"), m.heuristic("EA3"), m.heuristic("EA4")]
        [['Eac'], ['Eac', 'Eca'], ['Eac', 'Eca'], ['Eac', 'Eca']]
        >>> [m.heuristic("EI1"), m.heuristic("EI2"), m.heuristic("EI3"), m.heuristic("EI4")]
        [['Eac'], ['Eac', 'Eca'], ['Eac', 'Eca'], ['Eac']]
        >>> [m.heuristic("EE1"), m.heuristic("EE2"), m.heuristic("EE3"), m.heuristic("EE4")]
        [['Eac'], ['Eac', 'Eca'], ['Eac'], ['Eac']]
        >>> [m.heuristic("EO1"), m.heuristic("EO2"), m.heuristic("EO3"), m.heuristic("EO4")]
        [['Oac'], ['Oca'], ['Oca'], ['Oac']]

        >>> [m.heuristic("OA1"), m.heuristic("OA2"), m.heuristic("OA3"), m.heuristic("OA4")]
        [['Oac'], ['Oca'], ['Oac'], ['Oca']]
        >>> [m.heuristic("OI1"), m.heuristic("OI2"), m.heuristic("OI3"), m.heuristic("OI4")]
        [['Oac'], ['Oca'], ['Oac'], ['Oca']]
        >>> [m.heuristic("OE1"), m.heuristic("OE2"), m.heuristic("OE3"), m.heuristic("OE4")]
        [['Oac'], ['Oca'], ['Oac'], ['Oca']]
        >>> [m.heuristic("OO1"), m.heuristic("OO2"), m.heuristic("OO3"), m.heuristic("OO4")]
        [['Oac'], ['Oca'], ['Oac'], ['Oac']]
        """

        # rank order, see Khemlani 2013, p.13 (see also PHM)
        if "O" in syllogism:
            dominant_mood = "O"
        elif "E" in syllogism:
            dominant_mood = "E"
        elif "I" in syllogism:
            dominant_mood = "I"
        else:
            dominant_mood = "A"

        figure = syllogism[2]

        # From Scanner.lisp methods figure-1 to figure-4
        if figure == "1":  # ab-bc
            conclusion_order = ["ac"]
        elif figure == "2":  # ba-cb
            conclusion_order = ["ca"] if dominant_mood == "O" else ["ac", "ca"]
        elif figure == "3":  # ab-cb
            if syllogism[0] == syllogism[1]:
                conclusion_order = ["ac"]
            elif syllogism[0] == "O":
                conclusion_order = ["ac"]
            elif syllogism[1] == "O":
                conclusion_order = ["ca"]
            else:
                conclusion_order = ["ac", "ca"]
        elif figure == "4":  # ba-bc
            if syllogism[0] == syllogism[1]:
                conclusion_order = ["ac"]
            elif syllogism[0] == "O":
                conclusion_order = ["ca"]
            elif syllogism[1] == "O":
                conclusion_order = ["ac"]
            elif syllogism[0] == "I" and syllogism[1] == "E":
                conclusion_order = ["ac"]
            elif syllogism[0] == "E" and syllogism[1] == "I":
                conclusion_order = ["ac"]
            else:
                conclusion_order = ["ac", "ca"]

        return [dominant_mood + co for co in conclusion_order]

    def check_if_holds(self, mental_model, proposition):
        """ check if a proposition/conclusion holds in a mental model

        >>> # Check if premises hold in their initial models
        >>> m = MReasoner()
        >>> import ccobra
        >>> tests_premises = []
        >>> for s in ccobra.syllogistic.SYLLOGISMS:
        ...     premises = sylutil.syllogism_to_premises(s)
        ...     heuristic_conclusions = m.heuristic(s)
        ...     for size in range(2, 8):
        ...         for eps in m.param_grid["epsilon"]:
        ...             for _ in range(100):
        ...                 initial_model = m.encode(s, size, eps)
        ...                 for p in premises:
        ...                     tests_premises.append(m.check_if_holds(initial_model, p))
        >>> all(tests_premises)
        True
        >>> # Test cases from querying the lisp implementaiton
        >>> m.check_if_holds([['a', 'b'], ['a', 'b'], ['b'], ['b'], ['-b', 'c'], ['-b', 'c'], ['-b', 'c'], ['-b', 'c']], "Eac")
        True
        >>> m.check_if_holds([['a', 'b'], ['a', 'b'], ['b'], ['b'], ['-b', 'c'], ['-b', 'c'], ['-b', 'c'], ['-b', 'c']], "Eca")
        True
        >>> m.check_if_holds([['-a', 'b'], ['-a', 'b'], ['b'], ['b'], ['a'], ['-b', 'c'], ['-b', 'c'], ['-b', 'c'], ['-b', 'c']], "Oca")
        True
        >>> # Aac
        >>> m.check_if_holds([['a', 'b', 'c'], ['a', 'b', 'c'], ['a', 'b', 'c'], ['a', 'b', 'c']], "Aac")
        True
        >>> m.check_if_holds([['a', 'b', 'c'], ['a', 'b', 'c'], ['a', 'b'], ['a', 'b']], "Aac")
        False
        >>> m.check_if_holds([['a', '-b'], ['a', '-b'], ['a', '-b'], ['a', '-b'], ['b', '-c'], ['c']], "Aac")
        False
        >>> m.check_if_holds([['-a', 'b'], ['-a', 'b'], ['c'], ['c'], ['a']], "Aac")
        False
        >>> # Iac
        >>> m.check_if_holds([['a', 'b', 'c'], ['a', 'b', 'c'], ['a'], ['a']], "Iac")
        True
        >>> m.check_if_holds([['a', 'b', '-c'], ['a', 'b', '-c'], ['a', 'b', '-c'], ['a', 'b', '-c'], ['c']], "Iac")
        False
        >>> m.check_if_holds([['-a', 'b'], ['-a', 'b'], ['b'], ['b'], ['a'], ['-b', 'c']], "Iac")
        False
        >>> m.check_if_holds([['a', 'b', 'c'], ['a', 'b', 'c'], ['a', 'b', 'c'], ['a', 'b', 'c']], "Iac")
        True
        >>> # Eac
        >>> m.check_if_holds([['a', 'b', 'c'], ['a', 'b', 'c'], ['a', 'b', 'c'], ['a', 'b', 'c']], "Eac")
        False
        >>> m.check_if_holds([['a', '-b'], ['a', '-b'], ['a', '-b'], ['a', '-b'], ['b', '-c'], ['c']], "Eac")
        True
        >>> m.check_if_holds([['-a', 'b', '-c'], ['-a', 'b', '-c'], ['-a', 'b', '-c'], ['-a', 'b', '-c'], ['a'], ['c']], "Eac")
        True
        >>> m.check_if_holds([['a', 'b', 'c'], ['a', 'b', 'c'], ['a'], ['a']], "Eac")
        False
        >>> m.check_if_holds([['a', 'b', '-c'], ['a', 'b', '-c'], ['a', 'b', '-c'], ['a', 'b', '-c'], ['c']], "Eac")
        True
        >>> m.check_if_holds([['a', 'b', '-c'], ['a', 'b', '-c'], ['a'], ['a'], ['c']], "Eac")
        True
        >>> m.check_if_holds([['-a', 'b'], ['-a', 'b'], ['-a', 'b']], "Eac")
        False
        >>> m.check_if_holds([['a', '-b'], ['-a', 'b'], ['-a', 'b'], ['a', '-b']], "Eac")
        True
        >>> # Oac
        >>> m.check_if_holds([['a', 'b', 'c'], ['a', 'b', 'c'], ['a', 'b', 'c'], ['a', 'b', 'c']], "Oac")
        False
        >>> m.check_if_holds([['a', 'b', 'c'], ['a', 'b', 'c'], ['a', 'b'], ['a', 'b']], "Oac")
        True
        >>> m.check_if_holds([['a', 'b', 'c'], ['a', 'b', 'c'], ['b', 'c'], ['b', 'c']], "Oac")
        False
        >>> m.check_if_holds([['a', 'b', '-c'], ['a', 'b', '-c'], ['a', 'b'], ['a', 'b']], "Oac")
        True
        >>> m.check_if_holds([['b', 'c'], ['a', 'b'], ['c']], "Iac")
        False
        """

        subj, pred = proposition[1], proposition[2]
        x = [pred in row for row in mental_model if subj in row]  # occurences of pred in rows containing subj
        if proposition[0] == "A":
            return True if all(x) and any(x) else False  # all subj are pred and at least one subj is pred
        elif proposition[0] == "I":
            return True if any(x) else False  # at least one subj is pred
        elif proposition[0] == "E":
            return True if not any(x) and any(subj in row for row in mental_model) else False  # no subj is pred and there is at least one subj
        elif proposition[0] == "O":
            return True if not all(x) else False  # not all subj are pred

    def generic_models(self):
        """ Generate all possible models with positive/negative terms "a", "b", "c" across two individuals """

        models = []
        for neg1a in ["", "-"]:
            for neg1b in ["", "-"]:
                for neg1c in ["", "-"]:
                    for neg2a in ["", "-"]:
                        for neg2b in ["", "-"]:
                            for neg2c in ["", "-"]:
                                models.extend([[[neg1a+"a", neg1b+"b", neg1c+"c"], [neg2a+"a", neg2b+"b", neg2c+"c"]]])
        return models

    def search_counterexample(self, conclusion, syllogism):
        """ Brute force search for counterexamples over generic models """

        premises = [syllogism[i] + sylutil.term_order(syllogism[2])[i] for i in [0, 1]]

        for generic_model in self.generic_models():
            # only models that satisfy the premises can be valid counterexamples
            if all([self.check_if_holds(generic_model, prem) for prem in premises]):
                if not self.check_if_holds(generic_model, conclusion):
                    return generic_model
        return None

    def verify_conclusion(self, conclusion, syllogism, weaken=None):
        """ Verify a conclusion by counterexample search and (possibly) weakening the conclusion """
        if weaken is None:
            weaken = random.random()

        counterexample_model = self.search_counterexample(conclusion, syllogism)
        if counterexample_model is not None:
            # Counterexample found
            if weaken < self.params["Weaken"]:
                # Try to weaken conclusion
                if conclusion[0] in ["A", "E"]:
                    weaker_conclusion = {"A": "I", "E": "O"}[conclusion[0]] + conclusion[1:]
                    weaker_counterexample_model = self.search_counterexample(weaker_conclusion, syllogism)
                    if weaker_counterexample_model is not None:
                        # Counterexample found for weaker conclusion
                        return "NVC", False
                    else:
                        # No counterexample found for weaker conclusion
                        return weaker_conclusion, False
                else:
                    # Conclusion can not be weakened (= "I" or "O")
                    return "NVC", False
            else:
                # Don't try to weaken conclusion
                return "NVC", False
        # No counterexample found
        if weaken >= self.params["Weaken"]:
            return conclusion, True
        return conclusion, False

    def trunc_poisson_density(self, n, lam):
        """ Copied from mReasoner source code.

        >>> m = MReasoner()
        >>> m.trunc_poisson_density(1, 1)
        0.36787944117144233
        >>> m.trunc_poisson_density(10, 4)
        0.0052924766764201195
        >>> m.trunc_poisson_density(0, 4)
        0.01831563888873418
        """
        if n > 34:
            n = 34
        if lam > 0.0:
            x1 = math.pow(lam, n)
            x2 = math.exp(-lam)
            x3 = math.factorial(n)
            x = x1*x2/x3
            return x
        elif lam == 0.0:
            if n == 0:
                return 1.0
            else:
                return 0.0
        else:
            raise Exception

    def trunc_poisson_sample(self, lam):
        """ Copied from mReasoner source code."""
        u = random.random()
        p = 0
        i = 0
        while True:
            p += self.trunc_poisson_density(i, lam)
            if u < p:
                return i
            i += 1

    def predict(self, syllogism):
        size = 0
        while size < 2:
            size = int(self.trunc_poisson_sample(self.params["lambda"]))

        initial_model = self.encode(syllogism, size, self.params["epsilon"])
        conclusions = self.heuristic(syllogism)

        checked_conclusions = []
        for i, _ in enumerate(conclusions):
            if self.check_if_holds(initial_model, conclusions[i]):
                checked_conclusions.append(conclusions[i])

        conclusions = checked_conclusions
        weaken = random.random()

        add_nvc = False
        if random.random() < self.params["System 2"]:
            for i, _ in enumerate(conclusions):
                if conclusions[i] == "NVC":
                    continue
                x = self.verify_conclusion(conclusions[i], syllogism, weaken)
                conclusions[i] = x[0]
                if x[1]:
                    add_nvc = True
        # Only return NVC if no valid conclusion was found
        conclusions = set([c for c in conclusions if c != "NVC"])
        if add_nvc:
            conclusions.add("NVC")
        return sorted(list(conclusions)) if len(conclusions) != 0 else ["NVC"]
