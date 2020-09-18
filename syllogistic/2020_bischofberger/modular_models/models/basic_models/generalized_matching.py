import itertools
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from modular_models.models.basic_models.interface import SyllogisticReasoningModel


class GeneralizedMatching(SyllogisticReasoningModel):
    def __init__(self):
        SyllogisticReasoningModel.__init__(self)
        self.params["total_order"] = [["E", "O", "I", "A"], ["=", "=", ">"]]

        relations = [["="]*3, [">"]*3, ["=", "=", ">"], ["=", ">", "="], [">", "=", "="],
                                        ["=", ">", ">"], [">", "=", ">"], [">", ">", "="]]
        moods = list(itertools.permutations(["A", "E", "I", "O"]))
        self.param_grid["total_order"] = list([(tuple(x), tuple(y)) for x, y in itertools.product(moods, relations)])

    def heuristic_generalized_matching(self, syllogism):
        """
        >>> m = GeneralizedMatching()
        >>> m.params["total_order"] = [["E", "O", "I", "A"], ["=", "=", ">"]]  # E = O = I > A
        >>> m.heuristic_generalized_matching("AA1")
        ['A']
        >>> m.heuristic_generalized_matching("AI1")
        ['I']
        >>> m.heuristic_generalized_matching("OE4")
        ['E', 'O']

        >>> m.params["total_order"] = [["A", "I", "E", "O"], ["=", ">", ">"]]  # A = I > E > O
        >>> m.heuristic_generalized_matching("OA4")
        ['A']
        >>> m.heuristic_generalized_matching("EI2")
        ['I']
        >>> m.heuristic_generalized_matching("AI3")
        ['A', 'I']

        >>> m.params["total_order"] = [["A", "E", "I", "O"], ["=", "=", "="]]  # A = I > E > O
        >>> m.heuristic_generalized_matching("OA4")
        ['A', 'O']
        >>> m.heuristic_generalized_matching("EI1")
        ['E', 'I']
        >>> m.heuristic_generalized_matching("EA3")
        ['A', 'E']
        >>> m.heuristic_generalized_matching("AO2")
        ['A', 'O']
        """
        moods = self.params["total_order"][0]
        relations = self.params["total_order"][1]

        # AA -> A, EE -> E, ...
        if syllogism[0] == syllogism[1]:
            return [syllogism[0]]

        # get positions of premise moods in total order
        i_premise1, i_premise2 = moods.index(syllogism[0]), moods.index(syllogism[1])
        i, j = min(i_premise1, i_premise2), max(i_premise1, i_premise2)

        # the lower index (more or equally conservative) mood is certainly in the result
        res_moods = [moods[i]]

        # the higher index mood is in the result if it is reached by traversing only "=" relations from the lower index
        for ix in range(i, j):
            if relations[ix] == ">":
                break
            if ix == j-1:
                res_moods.append(moods[j])

        return res_moods

    def predict(self, syllogism):
        """
        >>> from modular_models.models.basic_models import Matching
        >>> import ccobra
        >>> m = Matching()
        >>> gm = GeneralizedMatching()
        >>> gm.params["total_order"] = [["E", "O", "I", "A"], ["=", "=", ">"]]
        >>> for s in ccobra.syllogistic.SYLLOGISMS:
        ...     print(s, m.predict(s) == gm.predict(s))
        AA1 True
        AA2 True
        AA3 True
        AA4 True
        AI1 True
        AI2 True
        AI3 True
        AI4 True
        AE1 True
        AE2 True
        AE3 True
        AE4 True
        AO1 True
        AO2 True
        AO3 True
        AO4 True
        IA1 True
        IA2 True
        IA3 True
        IA4 True
        II1 True
        II2 True
        II3 True
        II4 True
        IE1 True
        IE2 True
        IE3 True
        IE4 True
        IO1 True
        IO2 True
        IO3 True
        IO4 True
        EA1 True
        EA2 True
        EA3 True
        EA4 True
        EI1 True
        EI2 True
        EI3 True
        EI4 True
        EE1 True
        EE2 True
        EE3 True
        EE4 True
        EO1 True
        EO2 True
        EO3 True
        EO4 True
        OA1 True
        OA2 True
        OA3 True
        OA4 True
        OI1 True
        OI2 True
        OI3 True
        OI4 True
        OE1 True
        OE2 True
        OE3 True
        OE4 True
        OO1 True
        OO2 True
        OO3 True
        OO4 True
        >>> gm.params["total_order"] = [["E", "O", "I", "A"], ["=", "=", "="]]
        >>> from modular_models.util import sylutil
        >>> all([gm.predict(s) == sorted(list(set([m+ac for m in s[:2] for ac in ["ac", "ca"]]))) for s in ccobra.syllogistic.SYLLOGISMS])
        True
        >>> gm.params["total_order"] = [["I", "A", "O", "E"], [">", ">", ">"]]
        >>> gm.predict("EO1")
        ['Oac', 'Oca']
        >>> gm.predict("EI2")
        ['Iac', 'Ica']
        >>> gm.predict("AI3")
        ['Iac', 'Ica']
        """

        concl_moods = self.heuristic_generalized_matching(syllogism)
        return sorted([mood + ac for ac in ["ac", "ca"] for mood in concl_moods])
