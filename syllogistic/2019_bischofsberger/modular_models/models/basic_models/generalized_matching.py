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
        concl_moods = self.heuristic_generalized_matching(syllogism)
        return [mood + ac for ac in ["ac", "ca"] for mood in concl_moods]
