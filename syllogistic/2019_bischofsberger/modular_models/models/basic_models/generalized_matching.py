import itertools
import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from modular_models.models.basic_models.interface import SyllogisticReasoningModel


class GeneralizedMatching(SyllogisticReasoningModel):
    def __init__(self):
        SyllogisticReasoningModel.__init__(self)
        self.params["total_order"] = [["E", "O", "I", "A"], ["=", "=", ">"]]
        self.params["conclusion_order"] = 0.6

        relations = [["="]*3, [">"]*3, ["=", "=", ">"], ["=", ">", "="], [">", "=", "="],
                                        ["=", ">", ">"], [">", "=", ">"], [">", ">", "="]]
        moods = list(itertools.permutations(["A", "E", "I", "O"]))
        self.param_grid["total_order"] = list([(tuple(x), tuple(y)) for x, y in itertools.product(moods, relations)])
        self.param_grid["conclusion_order"] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    def heuristic_generalized_matching(self, syllogism):
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
        concl_mood = random.choice(self.heuristic_generalized_matching(syllogism))
        concl_order = "ca"
        if random.random() < self.params["conclusion_order"]:
            concl_order = "ac"
        return [concl_mood + concl_order]
