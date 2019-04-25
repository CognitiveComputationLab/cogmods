import ccobra
import itertools
import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from modular_models.models.basic_models.interface import SyllogisticReasoningModel


class GeneralizedPHM(SyllogisticReasoningModel):
    """ PHM based on Chater & Oaksford 1999
    """
    def __init__(self):
        SyllogisticReasoningModel.__init__(self)

        self.params["p-entailment"] = 0.6
        self.params["conclusion_order"] = 0.6
        self.params["total_order"] = ["A", "I", "E", "O"]

        self.param_grid["p-entailment"] = [0.0, 1/6, 2/6, 0.5, 4/6, 5/6, 1.0]
        self.param_grid["conclusion_order"] = [0.0, 1/6, 2/6, 0.5, 4/6, 5/6, 1.0]
        self.param_grid["total_order"] = list(itertools.permutations(["A", "I", "E", "O"]))

    f_p_entailment = {"A": "I",
                      "E": "O",
                      "I": "O",
                      "O": "I"}

    def min_heuristic(self, syllogism):
        i1, i2 = self.params["total_order"].index(syllogism[0]), self.params["total_order"].index(syllogism[1])
        return self.params["total_order"][max(i1, i2)]

    def predict(self, syllogism):
        concl_mood = self.min_heuristic(syllogism)
        if random.random() < self.params["p-entailment"]:
            concl_mood = self.f_p_entailment[concl_mood]
        concl_order = "ca"
        if random.random() < self.params["conclusion_order"]:
            concl_order = "ac"

        return [concl_mood + concl_order]
