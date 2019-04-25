import ccobra
import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from modular_models.models.basic_models.interface import SyllogisticReasoningModel


class PHM(SyllogisticReasoningModel):
    """ PHM based on Chater & Oaksford 1999 """

    def __init__(self):
        SyllogisticReasoningModel.__init__(self)

        self.params["p-entailment"] = 0.6
        self.params["conclusion_order"] = 0.6

        self.param_grid["p-entailment"] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.param_grid["conclusion_order"] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    f_min_heuristic = {"AA": "A",
                       "AE": "E", "EA": "E",
                       "AI": "I", "IA": "I",
                       "AO": "O", "OA": "O",
                       "EE": "E",
                       "EI": "E", "IE": "E",
                       "EO": "O", "OE": "O",
                       "II": "I",
                       "IO": "O", "OI": "O",
                       "OO": "O"}

    f_p_entailment = {"A": "I",
                      "E": "O",
                      "I": "O",
                      "O": "I"}

    def predict(self, syllogism):
        concl_mood = self.f_min_heuristic[syllogism[:2]]
        if random.random() < self.params["p-entailment"]:
            concl_mood = self.f_p_entailment[concl_mood]
        concl_order = "ca"
        if random.random() < self.params["conclusion_order"]:
            concl_order = "ac"

        return [concl_mood + concl_order]
