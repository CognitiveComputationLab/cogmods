import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from modular_models.models.basic_models.interface import SyllogisticReasoningModel


class Atmosphere(SyllogisticReasoningModel):
    """ Predictive version of the Atmospheric model based on Woodworth & Sells (1935). Predicts the mood
    of the conclusion according to the theory + conclusion order is 50/50
    """

    def __init__(self):
        SyllogisticReasoningModel.__init__(self)
        self.params["conclusion_order"] = 0.5
        self.param_grid["conclusion_order"] = [0.0, 1/6, 2/6, 0.5, 4/6, 5/6, 1.0]

    f = {"AA": ["Aac", "Aca"],
         "AE": ["Eac", "Eca"], "EA": ["Eac", "Eca"],
         "AI": ["Iac", "Ica"], "IA": ["Iac", "Ica"],
         "AO": ["Oac", "Oca"], "OA": ["Oac", "Oca"],
         "EE": ["Eac", "Eca"],
         "EI": ["Oac", "Oca"], "IE": ["Oac", "Oca"],
         "EO": ["Oac", "Oca"], "OE": ["Oac", "Oca"],
         "II": ["Iac", "Ica"],
         "IO": ["Oac", "Oca"], "OI": ["Oac", "Oca"],
         "OO": ["Oac", "Oca"]}

    @staticmethod
    def heuristic_atmosphere(syllogism):
        return Atmosphere.f[syllogism[:2]]

    def predict(self, syllogism):
        concl_mood = self.heuristic_atmosphere(syllogism)[0][0]
        concl_order = "ca"
        if random.random() < self.params["conclusion_order"]:
            concl_order = "ac"
        return [concl_mood + concl_order]
