import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from modular_models.models.basic_models.interface import SyllogisticReasoningModel


class Matching(SyllogisticReasoningModel):
    """ Matching theory based on Wetherick & Gilhooly (1995)
    """

    def __init__(self):
        SyllogisticReasoningModel.__init__(self)

    f = {"AA": ["Aac", "Aca"],
         "AE": ["Eac", "Eca"], "EA": ["Eac", "Eca"],
         "AI": ["Iac", "Ica"], "IA": ["Iac", "Ica"],
         "AO": ["Oac", "Oca"], "OA": ["Oac", "Oca"],
         "EE": ["Eac", "Eca"],
         "EI": ["Eac", "Eca", "Iac", "Ica"], "IE": ["Eac", "Eca", "Iac", "Ica"],
         "EO": ["Eac", "Eca", "Oac", "Oca"], "OE": ["Eac", "Eca", "Oac", "Oca"],
         "II": ["Iac", "Ica"],
         "IO": ["Iac", "Ica", "Oac", "Oca"], "OI": ["Iac", "Ica", "Oac", "Oca"],
         "OO": ["Oac", "Oca"]}

    @staticmethod
    def heuristic_matching(syllogism):
        return Matching.f[syllogism[:2]]

    def predict(self, syllogism):
        return self.heuristic_matching(syllogism)
