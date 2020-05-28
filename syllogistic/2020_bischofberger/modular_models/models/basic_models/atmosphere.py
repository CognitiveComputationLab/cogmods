import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from modular_models.models.basic_models.interface import SyllogisticReasoningModel


class Atmosphere(SyllogisticReasoningModel):
    """ Predictive version of the Atmospheric model based on Woodworth & Sells (1935). Predicts the mood
    of the conclusion according to the theory.
    """

    def __init__(self):
        SyllogisticReasoningModel.__init__(self)

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

    def heuristic_atmosphere(self, syllogism):
        return Atmosphere.f[syllogism[:2]]

    def predict(self, syllogism):
        return self.heuristic_atmosphere(syllogism)
