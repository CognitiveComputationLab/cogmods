import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from modular_models.models.basic_models.interface import SyllogisticReasoningModel


class GeneralizedAtmosphere(SyllogisticReasoningModel):
    """ Generalized version of the Atmospheric model with variable dominance relations.
    """

    def __init__(self):
        SyllogisticReasoningModel.__init__(self)

        self.params["dominant_quality"] = ["neg"]
        self.params["dominant_quantifier"] = ["some"]

        self.param_grid["dominant_quality"] = ["neg", "pos"]
        self.param_grid["dominant_quantifier"] = ["some", "all"]

    def heuristic_abstract_atmosphere(self, syllogism):
        m2qual = {"A": "pos", "I": "pos", "E": "neg", "O": "neg"}
        m2quant = {"A": "all", "I": "some", "E": "all", "O": "some"}
        quals = [m2qual[m] for m in syllogism[:2]]
        quants = [m2quant[m] for m in syllogism[:2]]

        quality = self.params["dominant_quality"] if self.params["dominant_quality"] in quals else [q for q in ["pos", "neg"] if q != self.params["dominant_quality"]][0]
        quantifier = self.params["dominant_quantifier"] if self.params["dominant_quantifier"] in quants else [q for q in ["all", "some"] if q != self.params["dominant_quantifier"]][0]
        return {("pos", "some"): "I", ("pos", "all"): "A", ("neg", "some"): "O", ("neg", "all"): "E"}[(quality, quantifier)]

    def predict(self, syllogism):
        concl_mood = self.heuristic_abstract_atmosphere(syllogism)[0][0]
        return [concl_mood + ac for ac in ["ac", "ca"]]
