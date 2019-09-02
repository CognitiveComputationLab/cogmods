import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from modular_models.models.basic_models.interface import SyllogisticReasoningModel


class GeneralizedAtmosphere(SyllogisticReasoningModel):
    """ Generalized version of the Atmospheric model with variable dominance relations.
    """

    def __init__(self):
        SyllogisticReasoningModel.__init__(self)

        self.params["dominant_quality"] = "neg"
        self.params["dominant_quantifier"] = "some"

        self.param_grid["dominant_quality"] = ["neg", "pos"]
        self.param_grid["dominant_quantifier"] = ["some", "all"]

    def heuristic_abstract_atmosphere(self, syllogism, dom_qual, dom_quant):
        m2qual = {"A": "pos", "I": "pos", "E": "neg", "O": "neg"}
        m2quant = {"A": "all", "I": "some", "E": "all", "O": "some"}
        quals = [m2qual[m] for m in syllogism[:2]]
        quants = [m2quant[m] for m in syllogism[:2]]

        quality = dom_qual if dom_qual in quals else [q for q in ["pos", "neg"] if q != dom_qual][0]
        quantifier = dom_quant if dom_quant in quants else [q for q in ["all", "some"] if q != dom_quant][0]

        return {("pos", "some"): "I", ("pos", "all"): "A", ("neg", "some"): "O", ("neg", "all"): "E"}[(quality, quantifier)]

    def predict(self, syllogism):
        """
        >>> from modular_models.models.basic_models import Atmosphere
        >>> import ccobra
        >>> a = Atmosphere()
        >>> ga = GeneralizedAtmosphere()
        >>> ga.params["dominant_quality"] = "neg"
        >>> ga.params["dominant_quantifier"] = "some"
        >>> for s in ccobra.syllogistic.SYLLOGISMS:
        ...     print(s, a.predict(s) == ga.predict(s))
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
        >>> ga.params["dominant_quality"] = "pos"
        >>> ga.params["dominant_quantifier"] = "all"
        >>> ga.predict("AO1")
        ['Aac', 'Aca']
        >>> ga.predict("EO1")
        ['Eac', 'Eca']
        >>> ga.predict("IO1")
        ['Iac', 'Ica']
        >>> ga.predict("AI1")
        ['Aac', 'Aca']
        """
        concl_mood = self.heuristic_abstract_atmosphere(syllogism, self.params["dominant_quality"], self.params["dominant_quantifier"])
        return [concl_mood + ac for ac in ["ac", "ca"]]
