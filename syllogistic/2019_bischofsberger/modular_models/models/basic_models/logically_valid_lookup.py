import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from modular_models.util import sylutil
from modular_models.models.basic_models.interface import SyllogisticReasoningModel


def syllogism_combinations(syllogism, additional_premises):
    """
    >>> syllogism_combinations("AA1", ["Aba"])
    ['AA4']
    """

    for p in additional_premises:
        if p[0] not in syllogism:
            raise NotImplementedError

    new_syllogisms = []
    for p in additional_premises:
        i = 0 if "a" in p and "b" in p else 1
        i_other = (i + 1) % 2
        premises = [None, None]
        premises[i] = p
        premises[i_other] = syllogism[i_other] + sylutil.term_order(syllogism[2])[i_other]

        if premises[0][1:] == "ab" and premises[1][1:] == "bc":
            figure = "1"
        elif premises[0][1:] == "ba" and premises[1][1:] == "cb":
            figure = "2"
        elif premises[0][1:] == "ab" and premises[1][1:] == "cb":
            figure = "3"
        elif premises[0][1:] == "ba" and premises[1][1:] == "bc":
            figure = "4"
        else:
            raise Exception

        new_syllogisms.append(premises[0][0] + premises[1][0] + figure)
    return new_syllogisms


class LogicallyValidLookup(SyllogisticReasoningModel):
    """ Randomly predicts one of the logically valid conclusions to a syllogism under the following
    assumptions:
        - Existential assumption: Axy implies Ixy, Exy implies Oxy
        - No conlusion order is preferred in case of logical symmetry (e.g. Iac vs. Ica)
        - A stronger conclusion is preferred over its weaker implication (e.g. Aac vs. Iac)
    """

    # Logical validity truth table with existential assumption. See e.g. Khemlani (2012) or
    # Wikipedia (https://en.wikipedia.org/wiki/List_of_valid_argument_forms#Valid_syllogistic_forms)
    f_val_ex = {"AA1": ["Aac", "Iac", "Ica"],
                "AA2": ["Aca", "Iac", "Ica"],
                "AA3": ["NVC"],
                "AA4": ["Iac", "Ica"],
                "AI1": ["NVC"],
                "AI2": ["Iac", "Ica"],
                "AI3": ["NVC"],
                "AI4": ["Iac", "Ica"],
                "AE1": ["Eac", "Eca", "Oac", "Oca"],
                "AE2": ["Oac"],
                "AE3": ["Eac", "Eca", "Oac", "Oca"],
                "AE4": ["Oac"],
                "AO1": ["NVC"],
                "AO2": ["NVC"],
                "AO3": ["Oca"],
                "AO4": ["Oac"],

                "IA1": ["Iac", "Ica"],
                "IA2": ["NVC"],
                "IA3": ["NVC"],
                "IA4": ["Iac", "Ica"],
                "II1": ["NVC"],
                "II2": ["NVC"],
                "II3": ["NVC"],
                "II4": ["NVC"],
                "IE1": ["Oac"],
                "IE2": ["Oac"],
                "IE3": ["Oac"],
                "IE4": ["Oac"],
                "IO1": ["NVC"],
                "IO2": ["NVC"],
                "IO3": ["NVC"],
                "IO4": ["NVC"],

                "EA1": ["Oca"],
                "EA2": ["Eac", "Eca", "Oac", "Oca"],
                "EA3": ["Eac", "Eca", "Oac", "Oca"],
                "EA4": ["Oca"],
                "EI1": ["Oca"],
                "EI2": ["Oca"],
                "EI3": ["Oca"],
                "EI4": ["Oca"],
                "EE1": ["NVC"],
                "EE2": ["NVC"],
                "EE3": ["NVC"],
                "EE4": ["NVC"],
                "EO1": ["NVC"],
                "EO2": ["NVC"],
                "EO3": ["NVC"],
                "EO4": ["NVC"],

                "OA1": ["NVC"],
                "OA2": ["NVC"],
                "OA3": ["Oac"],
                "OA4": ["Oca"],
                "OI1": ["NVC"],
                "OI2": ["NVC"],
                "OI3": ["NVC"],
                "OI4": ["NVC"],
                "OE1": ["NVC"],
                "OE2": ["NVC"],
                "OE3": ["NVC"],
                "OE4": ["NVC"],
                "OO1": ["NVC"],
                "OO2": ["NVC"],
                "OO3": ["NVC"],
                "OO4": ["NVC"],
                }

    # same but prefers E over O and A over I
    f_val_ex_stronger_prefered = {"AA1": ["Aac"],
                "AA2": ["Aca"],
                "AA3": ["NVC"],
                "AA4": ["Iac", "Ica"],
                "AI1": ["NVC"],
                "AI2": ["Iac", "Ica"],
                "AI3": ["NVC"],
                "AI4": ["Iac", "Ica"],
                "AE1": ["Eac", "Eca"],
                "AE2": ["Oac"],
                "AE3": ["Eac", "Eca"],
                "AE4": ["Oac"],
                "AO1": ["NVC"],
                "AO2": ["NVC"],
                "AO3": ["Oca"],
                "AO4": ["Oac"],

                "IA1": ["Iac", "Ica"],
                "IA2": ["NVC"],
                "IA3": ["NVC"],
                "IA4": ["Iac", "Ica"],
                "II1": ["NVC"],
                "II2": ["NVC"],
                "II3": ["NVC"],
                "II4": ["NVC"],
                "IE1": ["Oac"],
                "IE2": ["Oac"],
                "IE3": ["Oac"],
                "IE4": ["Oac"],
                "IO1": ["NVC"],
                "IO2": ["NVC"],
                "IO3": ["NVC"],
                "IO4": ["NVC"],

                "EA1": ["Oca"],
                "EA2": ["Eac", "Eca"],
                "EA3": ["Eac", "Eca"],
                "EA4": ["Oca"],
                "EI1": ["Oca"],
                "EI2": ["Oca"],
                "EI3": ["Oca"],
                "EI4": ["Oca"],
                "EE1": ["NVC"],
                "EE2": ["NVC"],
                "EE3": ["NVC"],
                "EE4": ["NVC"],
                "EO1": ["NVC"],
                "EO2": ["NVC"],
                "EO3": ["NVC"],
                "EO4": ["NVC"],

                "OA1": ["NVC"],
                "OA2": ["NVC"],
                "OA3": ["Oac"],
                "OA4": ["Oca"],
                "OI1": ["NVC"],
                "OI2": ["NVC"],
                "OI3": ["NVC"],
                "OI4": ["NVC"],
                "OE1": ["NVC"],
                "OE2": ["NVC"],
                "OE3": ["NVC"],
                "OE4": ["NVC"],
                "OO1": ["NVC"],
                "OO2": ["NVC"],
                "OO3": ["NVC"],
                "OO4": ["NVC"],
                }

    def syllogism_is_valid(self, syllogism):
        if self.f_val_ex_stronger_prefered[syllogism] == ["NVC"]:
            return False
        return True

    def predict(self, syllogism, additional_premises=None):
        if additional_premises is None:
            additional_premises = []
        new_syllogisms = syllogism_combinations(syllogism, additional_premises)
        conclusions = set()
        for syl in [syllogism] + new_syllogisms:
            conclusions.update(self.f_val_ex_stronger_prefered[syl])
        if "NVC" in conclusions and len(conclusions) != 1:
            conclusions.remove("NVC")
        return list(conclusions)
