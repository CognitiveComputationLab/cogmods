import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from modular_models.util import sylutil
from modular_models.models.basic_models.interface import SyllogisticReasoningModel
from modular_models.models.basic_models.logically_valid_lookup import LogicallyValidLookup


class IllicitConversion(SyllogisticReasoningModel):
    """ Prediction model for Illicit Conversion based on Chapman & Chapman (1959) and Revlis (1975).
    Only includes the idea of illicitly reversing premises, without the probabilistic inference
    component described by Chapman & Chapman (1959).
    """

    def __init__(self):
        SyllogisticReasoningModel.__init__(self)

        # Use blackbox logically correct reasoning as deduction mechanism
        self.reasoning_model = LogicallyValidLookup()

        self.params["reverse_first_premise"] = 0.6
        self.params["reverse_second_premise"] = 0.6
        self.params["reverse_A"] = 0.6
        self.params["reverse_O"] = 0.6

        self.param_grid["reverse_first_premise"] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.param_grid["reverse_second_premise"] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.param_grid["reverse_A"] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.param_grid["reverse_O"] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    @staticmethod
    def reverse_premises(syllogism, first=True, second=True):
        additional_premises = []
        if first:
            additional_premises.append(syllogism[0] + sylutil.term_order(syllogism[2])[0][::-1])
        elif second:
            additional_premises.append(syllogism[1] + sylutil.term_order(syllogism[2])[1][::-1])
        return additional_premises

    def predict(self, syllogism):
        p1_mood = {"A": self.params["reverse_A"], "O": self.params["reverse_O"], "E": 0, "I": 0}[syllogism[0]]
        p2_mood = {"A": self.params["reverse_A"], "O": self.params["reverse_O"], "E": 0, "I": 0}[syllogism[1]]
        reverse_first_premise = True if random.random() < self.params["reverse_first_premise"] * p1_mood else False
        reverse_second_premise = True if random.random() < self.params["reverse_second_premise"] * p2_mood else False
        additional_premises = self.reverse_premises(syllogism, reverse_first_premise, reverse_second_premise)
        return self.reasoning_model.predict(syllogism, additional_premises)
