import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from modular_models.models.basic_models.interface import SyllogisticReasoningModel
from modular_models.util import sylutil


class PHM(SyllogisticReasoningModel):
    """ PHM based on Chater & Oaksford 1999 """

    def __init__(self):
        SyllogisticReasoningModel.__init__(self)

        self.params["p-entailment"] = 0.6

        # max heuristic
        self.params["confidenceA"] = 0.7
        self.params["confidenceI"] = 0.6
        self.params["confidenceE"] = 0.5
        self.params["confidenceO"] = 0.4

        self.param_grid["p-entailment"] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.param_grid["confidenceA"] = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.param_grid["confidenceI"] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.param_grid["confidenceE"] = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.param_grid["confidenceO"] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

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

    mood_to_phrase = {"A": "All", "I": "Some", "E": "No", "O": "Some"}
    informativeness = {"A": 4, "I": 3, "E": 2, "O": 1}

    def generate_param_configurations(self):
        configs = SyllogisticReasoningModel.generate_param_configurations(self)
        configs = [c for c in configs if c["confidenceA"] > c["confidenceI"] and c["confidenceI"] > c["confidenceE"] and c["confidenceE"] > c["confidenceO"]]
        return configs

    def max_premise(self, syllogism):
        premises = sylutil.syllogism_to_premises(syllogism)
        info = [self.informativeness[prem[0]] for prem in premises]
        max_premise = premises[info.index(max(info))]
        return max_premise

    def attachment(self, mood, syllogism):
        premises = sylutil.syllogism_to_premises(syllogism)
        phrases = [(self.mood_to_phrase[prem[0]], prem[1]) for prem in premises]
        candidates = [(self.mood_to_phrase[mood], subj) for subj in ["a", "c"]]
        candidates = [c for c in candidates if c in phrases]

        if len(candidates) == 0:
            # Criterion: Use the end term of the max premise. Does not apply if premises have equal mood.
            if premises[0][0] == premises[1][0]:
                return [mood+"ac", mood+"ca"]

            info = [self.informativeness[prem[0]] for prem in premises]
            max_premise = premises[info.index(max(info))]
            subj = "a" if "a" in max_premise else "c"
            return [mood + subj + ("a" if subj == "c" else "c")]

        elif len(candidates) == 1:
            return [mood + candidates[0][1] + ("a" if candidates[0][1] == "c" else "c")]

        elif len(candidates) == 2:
            # Tie-break criterion: Attach to more informative premise. Does not apply if premises have equal mood.
            orders = [mood + c[1] + ("a" if c[1] == "c" else "c") for c in candidates]
            if premises[0][0] == premises[1][0]:
                return orders
            info = [self.informativeness[prem[0]] for prem in premises]
            return [orders[info.index(max(info))]]

    def predict(self, syllogism):
        # Apply min heuristic
        min_mood = self.f_min_heuristic[syllogism[:2]]
        min_conclusions = self.attachment(min_mood, syllogism)
        conclusions = min_conclusions
        if random.random() < self.params["p-entailment"]:
            # Apply p-entailment
            pent_mood = self.f_p_entailment[min_mood]
            pent_conclusions = self.attachment(pent_mood, syllogism)
            conclusions = pent_conclusions
        # Respond NVC if not confident enough in min-conclusion
        if random.random() >= self.params["confidence" + self.max_premise(syllogism)[0]]:
            return ["NVC"]
        return conclusions
