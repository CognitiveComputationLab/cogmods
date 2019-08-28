import copy
import json
import os
import sys

import ccobra

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from modular_models.models.basic_models import PSYCOP
from modular_models.models.ccobra_models.interface import CCobraWrapper
from modular_models.models.ccobra_models.interface.ccobra_wrapper import CACHE_DIR
from modular_models.util import sylutil


class CCobraPSYCOP(CCobraWrapper, ccobra.CCobraModel):
    def __init__(self):
        CCobraWrapper.__init__(self, model=PSYCOP)
        ccobra.CCobraModel.__init__(self, "PSYCOP", ["syllogistic"], ["single-choice"])

    def predict_deterministic(self, syllogism):
        premises = self.model.encode_premises(syllogism,
                                        ex_implicatures=self.model.params["premise_implicatures_existential"],
                                        grice_implicatures=self.model.params["premise_implicatures_grice"])

        # 1. Try to get conclusions by applying forward rules
        fw_propositions = self.model.run_forward_rules(premises)
        fw_conclusions = []
        for prop in fw_propositions:
            for c in ccobra.syllogistic.RESPONSES:
                conclusion = self.model.encode_proposition(c, hat=False)
                if self.model.proposition_to_string(conclusion) == self.model.proposition_to_string(prop):
                    fw_conclusions.append(c)
        if len(fw_conclusions) != 0:
            return [(1 / len(fw_conclusions), c) for c in fw_conclusions]

        responses = []

        tentative_conclusions = [self.model.heuristic(syllogism) + ac for ac in ["ac", "ca"]]
        tcs_enc = [self.model.encode_proposition(tc, hat=False) for tc in tentative_conclusions]
        for tc_enc, tc in zip(tcs_enc, tentative_conclusions):
            self.model.subformulas = self.model.extract_all_atomic_subformulas(premises + [tc_enc])
            success = self.model.run_backward_rules(fw_propositions, tc_enc)
            if success:
                if self.model.params["conclusion_implicatures"]:
                    c_impl = sylutil.add_implicatures([tc], True, True)[1]
                    conclusion_impl = self.model.encode_proposition(c_impl, hat=False)
                    self.model.subformulas = self.model.extract_all_atomic_subformulas(
                        premises + [conclusion_impl])
                    success_impl = self.model.run_backward_rules(fw_propositions, conclusion_impl)
                    if success_impl:
                        responses.append((0.5, tc))
                        continue
                else:
                    responses.append((0.5, tc))
                    continue
            pg = self.model.params["guess"] * 0.5 * 1/8
            pn = (1 - self.model.params["guess"]) * 0.5
            responses.extend([(pg, "Aac"), (pg, "Aca"), (pg, "Iac"), (pg, "Ica"), (pg, "Eac"), (pg, "Eca"),
                                  (pg, "Oac"), (pg, "Oca")])
            responses.append((pn, "NVC"))
        responses_summed = [(0.0, c) for c in ccobra.syllogistic.RESPONSES]
        for (y, c) in responses:
            i = ccobra.syllogistic.RESPONSES.index(c)
            responses_summed[i] = (responses_summed[i][0] + y, responses_summed[i][1])
        return responses_summed

    def generate_predictions(self):
        """ Custom faster method because the PSYCOP model is both stochastic and slow """

        # Name of cache file is built from name of internal model
        cache_file = os.path.join(CACHE_DIR + self.model.__class__.__name__ + "_PREDICITON_CACHE.json")

        # Try loading predictions from cache
        try:
            predictions = json.load(open(cache_file, 'r'))
        except (IOError, ValueError):
            predictions = {}

        # No cache available - generate predictions regularly
        if predictions == {}:
            parameters_before = copy.deepcopy(self.model.params)
            predictions = {syllogism: [] for syllogism in ccobra.syllogistic.SYLLOGISMS}
            for syllogism in ccobra.syllogistic.SYLLOGISMS:
                print(syllogism)
                i = 0
                for param_configuration in self.configurations:
                    i += 1
                    self.model.set_params(param_configuration)
                    conclusions = self.predict_deterministic(syllogism)
                    y = []

                    for resp in ccobra.syllogistic.RESPONSES:
                        if resp in [c for (y, c) in conclusions]:
                            index = [c for (y, c) in conclusions].index(resp)
                            y.append([y for (y, c) in conclusions][index])
                        else:
                            y.append(0.0)

                    if sum(y) < 0.99 or sum(y) > 1.01:
                        raise Exception

                    predictions[syllogism].append(y)

            # Reset parameters
            self.model.set_params(parameters_before)

            # Write cache
            json.dump(predictions, open(cache_file, 'w'))

        return predictions
