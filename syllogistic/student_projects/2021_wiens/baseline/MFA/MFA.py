import random
import ccobra

class MFAModel(ccobra.CCobraModel):
    """
    Returns the most frequent answers of the participants.
    Slightly changed code for verify domain. For original source code see
    https://github.com/CognitiveComputationLab/ccobra/blob/master/benchmarks/syllogistic/models/Baseline/MFA-Model/mfa_model.py
    """
    def __init__(self, name='MFAModel'):
        super(MFAModel, self).__init__(name, ["syllogistic"], ["verify"])
        self.mfa_population = dict()

    def pre_train(self, dataset):
        for subj_data in dataset:
            for task_data in subj_data:
                syllogism = ccobra.syllogistic.Syllogism(task_data['item'])
                encoded_task = syllogism.encoded_task
                response = task_data['response']
                encoded_target = syllogism.encode_response(task_data['item'].choices[0])
                syl_type = encoded_task + "-" + encoded_target

                if syl_type not in self.mfa_population:
                    self.mfa_population[syl_type] = dict()

                self.mfa_population[syl_type][response] = \
                    self.mfa_population[syl_type].get(response, 0) + 1

    def get_mfa_prediction(self, item, mfa_dictionary):
        syllogism = ccobra.syllogistic.Syllogism(item)
        encoded_task = syllogism.encoded_task
        encoded_target = syllogism.encode_response(item.choices[0])
        syl_type = encoded_task + "-" + encoded_target

        if syl_type in mfa_dictionary:
            potential_responses = []
            for response, count in mfa_dictionary[syl_type].items():
                potential_responses.append((response, count))

            if potential_responses:
                max_count = -1
                max_responses = []
                for response, count in potential_responses:
                    if count > max_count:
                        max_count = count
                        max_responses = []

                    if count >= max_count:
                        max_responses.append(response)

                encoded_prediction = max_responses[random.randint(0, len(max_responses) - 1)]
                return encoded_prediction

    def predict(self, item, **kwargs):
        population_prediction = self.get_mfa_prediction(item, self.mfa_population)

        if population_prediction is not None:
            return population_prediction

        return random.choice([True, False])
