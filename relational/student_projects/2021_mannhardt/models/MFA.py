import numpy as np

import ccobra
from taskencoder import *

class MFA(ccobra.CCobraModel):
    def __init__(self, name='MFA'):
        super(MFA, self).__init__(name, ["spatial-relational"], ["single-choice", "verify"])

        # Initialize member variables
        self.mfa_population = dict()

        self.item_count = dict()

    def predict(self, item, **kwargs):
        # Return the population MFA if available
        population_prediction = self.get_mfa_prediction(item, self.mfa_population)
        if population_prediction is not None:
            return population_prediction

        # Return a random response if no MFA data is available
        return item.choices[np.random.randint(0, len(item.choices))]

    def get_mfa_prediction(self, item, mfa_dictionary):  
        encoded_task = list_to_string(item.task) + list_to_string(item.choices)
        encoded_choices = [choice[0][0] for choice in item.choices]
        if encoded_task in mfa_dictionary:
            # Extract the potential MFA responses which are allowed in terms
            # of the possible response choices
            potential_responses = []
            for response, count in mfa_dictionary[encoded_task].items():
                if response in encoded_choices:
                    potential_responses.append((response, count))
                elif response == True or response == False:
                    potential_responses.append((response, count))

            # If potential responses are available, determine the one with
            # maximum frequency
            if potential_responses:
                max_count = -1
                max_responses = []
                for response, count in potential_responses:
                    if count > max_count:
                        max_count = count
                        max_responses = []

                    if count >= max_count:
                        max_responses.append(response)

                # In case of ties, draw the MFA response at random from the options
                # with maximum frequency
                encoded_prediction = max_responses[np.random.randint(0, len(max_responses))]
                return encoded_prediction

        # If no MFA response is available, return None
        return None

    def pre_train(self, dataset):
        # Iterate over subjects in the dataset
        for subj_data in dataset:
            # Iterate over the task for an individual subject
            for task_data in subj_data:
                # Create the syllogism object and extract the task and response encodings
                item = task_data['item']
                response = task_data['response']
                encoded_task = list_to_string(item.task) + list_to_string(item.choices)
                encoded_response = response[0][0]

                # Prepare the response counter for this task if not present already
                if encoded_task not in self.mfa_population:
                    self.mfa_population[encoded_task] = dict()

                # Increment the response count for the present task
                self.mfa_population[encoded_task][encoded_response] = \
                    self.mfa_population[encoded_task].get(encoded_response, 0) + 1