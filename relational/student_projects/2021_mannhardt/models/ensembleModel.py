import numpy as np

import ccobra
from taskencoder import *


class EnsembleModel(ccobra.CCobraModel):
    def __init__(self, name='Ensemble'):
        super(EnsembleModel, self).__init__(name, ["spatial-relational"], ["single-choice", "verify"])

        # Initialize member variables
        self.mfa_population = dict()

        self.item_count = dict()
        
        # stores count of LO and LA relocations as e.g. 
        # [<LO>, <RO>, <LA>, <FA>, <Plausible>]
        # LocatedObject, ReferenceObject, LastAdded, FirstAdded, Plausible, LeftChoice, RightChoice
        self.preferences = [1, 0, 0, 0, 0, 0, 0]

    def predict(self, item, **kwargs):
        # Return the population MFA if available
        population_prediction = self.get_mfa_prediction(item, self.mfa_population)

        # Go for preference if seq is 3 and LO is not the dominant strategy
        if item.sequence_number == 3:
            max_value = max(self.preferences)
            # LO
            if self.preferences[0] == max_value:
                return population_prediction

            # Lastly Added
            elif self.preferences[2] == max_value:
                initialModel = item.task[0][0]
                retModel = ""
                # C on left side
                if initialModel[0] == "C":
                    retModel = initialModel[1:]+initialModel[0]
                # C on right side
                elif initialModel[2] == "C":
                    retModel = initialModel[2]+initialModel[:2]
                return retModel
            
            # RO-preference
            elif self.preferences[1] == max_value:
                # do the reverse of MFA, that is RO-relocation
                if population_prediction == item.choices[0][0][0]:
                    return item.choices[1][0][0]
                elif population_prediction == item.choices[1][0][0]:
                    return item.choices[0][0][0]
                
            # First added 
            elif self.preferences[3] == max_value:
                initialModel = item.task[0][0]
                retModel = ""
                # C on left side
                if initialModel[0] == "C":
                    retModel = initialModel[2]+initialModel[:2]
                # C on right side
                elif initialModel[2] == "C":
                    retModel = initialModel[1:]+initialModel[0]
                return retModel

            # return plausible model
            elif self.preferences[4] == max_value:
                pls = kwargs['event'].split("_")[-3]
                if pls.endswith("le"):
                    return item.choices[0][0][0]
                elif pls.endswith("ri"):
                    return item.choices[1][0][0]

            # return left choice
            elif self.preferences[5] == max_value:
                return item.choices[0][0][0]

            # return right choice
            elif self.preferences[6] == max_value:
                return item.choices[1][0][0]

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
    
    def adapt(self, item, truth, **kwargs):
        # [<LO>, <RO>, <LA>, <FA>, <Left>, <Right>]
        # LocatedObject, ReferenceObject, LastAdded, FirstAdded, LeftObject, RightObject, PlausibleModel
        if item.sequence_number != 3:
            return

        # LO/RO relocation
        if item.task[1][1] == truth[0][0][0] or item.task[1][1] == truth[0][0][2]:
            self.preferences[0] += 1
        else:
            self.preferences[1] += 1

        # last added object relocation
        if truth[0][0][1] != "C":
            self.preferences[2] += 1

        # first added object relocation
        if truth[0][0][1] == "C":
            self.preferences[3] += 1
            
        event = ""
        try:
            event = kwargs['aux']['event']
        except:
            event = kwargs['event']
        # plausibility
        if event.split("_")[-3].endswith("le"):
            if item.choices[0][0][0] == truth[0][0]:
                self.preferences[4] += 1
        if event.split("_")[-3].endswith("ri"):
            if item.choices[1][0][0] == truth[0][0]:
                self.preferences[4] += 1

        # relocate left choice
        if truth[0][0] == item.choices[0][0][0]:
            self.preferences[5] += 1

        # relocate right choice
        if truth[0][0] == item.choices[1][0][0]:
            self.preferences[6] += 1

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
                    
    def end_participant(self, subj_id, model_log, **kwargs):
        return
