import numpy as np

import ccobra
from taskencoder import *

def task_to_string(syntaxTree):
    """
    Changes e.g. ["a", "And", ["b", "Or", ["Not", "c"]]] to "a And b Or -c"
    Args:
        syntaxTree
    Returns:
        string - syntaxTree written as string
    """
    if isinstance(syntaxTree[0][0], bool):
        return syntaxTree[0][0]

    if isinstance(syntaxTree, str) or isinstance(syntaxTree, bool):
        return syntaxTree

    all = ""
    for elem in syntaxTree:
        all += task_to_string(elem)
    return all

def keywithmaxval(d):
     """ a) create a list of the dict's keys and values; 
         b) return the key with the max value"""  
     v=list(d.values())
     k=list(d.keys())
     return k[v.index(max(v))]

class IBCFModel(ccobra.CCobraModel):
    def __init__(self, name='IBCF'):
        super(IBCFModel, self).__init__(name, ["spatial-relational"], ["single-choice", "verify"])

        # Initialize member variables
        self.mfa_population = dict()

        # keys are task, value is list of lists
        self.task_preference = dict()
        
        # 4 works best
        self.MIN_SAMPLES = 4

    def predict(self, item, **kwargs):
        i = 0
        with open("itemsIndexes.txt", "w") as file:
            file.write("{")
            for key in self.mfa_population:
                file.write("\""+key + "\" : " + str(i) + ",\n")
                i += 1
            file.write("}")
        # Return the population MFA if available
        population_prediction = self.get_mfa_prediction(item, self.mfa_population)
        cat = kwargs['event'].split("_")[1]
        item_prediction = self.get_item_prediction(item, cat)
        if item_prediction is not None and item.sequence_number == 3:
            return item_prediction
        elif population_prediction is not None:
            return population_prediction

        # Return a random response if no MFA data is available
        return item.choices[np.random.randint(0, len(item.choices))]

    def get_item_prediction(self, item, cat):
        task_encoding = task_to_string(item.task) + task_to_string(item.choices)
        if task_encoding in self.task_preference:
            choices = self.task_preference[task_encoding]
            response = keywithmaxval(choices)
            if self.task_preference[task_encoding][response] >= self.MIN_SAMPLES:
                return response
        return None

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
    
    def adapt(self, item, target, **kwargs):
        event = kwargs['aux']['event']
        if item.sequence_number != 3:
            return
        task_encoding = task_to_string(item.task) + task_to_string(item.choices)
        response = task_to_string(target)
        if task_encoding not in self.task_preference:
            self.task_preference[task_encoding] = {response: 1}
            return
        if response not in self.task_preference[task_encoding]:
            self.task_preference[task_encoding][response] = 1
        else:
            self.task_preference[task_encoding][response] += 1

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
                    
    def end_participant(self, identifier, model_log, **kwargs):
        return
        if identifier == 11:
            print(self.task_preference)