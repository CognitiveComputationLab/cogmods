# User-based Collaborative Filtering for preferences in belief revision

import ccobra
from taskencoder import *
import numpy as np
import math

MIN_NEIGHBORS = 2
MIN_USERS = 0


class UBCFModel(ccobra.CCobraModel):
    def __init__(self, name="UBCF"):
        super(UBCFModel, self).__init__(name, ["spatial-relational"], ["single-choice", "verify"])
        
        # saves response vector for each user to calc euclidean similarity
        # e.g. e.g. eventDict{2 : {'BACRightBCCBAACB': {'CBA': 2, 'ACB': 4}, 'ABCLeftCACABBCA': {'CAB': 10, 'BCA': 2}, etc.}}
        self.userVectors = dict()
        self.subjVectors = dict()

        # looks like:
        # {1: [0.090, 2.0, 9.0, 2.0, 12.0, 0.272, 0.1, 3.335, 12.0, 0.93], 2: [2.0, 0, etc.
        # values are quotients of answer count
        self.userVec = dict()
        self.subjVec = [0,0,0,0,0,0,0,0,0,0]

        # responses of current user
        self.currentUserVector = list()

        # saves most similar subjects as list e.g. [(17, 415), (4, 412), (12, 411), (9, 407), etc.]
        self.similarUsers = list()
        
        # set of all events e.g. 100_R_Incon_gu_abab_modle_factauf_revgle_lori_fact
        self.allEvents = set()
        
        # each events gets assigned a number e.g. eventDict['BACRightBCCBAACB'] = 1
        self.eventDict = dict()

        # Initialize member variables
        self.mfa_population = dict()
        
        self.NEIGHBORS = 20

    def predict(self, item, **kwargs):
        task_encoding = task_to_string(item.task) + task_to_string(item.choices)
        ident = item.identifier

        # get mfa prediction
        population_prediction = self.get_mfa_prediction(item, self.mfa_population)
        
        # get user prediction if available
        UBCF_prediction, allAnswers = self.get_UBCF_prediction(task_encoding, ident)

        if UBCF_prediction is not None and allAnswers > MIN_USERS:
            return UBCF_prediction
        elif population_prediction is not None:
            return population_prediction

        # Return a random response if no MFA data is available
        return item.choices[np.random.randint(0, len(item.choices))]

    def get_UBCF_prediction(self, event, ident):
        self.similarUsers.clear()
        if len(self.subjVec) == 0:
            return None, None
        for key, value in self.userVec.items():
            self.similarUsers.append((key, angle(value, self.subjVec)))
        self.similarUsers = sorted(self.similarUsers, key=lambda elem: elem[1])
        neighbors = [key[0] for key in self.similarUsers][:self.NEIGHBORS]
        if self.NEIGHBORS > MIN_NEIGHBORS:
            self.NEIGHBORS -= 1
        answers = dict()
        for user in neighbors:
            try:
                for answer in self.userVectors[user][event]:
                    if answer not in answers:
                        answers[answer] = 0
                    answers[answer] += self.userVectors[user][event][answer]
            except:
                return None, None
        sumOfChoices = 0
        for key, value in self.subjVectors.items():
            for _, v in value.items():
                sumOfChoices += v
        return keywithmaxval(answers), sumOfChoices

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
        if target[0][0] == "weiter" or item.sequence_number != 3:
            return
        event = task_to_string(item.task) + task_to_string(item.choices)
        for choice in item.choices:
            if choice[0][0] not in self.subjVectors[event]:
                self.subjVectors[event][choice[0][0]] = 1
        self.subjVectors[event][target[0][0]] += 1
        self.subjVec.clear()
        for task in self.allEvents:
            try:
                key1 = list(self.subjVectors[task].keys())[0]
                key2 = list(self.subjVectors[task].keys())[1]
                quot = self.subjVectors[task][key1] / self.subjVectors[task][key2]
                quot = round(quot, 4)
                self.subjVec.append(quot)
            except:
                self.subjVec.append(0)

    def pre_train(self, dataset):
        # Iterate over subjects in the dataset
        for subj_data in dataset:
            # Iterate over the task for an individual subject
            for task_data in subj_data:
                response = task_data['response']
                if response[0][0] == "weiter":
                    continue

                ###### MFA ######
                # Create the syllogism object and extract the task and response encodings
                item = task_data['item']
                encoded_task = list_to_string(item.task) + list_to_string(item.choices)
                encoded_response = response[0][0]

                # Prepare the response counter for this task if not present already
                if encoded_task not in self.mfa_population:
                    self.mfa_population[encoded_task] = dict()

                # Increment the response count for the present task
                self.mfa_population[encoded_task][encoded_response] = \
                    self.mfa_population[encoded_task].get(encoded_response, 0) + 1
                ###### MFA ######

                if task_data['item'].sequence_number != 3:
                    continue

                # vars of item object : dict_keys(['identifier', 'response_type', 'task_str', 'task', 'choices_str', 'choices', 'domain', 'sequence_number'])
                userId = task_data['item'].identifier
                event = list_to_string(item.task) + list_to_string(item.choices)
                self.allEvents.add(event)
                if userId not in self.userVectors:
                    self.userVectors[userId] = dict()
                if event not in self.userVectors[userId]:
                    self.userVectors[userId][event] = dict()
                for choice in item.choices:
                    if choice[0][0] not in self.userVectors[userId][event]:
                        self.userVectors[userId][event][choice[0][0]] = 1
                if isinstance(response, list):
                    self.userVectors[userId][event][response[0][0]] += 1
                else:
                    self.userVectors[userId][event][response] += 1
        self.allEvents = sorted(list(self.allEvents), key=lambda x: x)

        # fill subject dict
        for task in self.allEvents:
            self.subjVectors[task] = dict()
        for user in self.userVectors:
            self.userVec[user] = list()
            for task in self.allEvents:
                try:
                    key1 = list(self.userVectors[user][task].keys())[0]
                    key2 = list(self.userVectors[user][task].keys())[1]
                    quot = self.userVectors[user][task][key1] / self.userVectors[user][task][key2]
                    quot = round(quot, 5)
                    self.userVec[user].append(quot)
                except:
                    self.userVec[user].append(0)

        # fill eventDict
        for i in range(len(self.allEvents)):
            self.eventDict[self.allEvents[i]] = i
  
    def end_participant(self, identifier, model_log, **kwargs):
        return
