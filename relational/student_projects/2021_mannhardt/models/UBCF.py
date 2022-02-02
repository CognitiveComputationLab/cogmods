# User-based Collaborative Filtering for preferences in belief revision

import ccobra
from taskencoder import *
import numpy as np
import math

MIN_NEIGHBORS = 5
MIN_ANSWERS = 5

allEvents = ["ABCLeftCABCACAB", "ABCLeftCACABBCA", "ABCRightACBCACAB", "ABCRightACCABBCA",
             "BACLeftCBACBCBA", "BACLeftCBCBAACB", "BACRightBCACBCBA", "BACRightBCCBAACB"]
WEIGHTS_ADD = [1,1,0.4,0.2,0.1]


class UBCFModel(ccobra.CCobraModel):
    def __init__(self, name="UBCF"):
        super(UBCFModel, self).__init__(name, ["spatial-relational"], ["single-choice", "verify"])
        
        # saves response vector for each user to calc euclidean similarity
        # e.g. e.g. eventDict{2 : {'BACRightBCCBAACB': {'CBA': 2, 'ACB': 4}, 'ABCLeftCACABBCA': {'CAB': 10, 'BCA': 2}, etc.}}
        self.userVectors = dict()
        self.currentAnswers = dict()

        # values are differences of answer count
        self.userVec = dict()

        # saves most similar subjects as list e.g. [(17, 415), (4, 412), (12, 411), (9, 407), etc.]
        self.similarUsers = list()

        # MFA
        self.mfa_population = dict()

        # start with including all 19 neighbors
        self.NEIGHBORS = 20
        self.WEIGHTS = [1, 1, 1, 1, 1]

    def predict(self, item, **kwargs):
        task_encoding = task_to_string(item.task) + task_to_string(item.choices)
        ident = item.identifier

        # get mfa prediction
        population_prediction = self.get_mfa_prediction(item, self.mfa_population)
        
        # get user prediction if available
        UBCF_prediction, answerCount = self.get_UBCF_prediction(task_encoding, ident)

        if UBCF_prediction is not None and answerCount > MIN_ANSWERS:
            return UBCF_prediction
        elif population_prediction is not None:
            return population_prediction

        # Return a random response if no MFA data is available
        return item.choices[np.random.randint(0, len(item.choices))]

    def get_UBCF_prediction(self, event, ident):
        self.similarUsers.clear()
        if event not in self.currentAnswers:
            return None, None
        choice1 = list(self.currentAnswers[event].keys())[0]
        choice2 = list(self.currentAnswers[event].keys())[1]

        # return None if subject has no preference
        if self.currentAnswers[event][choice1] == self.currentAnswers[event][choice2]:
            return None, None
        currentDiff = self.currentAnswers[event][choice1] - self.currentAnswers[event][choice2]
        ind = allEvents.index(event)
        for key, value in self.userVec.items():
            self.similarUsers.append((key, abs(currentDiff-value[ind])))
        self.similarUsers = sorted(self.similarUsers, key=lambda elem: elem[1])

        neighbors = [key[0] for key in self.similarUsers][:self.NEIGHBORS]
        # gradually decrease neighborhood size
        if self.NEIGHBORS > MIN_NEIGHBORS:
            self.NEIGHBORS -= 1
        answers = dict()
        ind = 0
        for user in neighbors:
            if ind == len(self.WEIGHTS):
                break
            try:
                for answer in list(self.currentAnswers[event].keys()):
                    if answer not in answers:
                        answers[answer] = 0
                    # add WEIGHTS_ADD to WEIGHTS to weight similar users responses more and more
                    # heavily over time
                    if self.NEIGHBORS == MIN_NEIGHBORS:
                        answers[answer] += self.WEIGHTS[ind]*self.userVectors[user][event][answer]
                        self.WEIGHTS = [sum(x) for x in zip(self.WEIGHTS, WEIGHTS_ADD)]
                    else:
                        answers[answer] += self.userVectors[user][event][answer]
            except:
                return None, None
            ind += 1
        sumOfChoices = 0
        for key, value in self.currentAnswers.items():
            for _, v in value.items():
                sumOfChoices += v
        if answers[choice1] == answers[choice2]:
            return None, None
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
        if event not in allEvents:
            return
        for choice in item.choices:
            if choice[0][0] not in self.currentAnswers[event]:
                self.currentAnswers[event][choice[0][0]] = 1
        # increase answer-count for the current subject
        self.currentAnswers[event][target[0][0]] += 1

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

                userId = task_data['item'].identifier
                event = list_to_string(item.task) + list_to_string(item.choices)
                if event not in allEvents:
                    continue
                if userId not in self.userVectors:
                    self.userVectors[userId] = dict()
                    # fill subject dict
                    for event in allEvents:
                        self.userVectors[userId][event] = dict()
                        self.userVectors[userId][event][event[-6:-3]] = 1
                        self.userVectors[userId][event][event[-3:]] = 1
                if isinstance(response, list):
                    try:
                        self.userVectors[userId][event][response[0][0]] += 1
                    except:
                        continue
                else:
                    self.userVectors[userId][event][response] += 1

        # fill subject dict
        for task in allEvents:
            self.currentAnswers[task] = dict()
            self.currentAnswers[task][task[-6:-3]] = 1
            self.currentAnswers[task][task[-3:]] = 1
        for user in self.userVectors:
            self.userVec[user] = list()
            for task in allEvents:
                try:
                    key1 = list(self.userVectors[user][task].keys())[0]
                    key2 = list(self.userVectors[user][task].keys())[1]
                    diff = self.userVectors[user][task][key1] - self.userVectors[user][task][key2]
                    self.userVec[user].append(diff)
                except:
                    self.userVec[user].append(0)

    def end_participant(self, identifier, model_log, **kwargs):
        return
