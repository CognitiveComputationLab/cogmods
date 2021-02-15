"""
CCobra Model - mSentential for propositional reasoning
(based on the mModalSentential by Guerth)

Sigma Parameter:
Is the probability that system 2 is engaged in making inferences
Defaults to 0.2 as stated used in the paper experiment

Paper:
Facts and Possibilities: A Model-Based Theory of Sentential Reasoning
"""
import ccobra
import random
from mSentential.assertion_parser import ccobra_to_assertion
from mSentential.reasoner import what_follows


class MentalModel(ccobra.CCobraModel):
    def __init__(self, name='mSentential σ = 0.2', sigma=0.2):
        super(MentalModel, self).__init__(
            name, ['propositional'], ['single-choice'])

        # Sigma changes the probability of engaging system 2
        self.sigma = sigma

        # Default reasoner type is system 1
        self.reasoner_type = "system 1"

        # Set of last predictions by system 1 & 2
        self.last_prediction = {"system 1": [],
                                "system 2": []}

    def predict(self, item, **kwargs):
        # print("------------------------------------------------------------------------------------------------")
        # print(item.identifier, item.sequence_number)
        # print("Task:", item.task)

        # Pre process the choices
        possible_choices = [x[0] for x in item.choices]

        # Pre process the task:
        pp_list = self.pre_processing(item.task)
        # print("pre_processing: ", pp_list)

        # Create a premises in CCobra syntax for the mSentential model to handle:
        premises = ccobra_to_assertion(pp_list)
        # print("Premises: ", premises)

        # Evaluate premises with system 1 & 2 with mSentential for further evaluation:
        # print("----------- SYSTEM 1 -----------")
        prediction_1nec, prediction_1pos = what_follows(premises, system=1)
        nec_predicted_choices1 = [x for x in prediction_1nec if x in possible_choices]
        pos_predicted_choices1 = [x for x in prediction_1pos if x in possible_choices]
        # print("nec_predicted_choices: ", nec_predicted_choices1)
        # print("pos_predicted_choices: ", pos_predicted_choices1)
        predicted_choices1 = nec_predicted_choices1 + pos_predicted_choices1
        # print("predicted_choices: ", predicted_choices1)

        # print("----------- SYSTEM 2 -----------")
        prediction_2nec, prediction_2pos = what_follows(premises, system=2)
        nec_predicted_choices2 = [x for x in prediction_2nec if x in possible_choices]
        pos_predicted_choices2 = [x for x in prediction_2pos if x in possible_choices]
        # print("nec_predicted_choices: ", nec_predicted_choices2)
        # print("pos_predicted_choices: ", pos_predicted_choices2)
        predicted_choices2 = nec_predicted_choices2 + pos_predicted_choices2
        # print("predicted_choices: ", predicted_choices2)

        # --------- Evaluate answer by mSentential ---------
        # If system 1 has more then one answer choose at random
        if len(predicted_choices1) > 1:
            choice1 = random.choice(predicted_choices1)
            self.last_prediction["system 1"] = [choice1]

        # If system 1 has no answer predict 'nothing'
        elif nec_predicted_choices1 == [] and pos_predicted_choices1 == []:
            self.last_prediction["system 1"] = [['nothing']]

        # If system 1 has an exact answer predict that answer
        else:
            self.last_prediction["system 1"] = predicted_choices1

        # If system 2 has more answers to choose from:
        if len(predicted_choices2) > 1:
            choice2 = random.choice(predicted_choices2)
            self.last_prediction["system 2"] = [choice2]

        # If system 2 has no answer predict 'nothing'
        elif nec_predicted_choices2 == [] and pos_predicted_choices2 == []:
            self.last_prediction["system 2"] = [['nothing']]

        # Else predict answer given by System 2
        else:
            self.last_prediction["system 2"] = predicted_choices2

        # print("----------- PREDICTION -----------")
        if self.reasoner_type == "system 1":
            # print("SYSTEM 1: ", self.last_prediction["system 1"])
            return self.last_prediction["system 1"]
        else:
            # print("SYSTEM 2: ", self.last_prediction["system 2"])
            return self.last_prediction["system 2"]

    def pre_train(self, dataset):
        if random.random() < self.sigma:
            # print("SYSTEM 2 engaged")
            self.reasoner_type = "system 2"
        # else:
        #      print("SYSTEM 1 engaged")

    def adapt(self, item, response, **kwargs):
        pass

    @staticmethod
    def pre_processing(task):
        """
        Take List of Lists of tasks and create flattened list:
        Combine List of List with 'and' to create premises

        Arguments:
            task {{list}{list}} -- list of lists of premise strings

        Returns:
            premises {list} -- list premise strings
        """

        premises = []
        for i in range(0, len(task)-1):
            premises.append('and')
        for sublist in task:
            for items in sublist:
                premises.append(items)
        return premises
