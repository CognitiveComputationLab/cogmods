"""
CCobra Model - mSentential for propositional reasoning
(based on the mModalSentential by Guerth)

This model is tuned for better performance with the CCobra framework:
Generally system 2 performs much better then system 1, so system 2 always
provides the prediction, for this model. Further, a dict of answered questions
and responses is kept. With that said, a bool of answered 'nothing' is saved
as well. This works as an indicator if a participant is not able to
comprehended a given task. And with that, might not be able to comprehended
similar tasks.

Options:
1. Consistence: If System 2 predicts multiple possible answers, this checks if
                system 1 has predicted an answer and chooses the answer that is
                predicted by both systems. This provides prediction consistence
                between both systems.

2. Necessary:   mSentential provides both necessary and possible predictions
                for a given premise. Some participants may only provide an
                answer if it follows necessarily. So with that option it is
                possible to return 'nothing' if nothing follows necessarily.

3. Size_limit:  Possible working memory size limit. This option, when enabled
                checks the task for more or equal to 3 sentential connectives because
                this may lead to working memory overload of the participant and
                nothing is predicted.

4. Memory:      If this option is enabled the dict for answered questions and
                responses is used to predict answers if a question was already
                answered before.
"""
import ccobra
import random
from mSentential.assertion_parser import ccobra_to_assertion
from mSentential.reasoner import what_follows


class MentalModel(ccobra.CCobraModel):
    def __init__(self, name="mSentential tuned", c=True, n=True, s=True, m=True):
        # name = f"{'c' if c else ''}{'n' if n else ''}{'s' if s else ''}{'m' if m else ''}"
        super(MentalModel, self).__init__(
            name, ['propositional'], ['single-choice'])

        # Possible tuning options
        self.options = {"consistence": c,
                        "necessary": n,
                        "size_limit": s,
                        "memory": m}

        # Set of already answered questions for reference
        self.answered_questions = {}

        # Set of last predictions by system 1 & 2
        self.last_prediction = {"system 1": [],
                                "system 2": []}

        # Track if participant has answered nothing
        self.answered_nothing = False

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

            # 1. Check if system 1 had an answer
            if len(predicted_choices1) > 0 and self.options["consistence"]:

                # 1.1 If already answered the question return that answer once again
                if str(item.task) in self.answered_questions and self.options["memory"]:
                    self.last_prediction["system 2"] = self.answered_questions.get(str(item.task))
                else:
                    # 1.2 Else, check if answers by system 1 & 2 are the same and predict that answer
                    for x in predicted_choices1:
                        for y in predicted_choices2:
                            if x == y:
                                self.last_prediction["system 2"] = [x]

            # 2. Check if there are no necessary conclusions from System 1 & 2 and participant answered_nothing already
            elif nec_predicted_choices2 == [] and nec_predicted_choices1 == [] \
                    and self.answered_nothing and self.options["necessary"]:

                # 2.1 If already answered the question return that answer once again
                if str(item.task) in self.answered_questions and self.options["memory"]:
                    self.last_prediction["system 2"] = self.answered_questions.get(str(item.task))
                else:
                    for x in possible_choices:
                        if x == ['nothing']:
                            self.last_prediction["system 2"] = [['nothing']]

            # Else choose one answer at random
            else:
                # If already answered the question return that answer once again
                if str(item.task) in self.answered_questions and self.options["memory"]:
                    self.last_prediction["system 2"] = self.answered_questions.get(str(item.task))
                else:
                    choice2 = random.choice(predicted_choices2)
                    self.last_prediction["system 2"] = [choice2]

        # Else predict answer given by System 2
        else:
            self.last_prediction["system 2"] = predicted_choices2

        # --------- General prediction modifications ---------
        # Possible Working Memory Size Limit
        connectives = 0
        for sublist in item.task:
            for connective in sublist:
                # Count 'and' & 'or' sentential connectives
                if connective == 'and' or connective == 'or':
                    connectives += 1
                # Remove one for 'Not'
                # elif connective == 'Not':
                #     connectives -= 1

        # More then 3 sentential connectives may lead to working memory overload and nothing is predicted
        # Only if already answered_nothing
        if connectives >= 3 and self.answered_nothing and self.options["size_limit"]:
            # If already answered the question return that answer once again
            if str(item.task) in self.answered_questions and self.options["memory"]:
                self.last_prediction["system 2"] = self.answered_questions.get(str(item.task))
            else:
                self.last_prediction["system 2"] = [['nothing']]

        # print("----------- PREDICTION -----------")
        # print("SYSTEM 1: ", self.last_prediction["system 1"])
        # print("SYSTEM 2: ", self.last_prediction["system 2"])
        return self.last_prediction["system 2"]

    def adapt(self, item, response, **kwargs):
        # --------- Evaluate answers of participant ---------
        # Track Task and response
        self.track_answered_questions(item.task, response)

        # Track if participant has answered nothing
        if response == [['nothing']]:
            self.answered_nothing = True

    def track_answered_questions(self, task, response):
        """
        Take the task and response insert it in a dict for reference

        Arguments:
            task str({{list}{list}}) -- str of list of lists of premise strings
            response {list} -- list of a response
        """
        self.answered_questions[str(task)] = {}
        self.answered_questions[str(task)] = response

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
