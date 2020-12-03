
import numpy as np
from itertools import product
import math
from data_structures import *


class TaskProcessor:
    """
    Class for processing 2020_kaltenbl logic task
    """

    def __init__(self):
        # Map to store information(variables and premises as trees) of tasks
        # {task: list(variables, premise trees)}
        self.taskMap = {}
        # Map to store feature vectors of the tasks
        # {task: list(features values)}
        self.feature_vectors = {}
        # Map which sores the the classifications of each choice for a given task
        # {task: {choice: list(classifications)}}
        self.choice_classes = {}

    def get_variables(self, task):
        """
        :param task: string representation of logic task
        :return: variables of a specific task - sorted list(variables|string-lowercase)
        """
        variables = set()
        premises = task.split("/")
        for p in premises:
            x = p.split(";")
            for y in x:
                y = y.lower()
                if (y != 'iff') and (y != 'if') and (y != 'and') and (y != 'or') and (y != 'not'):
                    variables.add(y)
        var = list(variables)
        return sorted(var)

    def get_number_variables(self, task):
        """
        :param task: string representation of logic task
        :return: number of different variables within task
        """
        return len(self.get_variables(task))

    def get_premises(self, task):
        """
        :param task: string representation of logic task
        :return: list of premises - list(premises|string)
        """
        return task.split("/")

    def get_number_premises(self, task):
        """
        :param task: string representation of logic task
        :return: number of premises
        """
        return len(task.split("/"))

    def get_premise_components(self, premise):
        return premise.split(";")

    def add_task_to_task_map(self, task):
        """
        adds a task to the taskMap - {key = task: value = list(list(variables), premiseTree- Objects}
        :param task: task as string
        """
        if task not in self.taskMap:
            self.taskMap[task] = []
            y = self.get_variables(task)
            self.taskMap[task].append(y)
            z = self.get_premises(task)
            for t in z:
                p = self.get_premise_components(t)
                tree = PremiseTree(y)
                tree.parse_premise_to_tree(p)
                self.taskMap[task].append(tree)

    def create_feature_vector(self, task):
        """
        creates a feature vector representing the task
        :param task: task in string form
        :return: feature vector
        """
        vector = np.zeros(12)
        tmp = self.taskMap[task]
        premise_trees = tmp[1:]
        # more than two variables
        if len(tmp[0]) > 2:
            vector[0] = 1
        else:
            vector[0] = 0
        # check for literal premises
        for x in premise_trees:
            if len(x.listOfNodes) <= 3:
                vector[1] = 1
                if len(x.listOfNodes) == 2:
                    vector[2] = 1
                else:
                    vector[3] = 1
        # check for iff or if node
        for x in premise_trees:
            for y in x.listOfNodes:
                if y.name == 'iff' or y.name == 'if':
                    vector[4] = 1.3
                    if y.name == 'iff':
                        vector[5] = 1.2
                    else:
                        vector[6] = 1.3
        # check for or
        for x in premise_trees:
            for y in x.listOfNodes:
                if y.name == 'or':
                    vector[7] = 1
        # check for and
        for x in premise_trees:
            for y in x.listOfNodes:
                if y.name == 'and':
                    vector[8] = 1
        # check for not in non literal premise
        for x in premise_trees:
            if len(x.listOfNodes) <= 3:
                continue
            for y in x.listOfNodes:
                if y.name == 'not':
                    vector[9] = 2.2
        # check for positioning of the variables in relation to the connectives
        if self.literal_premise_is_left_of_connector(premise_trees):
            vector[10] = 1
            vector[11] = 0
        elif self.literal_premise_is_right_of_connector(premise_trees):
            vector[10] = 0
            vector[11] = 1
        else:
            vector[10] = 0
            vector[11] = 0

        return vector

    def add_feature_vector(self, task):
        """
        Adds a feature_vector corresponding to task to map
        :param task: task to which the feature_vector gets added
        """
        tmp = self.create_feature_vector(task)
        self.feature_vectors[task] = tmp

    def literal_premise_is_left_of_connector(self, premise_trees):
        """
        checks if a literal premise is left child of a operator in another premise
        :param premise_trees: representing a task
        :return: True if literal premise is left child
        """
        literal = None
        for x in premise_trees:
            if len(x.listOfNodes) <= 3:
                literal = x.leafList[0].name
                break
        if literal is None:
            return False
        for x in premise_trees:
            if len(x.listOfNodes) > 3:
                y = x.listOfNodes[0].leftChild.leftChild
                if y.name == literal:
                    return True
        return False

    def literal_premise_is_right_of_connector(self, premise_trees):
        """
        checks if a literal premise is right child of a operator in another premise
        :param premise_trees: representing a task
        :return: True if literal premise is right child
        """
        literal = None
        for x in premise_trees:
            if len(x.listOfNodes) <= 3:
                literal = x.leafList[0].name
                break
        if literal is None:
            return False
        for x in premise_trees:
            if len(x.listOfNodes) > 3:
                y = x.listOfNodes[0].leftChild.rightChild
                if y is None:
                    return False
                if y.name == literal:
                    return True
        return False

    def assign_choice_classes(self, task, choices, logic_prediction):
        """
        assigns classifications to possible choices for a task
        :param task: task for which the choices shall be classified
        :param choices: choices which shall be classified
        :param logic_prediction: choice which is the logic prediction for the task
        """
        if task not in self.choice_classes:
            choice_class = {}
            for x in choices:
                choice_class[x] = []
                # check if choice is the logic choice
                if x.lower() == logic_prediction:
                    choice_class[x].append(ChoiceClasses.logic)
                # check if choice is 'nothing'
                if x.lower() == 'nothing':
                    choice_class[x].append(ChoiceClasses.nothing)
                # check if choice is negative literal premise
                if self.check_for_negative_literal_premise(task, x):
                    choice_class[x].append(ChoiceClasses.negative_literal_premise)
                # check if choice has same sign as premise
                if self.check_for_atmospheric_literal(task, x):
                    choice_class[x].append(ChoiceClasses.atmospheric)
                if self.feature_vectors[task][6] >= 1:
                    if self.check_for_if_to_iff(self.taskMap[task], task, x):
                        choice_class[x].append(ChoiceClasses.if_to_iff)
                if self.feature_vectors[task][5] >= 1:
                    if self.check_for_iff_to_if(self.taskMap[task], task, x):
                        choice_class[x].append(ChoiceClasses.iff_to_if)
                if self.check_for_ignore_literal_not(self.taskMap[task], task, x):
                    choice_class[x].append(ChoiceClasses.ignore_not)
                if self.check_for_or_to_and(self.taskMap[task], task, x):
                    choice_class[x].append(ChoiceClasses.or_to_and)
                if self.check_for_anti_atmospheric(task, x):
                    choice_class[x].append(ChoiceClasses.anti_atmospheric)
                # check if choice is not classified
                if not choice_class[x]:
                    choice_class[x].append(ChoiceClasses.not_classified)
                self.choice_classes[task] = choice_class

    def check_for_negative_literal_premise(self, task, choice):
        """
        checks if a choice is a negation of a premise of the task
        :param task: task for which a choice will be classified
        :param choice: choice which shall be classified
        :return: True if a choice is a negation of a premise of the task
        """
        premisetrees = self.taskMap[task][1:]
        for x in premisetrees:
            if len(x.listOfNodes) == 3:
                leaf = x.leafList[-1]
                if leaf.name == choice.lower():
                    return True
            if len(x.listOfNodes) == 2:
                leaf = x.leafList[-1]
                comp = 'not;' + leaf.name
                if comp == choice.lower():
                    return True
        return False

    def check_for_atmospheric_literal(self, task, choice):
        """
        Check if a choice has the same atmosphere as a premise of the task
        - ergo it has the same 'sign'
        :param task: task for which a choice will be classified
        :param choice: choice which shall be classified
        :return: True if a choice has the same atmosphere as a premise of the task
        """
        if self.check_for_negative_literal_premise(task, choice) is True:
            return False
        premisetrees = self.taskMap[task][1:]
        for x in premisetrees:
            if len(x.listOfNodes) == 3:
                y = choice.split(";")
                if len(y) == 2:
                    return True
            if len(x.listOfNodes) == 2:
                y = choice.split(";")
                if len(y) == 1 and y[0].lower() != 'nothing':
                    return True
        return False

    def check_for_anti_atmospheric(self, task, choice):
        if self.check_for_negative_literal_premise(task, choice) is True:
            return False
        premisetrees = self.taskMap[task][1:]
        for x in premisetrees:
            if len(x.listOfNodes) == 3:
                y = choice.split(";")
                if len(y) == 1 and y[0].lower() != 'nothing':
                    return True
            if len(x.listOfNodes) == 2:
                y = choice.split(";")
                if len(y) == 2:
                    return True
        return False

    def check_for_if_to_iff(self, task_map_entry, task, choice):
        """
        check if a choice would is the result of a misinterpretaion of a implication
        :param task_map_entry: holds the premise trees of a given task
        :param task: task for which a choice will be classified
        :param choice: choice which shall be classified
        :return: True if the choice would be chosen if an implication is
                 is interpreted as an equivalence operation by a participant
        """
        if_to_iff_pred = Predictor.alternate_prediction(task_map_entry, task, alt_if=True)
        if_to_iff_pred_false = Predictor.alternate_prediction(task_map_entry, task)
        if if_to_iff_pred[0] == if_to_iff_pred_false[0]:
            return False
        if choice.lower() == if_to_iff_pred[0].lower():
            return True
        else:
            return False

    def check_for_iff_to_if(self, task_map_entry, task, choice):
        """
        check if a choice would is the result of a misinterpretaion of a equivalence operator
        :param task_map_entry: holds the premise trees of a given task
        :param task: task for which a choice will be classified
        :param choice: choice which shall be classified
        :return: True if the choice would be chosen if an equivalence operation is
                is interpreted as an implication by a participant
        """
        iff_to_if_pred = Predictor.alternate_prediction(task_map_entry, task, alt_iff=True)
        iff_to_if_pred_false = Predictor.alternate_prediction(task_map_entry, task)
        if iff_to_if_pred[0] == iff_to_if_pred_false[0]:
            return False
        if choice.lower() == iff_to_if_pred[0].lower():
            return True
        else:
            return False

    def check_for_or_to_and(self, task_map_entry, task, choice):
        """
        check if a choice would is the result of a misinterpretaion of a equivalence operator
        :param task_map_entry: holds the premise trees of a given task
        :param task: task for which a choice will be classified
        :param choice: choice which shall be classified
        :return: True if the choice would be chosen if an equivalence operation is
                is interpreted as an implication by a participant
        """
        or_to_and_pred = Predictor.alternate_prediction(task_map_entry, task, alt_or=True)
        or_to_and_false = Predictor.alternate_prediction(task_map_entry, task)
        if or_to_and_pred[0] == or_to_and_false[0]:
            return False
        if choice.lower() == or_to_and_pred[0].lower():
            return True
        else:
            return False

    def check_for_ignore_literal_not(self, task_map_entry, task, choice):
        """
        check if a choice would is the result of ignoring a not operation in a literal premise
        :param task_map_entry: holds the premise trees of a given task
        :param task: task for which a choice will be classified
        :param choice: choice which shall be classified
        :return: True if the choice would be chosen if a not operation is ignored
        """
        ignore_not_pred = Predictor.alternate_prediction(task_map_entry, task, alt_not=True)
        ignore_not_pred_false = Predictor.alternate_prediction(task_map_entry, task)
        if len(ignore_not_pred) == 1:
            if ignore_not_pred[0] == ignore_not_pred_false[0]:
                return False
            if choice.lower() == ignore_not_pred[0].lower():
                return True
            else:
                return False


class Predictor:

    def __init__(self, task_processor):
        self.task_processor = task_processor

    @staticmethod
    def create_assignment_list(variables):
        """
        creates a list of all possible 0,1 assignments for the given variables
        :param variables: given variables
        :return: list of all possible assignments
        """
        n = len(variables)
        a = list(product(range(2), repeat=n))
        return a

    @staticmethod
    def calculate_premise_vector(premise_tree, assign, alternate=False):
        """
        Calculates a vector which holds all the resulting truth values of a premise tree
        according to all the possible assignments for the variables
        :param premise_tree: tree representing a premise of a task
        :param assign: list of all the assignments
        :return: vector of truth values (0,1) each component representing one assignment
        """
        vec = []
        for x in assign:
            val = premise_tree.calculate_value_for_assignment(x, alternate)
            premise_tree.clear_assignment()
            if val is True:
                vec.append(1)
            else:
                vec.append(0)
        return np.array(vec)

    @staticmethod
    def alternate_prediction(task_map_entry, key, alt_if=False, alt_iff=False, alt_not=False, alt_or=False):
        """
        calculates a 'logic' prediction under the assumptions that certain logic operators
        get evaluated wrongly - is only used in the classification process of the choices
        :param task_map_entry: has the premise trees of a given task
        :param key: task
        :param alt_if: if True an if-operator will be seen as an iff operator
        :param alt_iff: iff True an iff-operator will be seen as an if operator
        :param alt_not: if True a not operator will be ignored
        :param alt_or: if True an or operator will be seen as an and operator
        :return: a list of possible choices for a prediction
        """
        prediction = None
        tree_vectors = []
        # get variables for the given task
        variables = task_map_entry[0]
        # create list with all possible assignments for the variables
        assign = Predictor.create_assignment_list(variables)
        # get the assignment vectors for each premise in the task
        for x in task_map_entry[1:]:
            if len(x.listOfNodes) == 3:
                tree_vectors.append(Predictor.calculate_premise_vector(x, assign, alternate=alt_not))
                continue
            for y in x.listOfNodes:
                if y.name == 'if':
                    tree_vectors.append(Predictor.calculate_premise_vector(x, assign, alternate=alt_if))
                    break
                if y.name == 'iff':
                    tree_vectors.append(Predictor.calculate_premise_vector(x, assign, alternate=alt_iff))
                    break
                    #####
            for y in x.listOfNodes:
                if y.name == 'or':
                    tree_vectors.append(Predictor.calculate_premise_vector(x, assign, alternate=alt_or))
                    break
            else:
                tree_vectors.append(Predictor.calculate_premise_vector(x, assign, alternate=False))
        n = len(assign)
        m = len(tree_vectors)
        # calculate which assignments set the task to true
        sum_vec = np.zeros((n,), dtype=int)
        for y in tree_vectors:
            sum_vec += y
        relevant_idx = []
        for i in range(n):
            if sum_vec[i] == m:
                relevant_idx.append(i)
        if not relevant_idx:
            prediction = ['nothing']
            return prediction
        # calculate possible inferences for the prediction
        prediction_candidate = []
        for var_idx in range(len(variables)):
            # print(assign)
            # print(relevant_idx)
            t = assign[relevant_idx[0]][var_idx]
            is_candidate = True
            for j in relevant_idx:
                if assign[j][var_idx] != t:
                    is_candidate = False
                    break
            if is_candidate is True:
                string = ''
                if t == 0:
                    string += 'not;'
                    string += variables[var_idx]
                else:
                    string += variables[var_idx]
                prediction_candidate.append(string)
        if not prediction_candidate:
            prediction = ['nothing']
            return prediction
        # remove trivial predictions
        premises = key.split("/")
        for p in premises:
            p = p.lower()
            if p in prediction_candidate:
                prediction_candidate.remove(p)
        # calculate final prediction
        if not prediction_candidate:
            prediction = ['nothing']
        if prediction_candidate:
            prediction = prediction_candidate
        return prediction


class LogicPredictor(Predictor):
    """
    Class for calculating predictions - based on formal 2020_kaltenbl logic
    uses truth tables
    """

    def __init__(self, task_processor):
        super().__init__(task_processor)
        self.predictionMap = {}

    def calculate_predictions(self, key):
        """
        Calculates the formal logical prediction for a given task
        :param key: task as string
        :return: prediction for the task as string
        """
        if key in self.predictionMap:
            return self.predictionMap[key]
        tree_vectors = []
        # get variables for the given task
        variables = self.task_processor.taskMap[key][0]
        # create list with all possible assignments for the variables
        assign = self.create_assignment_list(variables)
        # get the assignment vectors for each premise in the task
        for x in self.task_processor.taskMap[key][1:]:
            tree_vectors.append(self.calculate_premise_vector(x, assign))
        n = len(assign)
        m = len(tree_vectors)
        # calculate which assignments set the task to true
        sum_vec = np.zeros((n,), dtype=int)
        for y in tree_vectors:
            sum_vec += y
        relevant_idx = []
        for i in range(n):
            if sum_vec[i] == m:
                relevant_idx.append(i)
        if not relevant_idx:
            self.predictionMap[key] = 'nothing'
        # calculate possible inferences for the prediction
        prediction_candidate = []
        for var_idx in range(len(variables)):
            t = assign[relevant_idx[0]][var_idx]
            is_candidate = True
            for j in relevant_idx:
                if assign[j][var_idx] != t:
                    is_candidate = False
                    break
            if is_candidate is True:
                string = ''
                if t == 0:
                    string += 'not;'
                    string += variables[var_idx]
                else:
                    string += variables[var_idx]
                prediction_candidate.append(string)
        if not prediction_candidate:
            self.predictionMap[key] = 'nothing'
        # remove trivial predictions
        premises = self.task_processor.get_premises(key)
        for p in premises:
            p = p.lower()
            if p in prediction_candidate:
                prediction_candidate.remove(p)
        # calculate final prediction
        s = ''
        for h in prediction_candidate:
            s += h
            s += '+'
        s = s[:-1]
        if s == '':
            s = 'nothing'
        self.predictionMap[key] = s
        return self.predictionMap[key]


class VectorPredictor(Predictor):

    def __init__(self, task_processor, finished_tasks):
        super().__init__(task_processor)
        self.finished_tasks = finished_tasks
        self.choices_given = None


    def cos_sim(self, vec1, vec2):
        """
        calculate the cosine similarity between to vectors
        :param vec1: vector 1
        :param vec2: vector 2
        :return: cosine similarity between vec1 and vec2
        """
        dot = np.dot(vec1, vec2)
        lenvec1 = math.sqrt(np.dot(vec1, vec1))
        lenvec2 = math.sqrt(np.dot(vec2, vec2))
        return dot / (lenvec1 * lenvec2)

    def top_most_similar_tasks(self, task):
        """
        Gets the most similar tasks for a task which were already finished
        :param task: task for which similar task will be calculated
        :return: list of the most similar tasks - list element consists of
                 task-sting and similarity value
        """
        if self.finished_tasks:
            similar = []
            for x in self.finished_tasks:
                y = self.task_processor.feature_vectors[task]
                z = self.task_processor.feature_vectors[x[0]]
                sim = self.cos_sim(y, z)
                if self.cos_sim(y, z) == self.cos_sim(y, y):
                    sim = sim * 3
                similar.append([x[0], sim])
            tmp = sorted(similar, key=lambda v: v[1], reverse=True)
            res = []
            i = 0
            while i < len(tmp) and tmp[i][1] > 0.7:
                res.append(tmp[i])
                i += 1
            return res
        else:
            return None

    def calculate_prediction_from_top(self, task):
        """
        Calculate the prediction for a task
        :param task: task fro which the prediction will be calculated
        :return: choice prediction
        """
        sim = self.top_most_similar_tasks(task)
        logic_choice = None
        # calculate logic choice
        for key in self.task_processor.choice_classes[task]:
            for x in self.task_processor.choice_classes[task][key]:
                if x is ChoiceClasses.logic:
                    logic_choice = key
                    break
        # if no similar task exist default to logic response
        if sim is None:
            choice = self.calculate_default_choice(task, logic_choice)
            return choice
        elif not sim:
            choice = self.calculate_default_choice(task, logic_choice)
            return choice
        elif len(self.finished_tasks) < 6:
            choice = self.calculate_default_choice(task, logic_choice)
            return choice
        elif len(sim) < 2 and sim[0][1] < 0.8:
            choice = self.calculate_default_choice(task, logic_choice)
            return choice
        else:
            # calculate response from similar tasks

            # normalize similarity weights
            summ = 0
            for x in sim:
                summ += x[1]
            for x in sim:
                x[1] = x[1] / summ
            classes_of_truth = []  # holds the classes of the response of a task
            #  similar to the current task and the similarity
            for x in sim:
                for y in self.finished_tasks:
                    if x[0] == y[0]:
                        classes_of_truth.append([self.get_choice_class_of_truth(y[0], y[1]), x[1]])
            if not classes_of_truth:  # default logic
                choice = self.calculate_default_choice(task, logic_choice)
                return choice
            choice = self.get_choice_for_task2(self.order_classes_of_truth(classes_of_truth), task)
            if choice is None:
                return self.calculate_default_choice(task, logic_choice)
        return choice

    def get_choice_class_of_truth(self, task, truth):
        """
        Gets the classifications of a response already given by the current participant
        :param task: task for which the classifications of the given response shall be gotten
        :param truth: response for which the classifications shall be gotten
        :return: classifications of a already given response
        """
        return self.task_processor.choice_classes[task][truth]

    def order_classes_of_truth(self, classes_of_truth):
        """
        orders the classes of the already given responses according
        their frequency and similarity
        :param classes_of_truth: classifications of the already given responses
        :return: map of classifications and their frequency and similarity
        """
        class_of_truth_map = {}
        for x in classes_of_truth:
            for y in x[0]:
                if y not in class_of_truth_map:
                    class_of_truth_map[y] = x[1] * self.get_weight(y)
                else:
                    class_of_truth_map[y] += x[1] * self.get_weight(y)

        return class_of_truth_map

    def get_weight(self, classifier):
        """
        Gives the weight of a classifier
        :param classifier: the classifier
        :return: the weight of the classifier
        """
        if classifier is ChoiceClasses.atmospheric:
            return 0.9
        elif classifier is ChoiceClasses.negative_literal_premise:
            return 0.1
        elif classifier is ChoiceClasses.anti_atmospheric:
            return 0.8
        elif classifier is ChoiceClasses.nothing:
            return 0.91
        elif classifier is ChoiceClasses.if_to_iff:
            return 1.5
        elif classifier is ChoiceClasses.iff_to_if:
            return 0.7
        elif classifier is ChoiceClasses.ignore_not:
            return 0.5
        elif classifier is ChoiceClasses.logic:
            return 0.9
        elif classifier is ChoiceClasses.or_to_and:
            return 1.2
        else:
            return 0


    def get_choice_for_task2(self, classes_of_truth, task):
        """
        Calculate the prediction for a task based on the given responses of similar tasks
        :param classes_of_truth: map of classifications and their frequency and similarity
        :param task: task for which a prediction shall be calculated
        :return: choice which will be predicted
        """
        choices = self.task_processor.choice_classes[task]
        choice_map = {}
        for x in choices:
            choice_map[x] = 0
            for y in choices[x]:
                if y in classes_of_truth:
                    choice_map[x] += classes_of_truth[y]
                else:
                    continue
        temp = sorted(choice_map.items(), key=lambda v: v[1], reverse=True)
        tmp = self.reevaluate(temp, task)
        candidates = []
        i = 1
        max_score = tmp[0][1]
        candidates.append(tmp[0][0])
        while i < len(tmp):
            if abs(tmp[i][1] - max_score) < 0.1:
                candidates.append(tmp[i][0])
                i += 1
            else:
                break
        if len(candidates) == 1:
            choice = candidates[0]
            return choice
        else:
            choice = self.decide_between_candidates(candidates, task)
        return choice

    def decide_between_candidates(self, candidates, task):
        """
        Decides between multiple choices with equal score
        :param candidates: choices with same prediction score
        :param task: task for which a prediction shall be calculated
        :return: Choice which shall be return as prediction
        """
        for x in ChoiceClasses:
            for y in candidates:
                z = self.task_processor.choice_classes[task][y]
                for v in z:
                    if v == x:
                        return y

    def reevaluate(self, choices, task):
        """"
        Reevaluates the choices based upon  the overlap of
        the choices given by the other subjects
        :param task: task for which a prediction shall be calculated
        :param choices: possible choices and their current score
        :return: Choice which shall be return as prediction
        """
        # number_finished = len(self.finished_tasks)
        overlap = self.calculate_overlap_mfa()
        if self.choices_given is None:
            return choices
        tmp = {}
        if task not in self.choices_given:
            return choices
        if not self.choices_given[task]:
            return choices
        temp = sorted(self.choices_given[task].items(), key=lambda v: v[1], reverse=True)
        summ = 0
        for x in temp:
            summ += x[1]
        for x in choices:
            for y in temp:
                if x[0] == y[0]:
                    new_score = x[1] + (y[1] / summ) * overlap
                    tmp[x[0]] = new_score
            if x[0] not in tmp:
                tmp[x[0]] = x[1]
        return sorted(tmp.items(), key=lambda v: v[1], reverse=True)

    def calculate_default_choice(self, task, logic_choice):
        """
        Calculates the default choice for the task
        :param task: task for which a prediction shall be calculated
        :param logic_choice: logical choice
        :return: default choice for task
        """
        if self.choices_given is None:
            return logic_choice
        if task not in self.choices_given:
            return logic_choice
        if not self.choices_given[task]:
            return logic_choice
        temp = sorted(self.choices_given[task].items(), key=lambda v: v[1], reverse=True)
        candidates = []
        i = 1
        max_score = temp[0][1]
        candidates.append(temp[0][0])
        while i < len(temp):
            if abs(temp[i][1] - max_score) < 0.1:
                candidates.append(temp[i][0])
                i += 1
            else:
                break
        if len(candidates) == 1:
            choice = candidates[0]
            return choice
        else:
            choice = self.decide_between_candidates(candidates, task)
        return choice

    def calculate_overlap_mfa(self):
        """
        Calculates the overlap factor between the choices of the
        individual and the majority
        :return:
        """
        overlap = 0
        if self.choices_given is None:
            return 0
        if len(self.finished_tasks) < 5:
            return 1
        for x in self.finished_tasks:
            temp = sorted(self.choices_given[x[0]].items(), key=lambda v: v[1], reverse=True)
            if x[1] == temp[0][0]:
                overlap += 1
        return overlap / len(self.finished_tasks)




