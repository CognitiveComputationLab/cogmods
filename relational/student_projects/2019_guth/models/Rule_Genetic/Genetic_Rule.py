import ccobra
import RuleTestPerson
import random
import GARule
import Rule
from copy import deepcopy
import pickle


class RuleApproach(ccobra.CCobraModel):
    def __init__(self, exp_nr, name):
        """ Model constructor.
            This class represents the rule approach of my thesis.
            All benchmarks can be started with this class.
        """
        super(RuleApproach, self).__init__(name, ['spatial-relational'], ['single-choice', 'verify'])

        # set this to False if you want to retrain
        self.load = True

        # a list of persons which will finally be used to answer the questions in the predict function.
        self.persons = []
        # the number of rules with which every person will be initialized.
        self.start_rule_number = 10
        # the size of the population of each genetic algorithm/ for each person in the dataset.
        self.test_subject_number = 100
        # number of runs the genetic algorithm will perform.
        self.GA_runs = 200
        # the number represents the experiment/benchmark. 1 = singelchoice; 2 = verification; 3 = figural;
        # 4 = premiseorder
        self.exp_nr = exp_nr
        # name of file from which to load an already trained model
        self.load_file = name + ".p"
        if not self.load:
            # the name will be visible in the website for the data analyse.
            self.name += " StrRule: " + str(self.start_rule_number) + " Pop: " + str(self.test_subject_number) + \
                         " Runs: " + str(self.GA_runs)
        # this is a counter to count the number of answers already given.
        self.current_person_nr = 0
        # saves the best person in the genetic algorithm
        self.best_person = None
        # saves the number of questions asked in the experiment.
        self.nr_of_questions = 0
        # set this to True if you want to safe the trained model in a pickle.
        self.save = False
        # some string operations to be able to save intermediate data
        self.save_file = self.name + ".txt"
        self.save_file = self.save_file.replace(" ", "")
        self.save_file = self.save_file.replace(":", "")
        if self.save:
            with open(self.save_file, "w") as f:
                print(self.name, file=f)


    def start_participant(self, **kwargs):
        """ Model initialization method. Used to setup the initial state of
        its datastructures, memory, etc.
        Initializes the experiment participant by setting the person score for each person to 0.
        """
        self.current_person_nr = 0
        for each in self.persons:
            each.person_score = 0
            for rule in each.rule_set:
                rule.used_counter = 0

    def pre_train(self, dataset):
        """ Pre-trains the model based on one or more datasets.
            This calls the genetic algorithm to train the rule approach.
        """
        # this is to load intermediate data and is not needed for the genetic algorithm.
        if self.load:
            pickle_obj = pickle.load(open("Trained_Models/" + self.load_file, "rb"))
            self.persons = pickle_obj
            self.best_person = self.persons[0]
            return

        # initiate a GARule for each person
        genetic_algorithm = self.make_ga(dataset)
        genetic_algorithm.make_deep_copy()

        # do the evaluation, selection and mutation for each GARule/Person for the amount in GA_runs
        for i in range(0, self.GA_runs):
            print((i / self.GA_runs) * 100, "% ")
            last_time = time.time()
            last_percent = (i / self.GA_runs) * 100
            # Do the evaluation/selection and mutation.
            genetic_algorithm.evaluate(self.exp_nr)
            # every ten steps save the current state of the genetic algorithm.
            if i % 10 == 0:
                self.save_current_state(i, genetic_algorithm)
            genetic_algorithm.selection()
            genetic_algorithm.make_deep_copy()
            genetic_algorithm.mutation()
        # once done with the amount of mutation/evaluation and selection do one evaluation to get the best person on top
        # and then add the best person to the list of persons who will do the prediction.
        genetic_algorithm.evaluate(self.exp_nr)
        self.save_current_state(5555, genetic_algorithm)
        self.persons = genetic_algorithm.test_persons
        self.persons.sort(key=lambda x: x.person_score, reverse=True)
        # set the best person of the genetic algorithm.
        self.best_person = self.persons[0]

    def predict(self, item, **kwargs):
        """ Predicts weighted responses to a given problem.
            The prediction function takes the currently best performing person of the GA to predict the answer for each
            task.
            For the first task the best person of the GA is used.
        """
        self.current_person_nr += 1
        answer = ""
        if self.current_person_nr == 1:
            if self.exp_nr == 1:
                answer = self.best_person.answer_question_exp1(self.item_to_question(item))
            if self.exp_nr == 2:
                answer = self.best_person.answer_question_exp2(self.item_to_question(item))
            if self.exp_nr == 3:
                answer = self.best_person.answer_question_exp3_2(self.item_to_question(item))
            return answer
        if self.exp_nr == 1:
            answer = self.persons[0].answer_question_exp1(self.item_to_question(item))
        if self.exp_nr == 2:
            answer = self.persons[0].answer_question_exp2(self.item_to_question(item))
        if self.exp_nr == 3:
            answer = self.persons[0].answer_question_exp3_2(self.item_to_question(item))
        return answer

    def adapt(self, item, target, **kwargs):
        """ Trains the model based on a given problem-target combination.
            The adapt function adapts the model to the current participant by increasing the score of each person
            if it predicts right and decreasing if it predicts wrong.
        """
        for each in self.persons:
            answer = ""
            if self.exp_nr == 1:
                answer = each.answer_question_exp1(self.item_to_question(item, True))
                if answer[0][0] == target[0][0]:
                    each.person_score += 1
                else:
                    each.person_score -= 1
            elif self.exp_nr == 2:
                answer = each.answer_question_exp2(self.item_to_question(item, True))
            elif self.exp_nr == 3:
                answer = each.answer_question_exp3_2(self.item_to_question(item, True))
            if self.exp_nr != 1:
                if answer == target:
                    each.person_score += 1
                else:
                    each.person_score -= 1
        self.persons.sort(key=lambda x: x.person_score, reverse=True)

    def make_ga(self, dataset):
        """
        initializes a set of GARule objects, by extracting all different persons out of the dataset
        (most likely the train set)
        :param dataset:
        :return:
        """
        population = []
        answers = []
        questions = []
        tasks = []
        first = True
        for person in dataset:
            # get the given answers
            for task in person:
                if 'Task-ID' in task:
                    quest_id = task['Task-ID']
                else:
                    quest_id = task['TaskID']
                ans = task['response']
                person_id = person[0]['item'].identifier
                answers.append([quest_id, ans, person_id])
                # create the questions
                quest = self.item_to_question(task['item'])
                quest.question_id = quest_id
                questions.append(quest)
                tasks.append([ans, quest])
            if first:
                self.nr_of_questions = len(tasks)
                print(len(tasks))
                first = False
        # make subjects
        rules = self.get_rules()
        for subject in range(0, self.test_subject_number):
            rules = deepcopy(self.get_rules())
            # make rules
            subject_rules = []
            for rule_pos in range(0, self.start_rule_number):
                subject_rules.append(Rule(rules[random.randrange(0, len(rules))].coding, False))
            population.append(RuleTestPerson(subject_rules, 0, answers))

        return GARule(population, tasks, self.get_rules)

    def get_rules(self):
        raise NotImplementedError

    def item_to_question(self, item, renamed=False):
        """
        :param item:
        :param renamed:
        :return:
        """
        raise NotImplementedError

    def rename_item(self, item):
        """
        renames an item to be able to be handled by the implementation of my rule approach.
        :param item: item of the ccobra framwork
        :return: an item with all words renamed.
        """
        raise NotImplementedError

    def save_current_state(self, runs, genetic_algorithm):
        if not self.save:
            return
        name = self.name + "runs" + str(runs) + ".p"
        name = name.replace(" ", "")
        name = name.replace(":", "")
        persons = genetic_algorithm.test_persons
        with open(name, 'wb') as f:
            pickle.dump(persons, f)
        print("pickled")
