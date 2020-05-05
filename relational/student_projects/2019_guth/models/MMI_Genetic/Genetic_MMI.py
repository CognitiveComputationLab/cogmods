import time
import ccobra
from GAModel import GAModel
from Model import Model
from ModelTestPerson import ModelTestPerson
import pickle


class ModelApproach(ccobra.CCobraModel):
    def __init__(self, exp_nr, name):
        """ Model constructor.
            This class represents the rule approach of my thesis.
            All benchmarks can be started with this class.
        """
        super(ModelApproach, self).__init__(name, ['spatial-relational'], ['verify', 'single-choice'])

         # set this to False if you want to retrain
        self.load = True

        # a list of persons which will finally be used to answer the questions in the predict function.
        self.persons = []
        # the number represents the experiment/benchmark. 1 = singelchoice; 2 = verification; 3 = figural;
        # 4 = premiseorder
        self.exp_nr = exp_nr
        # the size of the population of each genetic algorithm/ for each person in the dataset.
        self.test_subject_number = 100
        # number of runs the genetic algorithm will perform.
        self.GA_runs = 200
        # name of file from which to load an already trained model
        self.load_file = name + ".p"
        if not self.load:
            # the name will be visible in the website for the data analyse.
            self.name += " Pop: " + str(self.test_subject_number) + " Runs: " + str(self.GA_runs)
        # the number of questions of the experiment.
        self.nr_of_questions = 0
        self.nr_of_participants = 0
        # this is a counter to count the number of answers already given.
        self.current_person_nr = 0
        # some string operations to be able to save intermediate data
        self.save_file = self.name + ".txt"
        self.save_file = self.save_file.replace(" ", "")
        self.save_file = self.save_file.replace(":", "")
        self.best_model = None
        # set this to True if you want to safe the trained model in a pickle.
        self.save = False
        if self.save:
            with open(self.save_file, "w") as f:
                print(self.name, file=f)
        # set this to False if you want to retrain
        self.load = True


    def pre_train(self, dataset):
        """ Pre-trains the model based on one or more datasets.
        """
        # if you want to load older data use this
        if self.load:
            pickle_obj = pickle.load(open("Trained_Models/" + self.load_file, "rb"))
            self.persons = pickle_obj
            self.best_model = self.persons[0]
            return

        genetic_algorithm = self.initiate_genetic_algorithm(dataset)
        # perform the runs in the genetic algorithm
        self.genetic_algorithm_run(genetic_algorithm)
        # evaluate one last time.
        genetic_algorithm.evaluate(self.exp_nr)
        # save the last run.
        self.save_current_state(5555, genetic_algorithm)
        self.persons = genetic_algorithm.test_persons
        self.sort_by_score()
        self.best_model = self.persons[0]
        for each in self.persons:
            print("Person", each.person_id, "Models:", len(each.models), "Score:", each.person_score, "/",
                  self.nr_of_questions)
        print("finished with the GAModel, now answering questions. THIS MAY TAKE A WHILE!")

    def start_participant(self, **kwargs):
        """ Model initialization method. Used to setup the initial state of
        its datastructures, memory, etc.
        initializes each participant.
        """
        # reset the person score of each person
        self.current_person_nr = 0
        for each in self.persons:
            each.person_score = 0

    def predict(self, item, **kwargs):
        """ Predicts weighted responses to a given problem.
        The prediction function takes the currently best performing person of the GA to predict the answer for each task.
        For the first task the best person of the GA is used.
        """
        self.current_person_nr += 1
        answer = ""
        if self.current_person_nr == 1:
            if self.exp_nr == 1:
                answer = self.best_model.answer_question_exp1(item)
            if self.exp_nr == 2:
                answer = self.best_model.answer_question_exp2(item)
            if self.exp_nr == 3:
                answer = self.best_model.answer_question_exp3(item)
            return answer
        if self.exp_nr == 1:
            answer = self.persons[0].answer_question_exp1(item)
        if self.exp_nr == 2:
            answer = self.persons[0].answer_question_exp2(item)
        if self.exp_nr == 3:
            answer = self.persons[0].answer_question_exp3(item)
        return answer

    def adapt(self, item, target, **kwargs):
        """
        The adapt function adapts the model to the current participant by increasing the score of each person
            if it predicts right and decreasing if it predicts wrong.
        """
        for each in self.persons:
            answer = ""
            if self.exp_nr == 1:
                answer = each.answer_question_exp1(item)
            if self.exp_nr == 2:
                answer = each.answer_question_exp2(item)
            if self.exp_nr == 3:
                answer = each.answer_question_exp3(item)
            if answer == target:
                each.person_score += 1
            else:
                each.person_score -= 1
        self.sort_by_score()
        if self.save:
            if self.current_person_nr == self.nr_of_questions:
                with open(self.save_file, "a") as f:
                    print("new Person", file=f)
                    for each1 in self.persons[0].models:
                        print(each1.code_to_human_read(each1.coding), file=f)
                    print("score :",self.persons[0].person_score, file=f)

    def initiate_genetic_algorithm(self, dataset):
        """
        initializes the genetic algorithm with the given data
        :param dataset:
        :return:
        """
        answers = []
        questions = []
        tasks = []
        first = True
        for person in dataset:
            self.nr_of_participants += 1
            # get the given answers in a list of [question_id, given_answer]
            for task in person:
                if 'Task-ID' in task:
                    quest_id = task['Task-ID']
                else:
                    quest_id = task['TaskID']
                answer = task['response']
                answers.append([quest_id, answer])
                # create the questions
                item = task['item']
                if 'Task-ID' in task:
                    quest_id = task['Task-ID']
                else:
                    quest_id = task['TaskID']
                questions.append([quest_id, item])
                tasks.append([quest_id, item, answer])
            if first:
                self.nr_of_questions = len(tasks)
                #print(len(tasks))
                first = False
        population = []
        # create test_persons
        for i in range(0, self.test_subject_number):
            model = Model("")
            model.coding = model.random_code_generator()
            test_person = ModelTestPerson(0, [model], answers)
            population.append(test_person)
        return GAModel(population, tasks)

    def sort_by_score(self):
        self.persons.sort(key=lambda x: x.person_score, reverse=True)

    def genetic_algorithm_run(self, genetic_algorithm):
        """
        performs all steps needed for the genetic algorithm.
        :param genetic_algorithm:
        :return:
        """
        # time management stuff so one can see how long each step took and get an approximation of the remaining time.
        last_time = time.time()
        last_percent = 0.00001
        cnt = 0
        for i in range(0, self.GA_runs):

            # time stuff and print for the percentage of
            time_now = time.time()
            time_took = (time_now - last_time)
            percent_step = ((i / self.GA_runs) * 100) - last_percent
            estimated_time = (time_took * (100 - (i / self.GA_runs) * 100) / percent_step)
            print((i / self.GA_runs) * 100, "% ", "Time used: ", time_took / 60, " Estimated time left: ",
                  estimated_time / 60)
            last_time = time.time()
            last_percent = (i / self.GA_runs) * 100
            cnt += 1
            genetic_algorithm.evaluate(self.exp_nr)
            if i % 10 == 0:
                self.save_current_state(i, genetic_algorithm)
            genetic_algorithm.selection()
            genetic_algorithm.mutation()

    def save_current_state(self, runs, genetic_algorithm):
        """
        saves all current persons of the genetic algorithm.
        """
        if self.save:
            name = self.name + "runs" + str(runs) + ".p"
            name = name.replace(" ", "")
            name = name.replace(":", "")
            persons = genetic_algorithm.test_persons
            with open(name, 'wb') as f:
                pickle.dump(persons, f)
            print("saved")

