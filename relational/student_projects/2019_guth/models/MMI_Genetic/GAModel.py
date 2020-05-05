import random
from ModelTestPerson import ModelTestPerson
from Model import Model


class GAModel:
    """
        represents the genetic algorithm for the model approach.
    """
    def __init__(self, test_persons, task):
        self.test_persons = test_persons
        self.tasks = task
        self.population_split = 2

    def evaluate(self, exp_nr):
        """
        represents the evaluation part of the genetic algorithm.
        :param exp_nr: number of the experiment to know which answer should be taken
        :return:
        """
        for person in self.test_persons:
            person.person_score = 0
            for task in self.tasks:
                person_answer = None
                given_answer = task[2]
                if exp_nr == 1:
                    person_answer = person.answer_question_exp1(task[1])
                if exp_nr == 2:
                    person_answer = person.answer_question_exp2(task[1])
                if exp_nr == 3:
                    person_answer = person.answer_question_exp3(task[1])
                # with open("should_be.txt", "a") as f:
                #   print("Predicted:", person_answer, "should be:", given_answer, file=f)
                if person_answer == given_answer:
                    person.person_score += 1
        self.test_persons.sort(key=lambda x: x.person_score, reverse=True)
        print(self.test_persons[0].person_score)

    def selection(self):
        """
            represents the selection and crossover part of the genetic algorithm.
        :return:
        """
        nr_of_persons = len(self.test_persons)
        # drop the bottom half of the population since thy perform bad.
        self.test_persons = self.test_persons[:int(len(self.test_persons) / self.population_split)]
        # refill the population with new persons.
        while len(self.test_persons) < nr_of_persons:
            # select two persons for crossover
            pers1 = self.roulette_random_selection()
            pers2 = self.roulette_random_selection()
            # make new models for the new person
            new_models = []
            for i in range(0, len(self.test_persons[pers1].models)):
                if len(self.test_persons[pers2].models) <= i:
                    new_models.append(Model(self.test_persons[pers1].models[i].coding))
                else:
                    random_parter = random.randint(0, 63)
                    new_coding = self.test_persons[pers1].models[i].coding[:random_parter]
                    new_coding += self.test_persons[pers2].models[i].coding[random_parter:]
                    model = Model(new_coding)
                    new_models.append(model)
            new_person = ModelTestPerson(self.test_persons[0].person_id, new_models, self.test_persons[0].given_answers)
            # append the new person to the person set.
            self.test_persons.append(new_person)

    def mutation(self):
        """
        Represents the mutation process of the genetic algorithm.
        :return:
        """
        for test_person in self.test_persons:
            for model in test_person.models:
                # 50% chance to mutate the model
                if random.randint(0, 1000) < 500:
                    model.mutate_coding()
            # 10% chance to gain a model
            if random.randint(0, 1000) < 100:
                m = Model("")
                m.coding = m.random_code_generator()
                test_person.models.append(m)
            # 10% chance to loose a model
            if random.randint(0, 1000) < 100 and len(test_person.models) > 1:
                random_model = random.randint(0, len(test_person.models)-1)
                test_person.models.remove(test_person.models[random_model])

    def roulette_random_selection(self):
        """
        implementation of a random selection that selects a person based on its score.
        :return: the index of the selected person in the person list.
        """
        n = len(self.test_persons) - 1
        weight = []
        for pers in self.test_persons:
            weight.append(pers.person_score)
        max_weight = self.test_persons[0].person_score
        if max_weight == 0:
            return random.randint(0, len(self.test_persons)-1)
        while True:
            index = int(random.random() * n)
            if random.random() < weight[index] / max_weight:
                return index
