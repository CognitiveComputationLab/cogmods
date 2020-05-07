import random
import Rule
import RuleTestPerson
from copy import deepcopy


class GARule:
    """
    represents the genetic algorithm for the rule approach.
    """
    test_persons = []
    tasks = []

    def __init__(self, persons, tasks, rules, min_rules=10):
        self.test_persons = persons
        self.tasks = tasks
        self.rules = rules()
        self.rule_fct = rules
        # 100 for 10% 200 for 20% etc...
        self.mutation_probability = 100
        self.min_rules = min_rules

    def evaluate(self, exp_nr):
        """
        represents the evaluation part of the genetic algorithm.
        :param exp_nr: number of the experiment to know which answer should be taken
        :return:
        """
        # for each person evaluate how well it predicts all answers.
        for person in self.test_persons:
            person.person_score = 0
            # reset used counter for rules.
            for rule in person.rule_set:
                rule.used_counter = 0
            for task in self.tasks:
                given_answer = task[0]
                person_answer = None
                if exp_nr == 2:
                    person_answer = person.answer_question_exp2(task[1])
                if exp_nr == 1:
                    person_answer = person.answer_question_exp1(task[1])
                if exp_nr == 3:
                    person_answer = person.answer_question_exp3_2(task[1])
                # print("given:", person_answer, "expected:", given_answer)
                if person_answer == given_answer:
                    person.person_score += 1
        self.test_persons.sort(key=lambda x: x.person_score, reverse=True)
        print("best person: ", self.test_persons[0].person_score)

    def selection(self):
        """
        represents the selection and crossover part of the genetic algorithm.
        :return:
        """
        nr_of_persons = int(len(self.test_persons))
        # drop the bottom half of the population since thy perform bad.
        self.test_persons = self.test_persons[:int(nr_of_persons/2)]
        # refill the population with new persons.
        while len(self.test_persons) < nr_of_persons:
            # select two persons for crossover
            pers1 = self.roulette_random_selection()
            pers2 = self.roulette_random_selection()
            if pers1 == pers2:
                pers2 += 1
                if pers2 >= len(self.test_persons):
                    pers2 = 0
            new_rules = []
            self.test_persons[pers1].sort_rules()
            self.test_persons[pers2].sort_rules()
            # make a new rule set for the new person by inheriting the used rules from both persons.
            for i in range(0, int(len(self.test_persons[pers1].rule_set))):
                if self.test_persons[pers1].rule_set[i] not in new_rules and \
                        self.test_persons[pers1].rule_set[i].used_counter > 0 and i < 15:
                    new_rules.append(self.test_persons[pers1].rule_set[i])
            for i in range(0, int(len(self.test_persons[pers2].rule_set))):
                if self.test_persons[pers2].rule_set[i] not in new_rules and \
                        self.test_persons[pers2].rule_set[i].used_counter > 0 and i < 15:
                    new_rules.append(self.test_persons[pers2].rule_set[i])
            # if there were not enouth rules fill the rest of the new persons rule set with random rules.
            while len(new_rules) < self.min_rules:
                new_rules.append(self.get_random_rule())
            # remove equal rules form the set.
            for rule in new_rules:
                for next_rule in new_rules:
                    if rule.coding == next_rule.coding and rule != next_rule:
                        new_rules.remove(next_rule)
            new_person = RuleTestPerson(new_rules, self.test_persons[0].person_id, self.test_persons[0].given_answers)
            # append the new person to the person set.
            self.test_persons.append(new_person)

    def mutation(self):
        """
        Represents the mutation process of the genetic algorithm.
        :return:
        """
        first = True
        for person in self.test_persons:
            # do not change the best person.
            if first:
                first = False
                continue
            new_rules = []
            for rule in person.rule_set:
                # 5% chance to loose a rule
                if random.randrange(0, 1000) < 950:
                    new_rules.append(self.mutate_rule(rule))
                # 5% chance to get a new rule
                if random.randrange(0, 1000) > 950:
                    new_rules.append(self.rule_fct()[random.randrange(0, len(self.rules))])
            person.rule_set = new_rules

    def mutate_rule(self, rule):
        """
        mutates a rule
        :param rule:
        :return: a rule
        """
        # chance to not mutate the rule
        if random.randrange(0, 1000) > self.mutation_probability:
            return rule
        binary_code = rule.coding
        mutated_code = self.mutate_code(binary_code)
        new_rule = Rule(mutated_code, False)
        return new_rule

    def swap_bit(self, bit):
        if bit == "1":
            return "0"
        return "1"

    def get_random_rule(self):
        rand = random.randrange(0, len(self.rules)-1)
        return Rule(self.rule_fct()[rand].coding, False)

    def mutate_code(self, code):
        """
        mutates the lib for a rule.
        :param code:
        :return:
        """
        # case 00**** direction
        if code[0] == "0" and code[1] == "0":
            # make a higher mutation if it is on the last direction.
            if len(code) <= 16:
                if random.randrange(0, 1000) < 50:
                    code = code[:2] + self.swap_bit(code[2]) + code[3:]
                if random.randrange(0, 1000) < 100:
                    code = code[:3] + self.swap_bit(code[3]) + code[4:]
                if random.randrange(0, 1000) < 200:
                    code = code[:4] + self.swap_bit(code[4]) + code[5:]
                if random.randrange(0, 1000) < 300:
                    code = code[:5] + self.swap_bit(code[5]) + code[6:]
            # if in the premises part of the rule only mutate the direction not the position
            else:
                if random.randrange(0, 1000) < 50:
                    code = code[:2] + self.swap_bit(code[2]) + code[3:]
                if random.randrange(0, 1000) < 100:
                    code = code[:3] + self.swap_bit(code[3]) + code[4:]

            if len(code) > 6:
                return code[0:6] + self.mutate_code(code[6:])
            else:
                return code

        # case 10** connection
        elif code[0] == "1" and code[1] == "0":
            if len(code) > 4:
                return code[:4] + self.mutate_code(code[4:])
            else:
                return code

        # case 11*** object
        elif code[0] == "1" and code[1] == "1":
            if len(code) > 5:
                return code[:5] + self.mutate_code(code[5:])
            else:
                return code
        print("FEHLER IN DER MUTATION", code)

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
            return random.randint(0, len(self.test_persons) - 1)
        while True:
            index = int(random.random() * n)
            if random.random() < weight[index] / max_weight:
                return index

    def make_deep_copy(self):
        for person in self.test_persons:
            for rule in person.rule_set:
                rule = deepcopy(rule)

    def __str__(self):
        string = ""
        for each in self.tasks[1]:
            string += str(each)
        return string
