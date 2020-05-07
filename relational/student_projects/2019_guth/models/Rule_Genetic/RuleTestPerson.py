from copy import deepcopy

class RuleTestPerson:
    """ This class represents one test person of the experiment.
        It is in possession of a set of rules to answer the questions.
    """
    rule_set = []
    person_id = 0
    given_answers = []
    person_score = 0

    def __init__(self, rules, person_id, given_answers):
        self.rule_set = deepcopy(rules)
        self.person_id = person_id
        self.given_answers = given_answers

    def answer_question_exp1(self, question):
        """
        answeres questions of the singelchoice benchmark.
        :param question: the question form the ccobra item
        :return: the answer to the question
        """
        # find the rules which work with the premises
        working_rules = []
        for rule in self.rule_set:
            if rule.check_with_premises(question.premises):
                working_rules.append(rule)
                rule.used_counter += 1
                # print(rule.human_read)
        conclusions = []
        # print("found rules that work: ", len(working_rules))
        for conc in working_rules:
            for conclusion in conc.conclusions:
                conclusions.append(conclusion)
        direction = ""
        # pass the conclusions and make a final conclusion about the relation.
        for conclusion in conclusions:
            if question.question[0] == conclusion.left.symbol and question.question[1] == conclusion.right.symbol:
                renamed_direction = self.rename_direction(conclusion.symbol, False)
                if len(direction) == 0:
                    direction = renamed_direction
                if len(direction) == 4 and len(renamed_direction) == 5:
                    direction = renamed_direction + "-" + direction
                if len(direction) == 5 and len(renamed_direction) == 4:
                    direction = direction + "-" + renamed_direction

            if question.question[1] == conclusion.left.symbol and question.question[0] == conclusion.right.symbol:
                renamed_direction = self.rename_direction(conclusion.symbol, True)
                if len(direction) == 0:
                    direction = renamed_direction
                if len(direction) == 4 and len(renamed_direction) == 5:
                    direction = renamed_direction + "-" + direction
                if len(direction) == 5 and len(renamed_direction) == 4:
                    direction = direction + "-" + renamed_direction
        answer = [[direction, question.org_question[0], question.org_question[1]]]
        return answer

    def rename_direction(self, direction, reverted):
        """
            renames the directions of a question if east/west/north/south appear.
        :param direction: the given direction which is to be renamed
        :param reverted: if the renaming should be the opposite
        :return:
        """
        if "left" in direction:
            if reverted:
                return "east"
            return "west"
        if "right" in direction:
            if reverted:
                    return "west"
            return "east"
        if "front" in direction:
            if reverted:
                return "south"
            return "north"
        if "back" in direction:
            if reverted:
                return "north"
            return "south"

    def answer_question_exp2(self, question):
        """
        answers questions for the verification and figural benchmark.
        :param question: asked question
        :return:
        """
        # find rules that work with the premises
        working_rules = []
        for rule in self.rule_set:
            if rule.check_with_premises(question.premises):
                working_rules.append(rule)
                rule.used_counter += 1
        conclusions = []
        for conc in working_rules:
            for conclusion in conc.conclusions:
                conclusions.append(conclusion)
        # print("found rules that work: ", len(working_rules))
        # check if the conclusions disagree with the alignment.
        for conclusion in conclusions:
            if self.check_pos(conclusion.left.symbol, conclusion.right.symbol, question.question, conclusion.symbol):
                pass
            else:
                return False
        if len(conclusions) > 0:
            return True
        return False

    def answer_question_exp3_2(self, question):
        """
        answers questions for the premiseorder benchmark.
        :param question:
        :return:
        """
        # find working rules.
        working_rules = []
        for rule in self.rule_set:
            if rule.check_with_premises(question.premises):
                working_rules.append(rule)
                rule.used_counter += 1
        # check if the conclusions disagree with the given relation.
        for conc in working_rules:
            for conclusion in conc.conclusions:
                if self.check_pos(conclusion.left.symbol, conclusion.right.symbol, question.question, conclusion.symbol):
                    pass
                else:
                    return False
        if len(working_rules) > 0:
            return True
        return False

    def check_pos(self, obj1, obj2, question, direction):
        """
            checks whether or not the two given objects position is the same as the given direction.
        :param obj1: first object
        :param obj2: second object
        :param question: the asked question
        :param direction: direction for the objects
        :return: true if the position of the objects is ok. false if not.
        """
        pos1 = question.find(obj1)
        pos2 = question.find(obj2)
        if direction == "lefts" and pos1 > pos2:
            return True
        if direction == "left1" and pos1 == pos2 - 1:
            return True
        if direction == "left2" and pos1 == pos2 - 2:
            return True
        if direction == "left3" and pos1 == pos2 - 3:
            return True

        if direction == "rights" and pos1 > pos2:
            return True
        if direction == "right1" and pos1 == pos2 - 1:
            return True
        if direction == "right2" and pos1 == pos2 - 2:
            return True
        if direction == "right3" and pos1 == pos2 - 3:
            return True
        return False

    def __sortkey__(self):
        return self.person_score

    def sort_rules(self):
        self.rule_set.sort(key=lambda x: x.used_counter, reverse=True)

    def __str__(self):
        string = ""
        string += "ID: " + str(self.person_id)
        string += " Rules: "
        for each in self.rule_set:
            string += each.human_read
        return string

