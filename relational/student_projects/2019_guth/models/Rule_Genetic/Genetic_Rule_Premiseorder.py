import Genetic_Rule
from Node import Node
from Question import Question
from RuleCreator import RuleCreator
from copy import deepcopy


class RuleApproachExp32(Genetic_Rule.RuleApproach):
    def __init__(self):
        Genetic_Rule.RuleApproach.__init__(self, 3, "rule_premiseorder")
        rule_creator = RuleCreator()
        self.rules = rule_creator.create_three_step_left()
        self.rules += rule_creator.create_three_step_right()
        self.rules += rule_creator.create_one_sided_rules_left()
        self.rules += rule_creator.create_basic_rules_left_right()
        self.rules += rule_creator.create_one_sided_rules_right()
        self.rules += rule_creator.create_three_step_left()

    def item_to_question(self, item, renamed=False):
        """
        Transforms a ccobra item to a question class.
        :param item:
        :param renamed:
        :return:
        """
        if not renamed:
            self.rename_item(item)
        # a list of all premises for this question
        premises = []
        for premise in item.task:
            parent = Node(premise[0])
            lchild = Node(premise[1], None, None, parent)
            rchild = Node(premise[2], None, None, parent)
            parent.left = lchild
            parent.right = rchild
            parent.coding = parent.visit_make_coding()
            premises.append(parent)
        # a string of the question arrangement for experiment 2
        quest = ""
        if item.choices[0][0][0] == "rights":
            quest = item.choices[0][0][2]
            quest += item.choices[0][0][1]
        if item.choices[0][0][0] == "lefts":
            quest = item.choices[0][0][1]
            quest += item.choices[0][0][2]
        question = Question(premises, quest, 2, 0)
        return question

    def rename_item(self, item):
        """
            renames an item to be able to be handled by the implementation of my rule approach.
        :param item: item of the ccobra framwork
        :return: an item with all words renamed.
        """
        renamed = {}
        objects = ["A", "B", "C", "D", "E"]
        cnt = 0
        for each in item.task:
                # rename words in the tasks
                if each[0] == "Left":
                    each[0] = "lefts"
                elif each[0] == "Right":
                    each[0] = "rights"
                else:
                    print("ERROR: no task translation found.")
                if each[1] in renamed:
                    each[1] = renamed[each[1]]
                else:
                    renamed[each[1]] = objects[cnt]
                    each[1] = renamed[each[1]]
                    cnt += 1
                if each[2] in renamed:
                    each[2] = renamed[each[2]]
                else:
                    renamed[each[2]] = objects[cnt]
                    each[2] = renamed[each[2]]
                    cnt += 1
        for each in item.choices[0]:
            if each[0] == "Left":
                each[0] = "lefts"
            elif each[0] == "Right":
                each[0] = "rights"
            else:
                print("ERROR: no task translation found.")
            if each[1] in renamed:
                each[1] = renamed[each[1]]
            if each[2] in renamed:
                each[2] = renamed[each[2]]

    def get_rules(self):
        return self.rules
