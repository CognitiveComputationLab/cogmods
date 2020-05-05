import Genetic_Rule
from Node import Node
from Question import Question
from RuleCreator import RuleCreator


class RuleApproachExp1(Genetic_Rule.RuleApproach):
    def __init__(self):
        Genetic_Rule.RuleApproach.__init__(self, 1, "rule_singlechoice")

    def item_to_question(self, item, renamed=False):
        """
        Transforms a ccobra item to a question class.
        :param item:
        :param renamed:
        :return:
        """
        # get the original question before the item is renamed.
        org_quest = [item.choices[1][0][1], item.choices[1][0][2]]
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
        quest = [item.choices[0][0][1], item.choices[0][0][2]]
        question = Question(premises, quest, 2, 0, org_quest)
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
        # check for tasks with the combination of north-west usw... ans split them up in two tasks
        new_tasks = []
        for task in item.task:
            if len(task[0]) <= 5:
                new_tasks.append(task)
            else:
                new_tasks.append([task[0][:5], task[1], task[2]])
                new_tasks.append([task[0][6:], task[1], task[2]])
        item.task = new_tasks

        for each in item.task:
                # rename words in the tasks
                if each[0] == "west":
                    each[0] = "lefts"
                elif each[0] == "east":
                    each[0] = "rights"
                elif each[0] == "north":
                    each[0] = "fronts"
                elif each[0] == "south":
                    each[0] = "backs"
                else:
                    print("ERROR: no task translation found.")
                    print(item)
                    raise Exception
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
            if each[1] in renamed:
                each[1] = renamed[each[1]]
            if each[2] in renamed:
                each[2] = renamed[each[2]]

    def get_rules(self):
        """
        :return: rules that can be used for this experiment.
        """
        rule_creator = RuleCreator()
        rules = rule_creator.create_one_sided_rules_left()
        for each in rule_creator.create_basic_rules_left_right():
            rules.append(each)
        for each in rule_creator.create_one_sided_rules_right():
            rules.append(each)
        for each in rule_creator.create_basic_rules_front_back():
            rules.append(each)
        for each in rule_creator.create_one_sided_rules_front():
            rules.append(each)
        for each in rule_creator.create_one_sided_rules_back():
            rules.append(each)
        return rules
