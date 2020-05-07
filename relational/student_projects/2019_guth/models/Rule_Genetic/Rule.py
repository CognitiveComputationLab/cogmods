from Node import Node


class Rule:
    """
     This class represents a Rule for the rule approach.
     It has its Coding which is a string of its binary coding, a node named rule which is the starting node of the rule
     in its tree representation, a string representation of the polish notation of the Rule and an easily human readable
     representation of the rule.
    """
    human_read = ""
    polish_notation = ""
    coding = ""
    rule = None
    needed_premises = []
    conclusions = []
    used_counter = 0

    def __init__(self, rule, asnode=True):
        """
        :param rule: the rule
        :param asnode: change it to false if you are only passing a lib which should be transformed to a rule.

        This constructor takes either a Node which represents the starting node of a rule and fills in all other needed
        information. Or an coding which represents a rule in its binary coding, if you use a Binary coding you need to
        set asnode = False.
        """

        if not asnode:
            # if the rule got passed as an coding:
            self.translate_coding_to_rule(rule)
        else:
            # if the rule got passed as a node.
            self.rule = rule
            self.human_read = self.rule.visit_easy_read()
            self.polish_notation = self.rule.visit_with_polish_notation()
            self.coding = self.rule.visit_make_coding()
            self.find_needed_premises()
            self.find_conclusions()

    def translate_coding_to_rule(self, rule):
        """
        this transforms the coding of a rule to its tree representation.
        :param rule: string representation of the binary coding of the rule.
        :return: nothing
        """
        node = Node("", None, None, None)
        node.code_to_rule(rule, None)
        self.rule = node
        self.human_read = self.rule.visit_easy_read()
        self.polish_notation = self.rule.visit_with_polish_notation()
        self.coding = self.rule.visit_make_coding()
        self.find_needed_premises()
        self.find_conclusions()

    def find_needed_premises(self):
        """
            finds the premises needed for this rule to work.
        :return:
        """
        premises = []
        self.rule.left.visit_find_premises(premises)
        self.needed_premises = premises

    def find_conclusions(self):
        """
            finds the conclusions that can be taken form this rule
        :return:
        """
        conc = []
        self.rule.right.visit_find_premises(conc)
        self.conclusions = conc

    def check_with_premises(self, premises):
        """
            compares the given premises with the premises needed for the rule to work.
        :param premises:
        :return: true if the rule works with the given premises, false else.
        """
        cnt = 0
        for self_prem in self.needed_premises:
            self_prem_code = self_prem.visit_make_coding()
            for given_prem in premises:
                if self_prem_code == given_prem.coding:
                    cnt += 1
        if cnt == len(self.needed_premises):
            return True
        return False
