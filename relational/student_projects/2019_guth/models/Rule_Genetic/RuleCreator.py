from Rule import Rule
from Node import Node


class RuleCreator:
    """
        a class to crate all needed rules.
    """
    left = ["left1", "left2", "left3", "lefts"]
    left_code = ["000100", "000101", "000110", "000111"]
    right = ["right1", "right2", "right3", "rights"]
    right_code = ["000000", "000001", "000010", "000011"]
    front = ["front1", "front2", "front3", "fronts"]
    front_code = ["001000", "001001", "001010", "001011"]
    back = ["back1", "back2", "back3", "backs"]
    back_code = ["001100", "001101", "001110", "001111"]
    implications = ["->", "<->"]
    impl_code = ["1010", "1001"]
    connections = ["AND", "OR"]
    con_code = ["1000", "1001"]
    objects = ["A", "B", "C", "D", "E"]
    obj_code = ["11000", "11001", "11010", "11011", "11100"]
    rules = []

    def create_basic_rules_left_right(self):
        """
        creates rules like: a L b <-> b R a
        :return:
        """
        this_rules = []
        for direction in range(0, 4):
            for each1 in self.objects:
                for each2 in self.objects:
                    if each1 != each2:
                        # start making left/right rules
                        parent = Node("<->")
                        lchild = Node(self.left[direction], None, None, parent)
                        rchild = Node(self.right[direction], None, None, parent)
                        rrchild = Node(each1, None, None, rchild)
                        rlchild = Node(each2, None, None, rchild)
                        rchild.right = rrchild
                        rchild.left = rlchild
                        lrchild = Node(each2, None, None, lchild)
                        llchild = Node(each1, None, None, lchild)
                        lchild.left = llchild
                        lchild.right = lrchild
                        parent.left = lchild
                        parent.right = rchild
                        rule = Rule(parent)
                        this_rules.append(rule)
        return this_rules

    def create_basic_rules_front_back(self):
        """
        creates rules like: a F b <-> b B a
        :return:
        """
        this_rules = []
        for direction in range(0, 4):
            for each1 in self.objects:
                for each2 in self.objects:
                    if each1 != each2:
                        # start making front/back rules
                        parent = Node("<->")
                        lchild = Node(self.front[direction], None, None, parent)
                        rchild = Node(self.back[direction], None, None, parent)
                        rrchild = Node(each1, None, None, rchild)
                        rlchild = Node(each2, None, None, rchild)
                        rchild.right = rrchild
                        rchild.left = rlchild
                        lrchild = Node(each2, None, None, lchild)
                        llchild = Node(each1, None, None, lchild)
                        lchild.left = llchild
                        lchild.right = lrchild
                        parent.left = lchild
                        parent.right = rchild
                        rule = Rule(parent)
                        this_rules.append(rule)
        return this_rules

    def create_one_sided_rules_left(self):
        """
        creates rules like: a L b AND b L c -> a L c
        :return:
        """
        this_rules = []
        for direction in self.left_code:
            for obj1 in self.obj_code:
                for obj2 in self.obj_code:
                    for obj3 in self.obj_code:
                        if obj1 != obj2 and obj2 != obj3 and obj1 != obj3:
                            rule_str = ""
                            rule_str += self.impl_code[0]  # ->
                            rule_str += self.con_code[0]        # AND
                            rule_str += self.left_code[3]  # lefts
                            rule_str += obj1
                            rule_str += obj2
                            rule_str += self.left_code[3]  # lefts
                            rule_str += obj2
                            rule_str += obj3
                            rule_str += direction
                            rule_str += obj1
                            rule_str += obj3
                            this_rules.append(Rule(rule_str, False))
        return this_rules

    def create_one_sided_rules_right(self):
        """
        creates rules like: a R b AND b R c -> a R c
        :return:
        """
        this_rules = []
        for direction in self.right_code:
            for obj1 in self.obj_code:
                for obj2 in self.obj_code:
                    for obj3 in self.obj_code:
                        if obj1 != obj2 and obj2 != obj3 and obj1 != obj3:
                            rule_str = ""
                            rule_str += self.impl_code[0]  # ->
                            rule_str += self.con_code[0]        # AND
                            rule_str += self.left_code[3]  # rights
                            rule_str += obj1
                            rule_str += obj2
                            rule_str += self.left_code[3]  # rights
                            rule_str += obj2
                            rule_str += obj3
                            rule_str += direction
                            rule_str += obj1
                            rule_str += obj3
                            this_rules.append(Rule(rule_str, False))
        return this_rules

    def create_one_sided_rules_front(self):
        """
        creates rules like: a F b AND b F c -> a F c
        :return:
        """
        this_rules = []
        for direction in self.front_code:
            for obj1 in self.obj_code:
                for obj2 in self.obj_code:
                    for obj3 in self.obj_code:
                        if obj1 != obj2 and obj2 != obj3 and obj1 != obj3:
                            rule_str = ""
                            rule_str += self.impl_code[0]  # ->
                            rule_str += self.con_code[0]        # AND
                            rule_str += self.front_code[3]  # fronts
                            rule_str += obj1
                            rule_str += obj2
                            rule_str += self.front_code[3]  # fronts
                            rule_str += obj2
                            rule_str += obj3
                            rule_str += direction
                            rule_str += obj1
                            rule_str += obj3
                            this_rules.append(Rule(rule_str, False))
        return this_rules

    def create_one_sided_rules_back(self):
        """
        creates rules like: a B b AND b B c -> a B c
        :return:
        """
        this_rules = []
        for direction in self.back_code:
            for obj1 in self.obj_code:
                for obj2 in self.obj_code:
                    for obj3 in self.obj_code:
                        if obj1 != obj2 and obj2 != obj3 and obj1 != obj3:
                            rule_str = ""
                            rule_str += self.impl_code[0]  # ->
                            rule_str += self.con_code[0]        # AND
                            rule_str += self.back_code[3]  # backs
                            rule_str += obj1
                            rule_str += obj2
                            rule_str += self.back_code[3]  # backs
                            rule_str += obj2
                            rule_str += obj3
                            rule_str += direction
                            rule_str += obj1
                            rule_str += obj3
                            this_rules.append(Rule(rule_str, False))
        return this_rules

    def create_three_step_right_org(self):

        this_rules = []
        for direction in self.right_code:
            for obj1 in self.obj_code:
                for obj2 in self.obj_code:
                    for obj3 in self.obj_code:
                        for obj4 in self.obj_code:
                            if obj1 != obj2 and obj1 != obj3 and obj1 != obj4 and obj2 != obj3 and obj2 != obj4 and \
                                    obj3 != obj4:
                                rule_str = ""
                                rule_str += self.impl_code[0]  # ->
                                rule_str += self.con_code[0]  # AND
                                rule_str += self.right_code[3]  # lefts
                                rule_str += obj1
                                rule_str += obj2
                                rule_str += self.con_code[0]  # AND
                                rule_str += self.right_code[3]  # lefts
                                rule_str += obj3
                                rule_str += obj4
                                rule_str += self.right_code[3]  # lefts
                                rule_str += obj2
                                rule_str += obj3
                                rule_str += direction
                                rule_str += obj1
                                rule_str += obj3
                                this_rules.append(Rule(rule_str, False))
        return this_rules

    def create_three_step_right(self):
        this_rules = []
        for direction in self.right_code:
            for obj1 in self.obj_code:
                for obj2 in self.obj_code:
                    for obj3 in self.obj_code:
                        for obj4 in self.obj_code:
                            if obj1 != obj2 and obj1 != obj3 and obj1 != obj4 and obj2 != obj3 and obj2 != obj4 and \
                                    obj3 != obj4:
                                rule_str = ""
                                rule_str += self.impl_code[0]  # ->
                                rule_str += self.con_code[0]  # AND
                                rule_str += self.right_code[3]  # lefts
                                rule_str += obj1
                                rule_str += obj2
                                rule_str += self.con_code[0]  # AND
                                rule_str += self.right_code[3]  # lefts
                                rule_str += obj3
                                rule_str += obj4
                                rule_str += self.right_code[3]  # lefts
                                rule_str += obj2
                                rule_str += obj3

                                rule_str += direction
                                rule_str += obj1
                                rule_str += obj3
                                rule_str += self.con_code[0]  # AND
                                rule_str += direction
                                rule_str += obj1
                                rule_str += obj4
                                rule_str += self.con_code[0]  # AND
                                rule_str += direction
                                rule_str += obj2
                                rule_str += obj3
                                rule_str += self.con_code[0]  # AND
                                rule_str += direction
                                rule_str += obj2
                                rule_str += obj4
                                rule_str += direction
                                rule_str += obj3
                                rule_str += obj4
                                this_rules.append(Rule(rule_str, False))
        return this_rules

    def create_three_step_left_org(self):
        """
        creates rules like: a L b AND b L c AND c L d -> a L d
        :return:
        """
        this_rules = []
        for direction in self.left_code:
            for obj1 in self.obj_code:
                for obj2 in self.obj_code:
                    for obj3 in self.obj_code:
                        for obj4 in self.obj_code:
                            if obj1 != obj2 and obj1 != obj3 and obj1 != obj4 and obj2 != obj3 and obj2 != obj4 and \
                                    obj3 != obj4:
                                rule_str = ""
                                rule_str += self.impl_code[0]  # ->
                                rule_str += self.con_code[0]  # AND
                                rule_str += self.left_code[3]  # lefts
                                rule_str += obj1
                                rule_str += obj2
                                rule_str += self.con_code[0]  # AND
                                rule_str += self.left_code[3]  # lefts
                                rule_str += obj3
                                rule_str += obj4
                                rule_str += self.left_code[3]  # lefts
                                rule_str += obj2
                                rule_str += obj3
                                rule_str += direction
                                rule_str += obj1
                                rule_str += obj3
                                this_rules.append(Rule(rule_str, False))
        return this_rules

    def create_three_step_left(self):
        """
        creates rules like: a L b AND b L c AND c L d -> a L d
        :return:
        """
        this_rules = []
        for directionR in self.right_code:
            for directionL in self.left_code:
                for obj1 in self.obj_code:
                    for obj2 in self.obj_code:
                        for obj3 in self.obj_code:
                            for obj4 in self.obj_code:
                                if obj1 != obj2 and obj1 != obj3 and obj1 != obj4 and obj2 != obj3 and obj2 != obj4 and \
                                        obj3 != obj4:
                                    rule_str = ""
                                    rule_str += self.impl_code[0]  # ->
                                    rule_str += self.con_code[0]  # AND
                                    rule_str += self.left_code[3]  # lefts
                                    rule_str += obj1
                                    rule_str += obj2
                                    rule_str += self.con_code[0]  # AND
                                    rule_str += self.left_code[3]  # lefts
                                    rule_str += obj3
                                    rule_str += obj4
                                    rule_str += self.left_code[3]  # lefts
                                    rule_str += obj2
                                    rule_str += obj3

                                    rule_str += self.con_code[0]  # AND
                                    rule_str += directionL
                                    rule_str += obj1
                                    rule_str += obj3
                                    rule_str += self.con_code[0]  # AND
                                    rule_str += directionL
                                    rule_str += obj1
                                    rule_str += obj4
                                    rule_str += self.con_code[0]  # AND
                                    rule_str += directionL
                                    rule_str += obj2
                                    rule_str += obj4
                                    rule_str += self.con_code[0]  # AND
                                    rule_str += directionR
                                    rule_str += obj3
                                    rule_str += obj1
                                    rule_str += self.con_code[0]  # AND
                                    rule_str += directionR
                                    rule_str += obj4
                                    rule_str += obj1
                                    rule_str += directionR
                                    rule_str += obj4
                                    rule_str += obj2
                                    this_rules.append(Rule(rule_str, False))
        return this_rules
