class Node:
    """
        represents a node in the tree representation for the rule.
    """
    code = ""

    def __init__(self, symbol="", left=None, right=None, parent=None):
        """
        Basic constructor for a node class.
        :param symbol:
        :param left:
        :param right:
        :param parent:
        """
        self.symbol = symbol
        self.left = left
        self.right = right
        self.parent = parent
        self.make_code()
        self.coding = ""

    def __str__(self):
        return self.symbol

    def make_code(self):
        """
        translates its own symbol to its coding.
        :return: nothing
        """
        if self.symbol == "A":
            self.code = "11000"
        if self.symbol == "B":
            self.code = "11001"
        if self.symbol == "C":
            self.code = "11010"
        if self.symbol == "D":
            self.code = "11011"
        if self.symbol == "E":
            self.code = "11100"

        if self.symbol == "AND":
            self.code = "1000"
        if self.symbol == "OR":
            self.code = "1001"
        if self.symbol == "->":
            self.code = "1010"
        if self.symbol == "<->":
            self.code = "1011"

        if self.symbol == "right1":
            self.code = "000000"
        if self.symbol == "right2":
            self.code = "000001"
        if self.symbol == "right3":
            self.code = "000010"
        if self.symbol == "rights":
            self.code = "000011"

        if self.symbol == "left1":
            self.code = "000100"
        if self.symbol == "left2":
            self.code = "000101"
        if self.symbol == "left3":
            self.code = "000110"
        if self.symbol == "lefts":
            self.code = "000111"

        if self.symbol == "front1":
            self.code = "001000"
        if self.symbol == "front2":
            self.code = "001001"
        if self.symbol == "front3":
            self.code = "001010"
        if self.symbol == "fronts":
            self.code = "001011"

        if self.symbol == "back1":
            self.code = "001100"
        if self.symbol == "back2":
            self.code = "001101"
        if self.symbol == "back3":
            self.code = "001110"
        if self.symbol == "backs":
            self.code = "001111"

    def visit_with_polish_notation(self):
        """
        This visits the Node and creates the polish notation of its children.
        :return: polish notation of himself and his children
        """
        polish_notation = self.symbol + " "
        if self.left is not None:
            polish_notation += self.left.visit_with_polish_notation()
        if self.right is not None:
            polish_notation += self.right.visit_with_polish_notation()
        return polish_notation

    def visit_find_premises(self, prem):
        if "left" in self.symbol or "right" in self.symbol or "front" in self.symbol or "back" in self.symbol:
            prem.append(self)
        else:
            if self.left is not None:
                self.left.visit_find_premises(prem)
            if self.right is not None:
                self.right.visit_find_premises(prem)

    def visit_easy_read(self):
        """
        This visits the node and creates a easily readable presentation of its tree
        :return: easily readable representation of its tree.
        """
        left_string = " "
        right_string = " "
        if self.right is not None:
            right_string = self.right.visit_easy_read()
        if self.left is not None:
            left_string = self.left.visit_easy_read()
        return left_string + self.symbol + right_string

    def visit_make_coding(self):
        """
        this constructs the binary coding of the tree.
        :return: a string of the binary coding of the tree.
        """
        coding = self.code
        if self.left is not None:
            coding += self.left.visit_make_coding()
        if self.right is not None:
            coding += self.right.visit_make_coding()
        return coding

    def code_to_rule(self, code, parent):
        """
        This translates a Code to a rule in its tree representation.
        By step by step decoding of the given lib. It checks the first 2 bits of the lib to determine what will follow
        It then decodes the next bits depending on what was decoded in the first to bits and fills the node with the
        gathered information. After this it will create a child node and pass it the lib with the currently decoded
        part missing. It will always put the created child on the left side except for the case, that the left side is
        already a object (A,B,C,D,E)
        :param code: binary coding of the rule.
        :param parent: parent node of the current node.
        :return: nothing.
        """
        if code is None:
            return
        if len(code) == 0:
            return
        if code[0] == "1":
            # case 1
            if code[1] == "0":
                # case 10 AND/OR/->/<->
                if code[2] == "1" and code[3] == "0":
                    # case 1010 ->
                    self.symbol = "->"
                    self.code = "1010"
                elif code[2] == "1" and code[3] == "1":
                    # case 1011 ->
                    self.symbol = "<->"
                    self.code = "1011"
                elif code[2] == "0" and code[3] == "0":
                    # case 1000 AND
                    self.symbol = "AND"
                    self.code = "1000"
                elif code[2] == "0" and code[3] == "1":
                    # case 1001 OR
                    self.symbol = "OR"
                    self.code = "1001"
                child = Node("", None, None, self)
                self.left = child
                child.code_to_rule(code[4:], self)

            elif code[1] == "1":
                # case 11 A/B/C/D/E
                if code[2] == "0" and code[3] == "0" and code[4] == "0":
                    # case 11000 A
                    self.symbol = "A"
                    self.code = "11000"
                elif code[2] == "0" and code[3] == "0" and code[4] == "1":
                    # case 11001 B
                    self.symbol = "B"
                    self.code = "11001"
                elif code[2] == "0" and code[3] == "1" and code[4] == "0":
                    # case 11010 C
                    self.symbol = "C"
                    self.code = "11010"
                elif code[2] == "0" and code[3] == "1" and code[4] == "1":
                    # case 11011 D
                    self.symbol = "D"
                    self.code = "11011"
                elif code[2] == "1" and code[3] == "0" and code[4] == "0":
                    # case 11100 E
                    self.symbol = "E"
                    self.code = "11100"
                if parent.left == self:
                    # case left
                    sibling = Node("", None, None, self.parent)
                    self.parent.right = sibling
                    sibling.code_to_rule(code[5:], self.parent)
                elif parent.right == self:
                    # case right
                    current_node = self.parent
                    # search for the next node which has no entry on the right child
                    while current_node.right is not None and current_node.parent is not None:
                        current_node = current_node.parent
                    if len(code) > 5:
                        child = Node("", None, None, current_node)
                        current_node.right = child
                        child.code_to_rule(code[5:], current_node)

        elif code[0] == "0":
            # case 0
            if code[1] == "0":
                # case 00 right/left/front/back
                if code[2] == "0":
                    # case 000 right/left
                    if code[3] == "0":
                        # case 0000 right
                        if code[4] == "0" and code[5] == "0":
                            # case 000000 right1
                            self.symbol = "right1"
                            self.code = "000000"
                        elif code[4] == "0" and code[5] == "1":
                            # case 000001 right2
                            self.symbol = "right2"
                            self.code = "000001"
                        elif code[4] == "1" and code[5] == "0":
                            # case 000010 right3
                            self.symbol = "right3"
                            self.code = "000010"
                        elif code[4] == "1" and code[5] == "1":
                            # case 000011 rights
                            self.symbol = "rights"
                            self.code = "000011"

                    elif code[3] == "1":
                        # case 0001 left
                        if code[4] == "0" and code[5] == "0":
                            # case 000100 left1
                            self.symbol = "left1"
                            self.code = "000100"
                        elif code[4] == "0" and code[5] == "1":
                            # case 000101 left2
                            self.symbol = "left2"
                            self.code = "000101"
                        elif code[4] == "1" and code[5] == "0":
                            # case 000110 left3
                            self.symbol = "left3"
                            self.code = "000110"
                        elif code[4] == "1" and code[5] == "1":
                            # case 000111 left1
                            self.symbol = "lefts"
                            self.code = "000111"

                elif code[2] == "1":
                    # case 001 front/back
                    if code[3] == "0":
                        # case 0010 front
                        if code[4] == "0" and code[5] == "0":
                            # case 001000 front1
                            self.symbol = "front1"
                            self.code = "001000"
                        elif code[4] == "0" and code[5] == "1":
                            # case 001001 front2
                            self.symbol = "front2"
                            self.code = "001001"
                        elif code[4] == "1" and code[5] == "0":
                            # case 001010 front3
                            self.symbol = "front3"
                            self.code = "001010"
                        elif code[4] == "1" and code[5] == "1":
                            # case 001000 fronts
                            self.symbol = "fronts"
                            self.code = "001011"
                    elif code[3] == "1":
                        # case 0011 back
                        if code[4] == "0" and code[5] == "0":
                            # case 001100 back1
                            self.symbol = "back1"
                            self.code = "001100"
                        if code[4] == "0" and code[5] == "1":
                            # case 001101 back2
                            self.symbol = "back2"
                            self.code = "001101"
                        if code[4] == "1" and code[5] == "0":
                            # case 001110 back3
                            self.symbol = "back3"
                            self.code = "001110"
                        if code[4] == "1" and code[5] == "1":
                            # case 001111 backs
                            self.symbol = "backs"
                            self.code = "001111"
            child = Node("", None, None, self)
            self.left = child
            child.code_to_rule(code[6:], self)
