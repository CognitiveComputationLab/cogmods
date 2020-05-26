
from enum import Enum

 
class ChoiceClasses(Enum):
    logic = 1
    if_to_iff = 2
    or_to_and = 3
    iff_to_if = 4
    nothing = 5
    negative_literal_premise = 6
    atmospheric = 7
    ignore_not = 8
    anti_atmospheric = 9
    not_classified = 10


class PremiseTree:
    """
    Class representing premises as trees
    used for calculating truth values of premises
    """

    def __init__(self, variables):
        root = RootNode()
        self.listOfNodes = [root]
        self.leafList = []
        self.variables = variables

    def parse_premise_to_tree(self, premise):
        """
        Parses a premise into a tree data structure
        :param premise: list of premise components|string
                                premise components are the different variables and operators connecting them
                                i.e. ['not', 'and', 'A', 'B' ]
        """
        for x in premise:
            x = x.lower()
            if x == 'not':
                n = NotNode()
                self.build_tree_helper(n)
            elif x == 'if':
                n = IfNode()
                self.build_tree_helper(n)
            elif x == 'iff':
                n = IffNode()
                self.build_tree_helper(n)
            elif x == 'or':
                n = OrNode()
                self.build_tree_helper(n)
            elif x == 'and':
                n = AndNode()
                self.build_tree_helper(n)
            else:
                n = VariableNode()
                n.name = x
                self.build_tree_helper(n)
                self.leafList.append(n)

    def build_tree_helper(self, node):
        """
        helper function to build a tree for a premise
        :param node: next node to be added to the tree -node|PremiseTreeNode
        """
        for x in reversed(self.listOfNodes):
            if x.is_full():
                continue
            else:
                if x.leftChild is None:
                    x.leftChild = node
                    node.parent = x
                    break
                else:
                    x.rightChild = node
                    node.parent = x
                    break
        self.listOfNodes.append(node)

    def clear_assignment(self):
        """
        resets the values assigned to the nodes in the premise tree
        """
        for x in self.listOfNodes:
            x.value = None

    def calculate_value_for_assignment(self, assignment_list, alternate=False):
        """
        Calculates the truth- value of a premise according to the assignment
        :param alternate: if true uses alternate function to evaluate truth value of nodes
        :param assignment_list: tuple of 1s and 0s representing the values assigned to the premises
        :return: truth- value of the premise according to the assignment
                i.e. premise: and;A;B;
                assignment: (0,1)
                -> return False
        """
        # set assignment of the leaf nodes
        for x in self.leafList:
            for i in range(0, len(self.variables)):
                if x.name == self.variables[i]:
                    if assignment_list[i] == 1:
                        x.value = True
                    elif assignment_list[i] == 0:
                        x.value = False
                    break
        # propagate assignment of the leaf-nodes through the tree
        for y in reversed(self.listOfNodes):
            if y.value is not None:
                continue
            else:
                y.evaluate(alternate)
        return self.listOfNodes[0].value


class PremiseTreeNode:
    """
    Base Class of nodes in a premise tree
    """

    def __init__(self):
        self.parent = None
        self.value = None
        self.leftChild = None
        self.rightChild = None
        self.isLeaf = False
        self.name = None

    def evaluate(self, alternate=False):
        """
        calculate boolean value according to the values of the child-nodes
        :return: Value -True/False
        """
        pass

    def is_full(self):
        """
        Helper function for building a Tree from the node.
        :return: True if all possible children slots are already occupied
        """
        pass

    def children_have_assignments(self):
        pass


class RootNode(PremiseTreeNode):

    def __init__(self):
        super().__init__()
        self.name = 'root'

    def is_full(self):
        if self.leftChild is not None:
            return True
        else:
            return False

    def evaluate(self, alternate=False):
        self.value = self.leftChild.value


class AndNode(PremiseTreeNode):

    def __init__(self):
        super().__init__()
        self.name = 'and'

    def evaluate(self, alternate=False):
        self.value = self.leftChild.value and self.rightChild.value

    def is_full(self):
        if (self.leftChild is not None) and (self.rightChild is not None):
            return True
        else:
            return False


class OrNode(PremiseTreeNode):

    def __init__(self):
        super().__init__()
        self.name = 'or'

    def evaluate(self, alternate=False):
        if alternate is True:
            self.value = self.leftChild.value and self.rightChild.value
        else:
            self.value = self.leftChild.value or self.rightChild.value

    def is_full(self):
        if (self.leftChild is not None) and (self.rightChild is not None):
            return True
        else:
            return False


class IfNode(PremiseTreeNode):

    def __init__(self):
        super().__init__()
        self.name = 'if'

    def evaluate(self, alternate=False):
        if alternate is True:
            self.value = ((not self.leftChild.value) or self.rightChild.value) and (
                        self.leftChild.value or (not self.rightChild.value))
        else:
            self.value = (not self.leftChild.value) or self.rightChild.value

    def is_full(self):
        if (self.leftChild is not None) and (self.rightChild is not None):
            return True
        else:
            return False


class IffNode(PremiseTreeNode):

    def __init__(self):
        super().__init__()
        self.name = 'iff'

    def evaluate(self, alternate=False):
        if alternate is True:
            self.value = (not self.leftChild.value) or self.rightChild.value
        else:
            self.value = ((not self.leftChild.value) or self.rightChild.value) and (self.leftChild.value or (not self.rightChild.value))

    def is_full(self):
        if (self.leftChild is not None) and (self.rightChild is not None):
            return True
        else:
            return False


class NotNode(PremiseTreeNode):

    def __init__(self):
        super().__init__()
        self.name = 'not'

    def evaluate(self, alternate=False):
        if alternate is False:
            self.value = not self.leftChild.value
        else:
            self.value = self.leftChild.value

    def is_full(self):
        if self.leftChild is not None:
            return True
        else:
            return False


class VariableNode(PremiseTreeNode):

    def __init__(self):
        super().__init__()
        self.isLeaf = True
        self.name = None

    def evaluate(self, alternate=False):
        pass

    def is_full(self):
        return True
