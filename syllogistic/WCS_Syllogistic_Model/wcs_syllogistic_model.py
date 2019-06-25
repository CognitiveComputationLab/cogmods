"""
Weak-Completion-Semantics model for syllogistic reasoning.

Created on 24.10.2018

@author: Julia Mertesdorf <julia.mertesdorf@gmail.com>
"""


from operator import attrgetter
import random
import ccobra
import pandas as pd



# ----------------------------------------- MODEL CLASS --------------------------------------------

class WCSSyllogistic(ccobra.CCobraModel):
    """
    This class implements the Weak-Completion-Semantics approach for solving syllogistic
    reasoning tasks.
    The function "compute_problem" solves a single syllogism task by creating a logic program
    founded on the two quantors and the figure, and reason based on the least model of the
    logic program. The class "LogicProgram" is a helper-class which contains all functions
    to create and compute logic programs.
    """

    def __init__(self, name="WCSSyllogistic"):
        """
        The model class contains a logic program-object, which is used to compute solutions for
        syllogistic reasoning tasks. This program is reset when computing a new problem.
        The parameter "enabled_principles" specifies, which additional, advanced principles are
        currently enabled and will be used to solve a task:
        - First value: deliberate Generalization used for mood A & I
        - Second value: Converse interpretation used for mood I
        - Third value: Converse interpretation used for mood E
        - Fourth value: Contraposition used for mood A
        - Fifth value: Abduction used (all moods)

        The list enabled_filters determines, if any filters are used after the final conclusion
        is calculated:
        - First value: Filter conclusion with the matching strategy
        - Second value: Filter conclusion with biased conclusions for figure 1

        To speed up the calculation process, all answers for all possible combinations of the five
        advanced principles are saved and retrieved from a csv-file calles "all_variations".

        With the boolean variable "print_output" enabled, the problem solving process can
        be printed.
        """
        super(WCSSyllogistic, self).__init__(name, ['syllogistic'], ['single-choice'])
        self.lp_ = LogicProgram() # Internal logic program of the model

        self.dataset = None
        self.enabled_principles = [0, 0, 0, 0, 0]
        self.enabled_filters = [0, 0]
        self.print_output = True

        self.principle_combinations = []
        ass = [0, 1]
        for a__ in ass:
            for b__ in ass:
                for c__ in ass:
                    for d__ in ass:
                        for e__ in ass:
                            self.principle_combinations.append([str(a__)+str(b__)+str(c__)
                                                                +str(d__)+str(e__), 0])

        self.pred_dict = {}
        try:
            pred_df = pd.read_csv("all_variations.csv")
            self.pred_dict = dict(zip(pred_df["problem"].tolist(),
                                      [x.split(';') for x in pred_df['result']]))
        except FileNotFoundError:
            print("\n\n\nThe wcs-predictions are not computed yet!\n",
                  "Please run the function >compute_all_variations()< first.\n\n\n")

    def predict(self, item, **kwargs):
        """
        This function predicts the solution to a syllogistic input problem by a look up
        in a precalculated csv-file.
        """
        # Encode the task to a string of the form "AA1"
        enc_task = ccobra.syllogistic.encode_task(item.task)

        # Calculate which principles (and filters) are enabled for the current problem.
        best_comb = max(self.principle_combinations, key=lambda x: x[1])[0]

        # Retrieve answer from precalculated dict (csv-file).
        epr = [int(i) for i in best_comb]
        problem = enc_task+str(epr[0])+str(epr[1])+str(epr[2])+str(epr[3])+str(epr[4])
        result = self.pred_dict[problem][0]

        # Transform dictionary-entry (string) back to list of strings.
        cleaned_results = self.result_string_to_list(result)

        # Select and return a random answer, if multiple solutions exist.
        response = cleaned_results[random.randint(0, len(cleaned_results)-1)]
        dec_resp = ccobra.syllogistic.decode_response(response, item.task)
        return dec_resp

    def pre_train(self, dataset):
        for subj in dataset:
            for problem in subj:
                task = problem["item"].task
                self.determine_promising_combs(task, problem["response"])

        # Normalize the principle combination results by dividing through the number of subjects.
        for index, value in enumerate(self.principle_combinations):
            self.principle_combinations[index][1] = int(value[1]/len(dataset))

        # Multiply the principle combination results with a factor (here: 0.3) to determine the
        # influence of the pre-training results.
        for index, value in enumerate(self.principle_combinations):
            self.principle_combinations[index][1] = int(value[1] * 0.3)

    def adapt(self, item, target, **kwargs):
        self.determine_promising_combs(item.task, target)

    def determine_promising_combs(self, task, target):
        """
        Helperfunction for adapt and pre_train.
        This function calculates all responses/solutions to all possible principle combinations,
        given an input task (32 possible combinations). Then it determines, whether the responses
        are equal to the given input target (the response, the according subject has given).
        If that is the case, the counter for the according combination is increased and therefore
        the combination is more likely to be chosen in future.
        """
        enc_task = ccobra.syllogistic.encode_task(task)
        comb = self.principle_combinations

        # Iterate through all combinations and determine whether the response with regards
        # to each combination predicts the target successfully.
        for index, _ in enumerate(self.principle_combinations):
            problem = (enc_task+comb[index][0][0]+comb[index][0][1]+comb[index][0][2]
                       +comb[index][0][3]+comb[index][0][4])
            result = self.pred_dict[problem][0]
            result_list = self.result_string_to_list(result)
            for res in result_list:
                dec_resp = ccobra.syllogistic.decode_response(res, task)

                # Increase the counter for the combination, if this combination was successfull.
                if dec_resp == target:
                    self.principle_combinations[index][1] += 1

        self.principle_combinations = sorted(self.principle_combinations, key=lambda x: x[1])[::-1]

    @staticmethod
    def result_string_to_list(result_string):
        """
        Helper function for predict and adapt.
        This function takes a string as an input and returns a list of all separate items
        (since the string, which encodes a list, was saved as a simple string in the csv-file,
        it needs to be re-transformed to a list to be useable for the WCS-model).
        """
        result_list = result_string.split(",")
        cleaned_results = []
        for it_ in result_list:
            new_item = it_.replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
            cleaned_results.append(new_item)
        return cleaned_results

    def compute_problem_from_scratch(self, problem, enabled_principles=None):
        """
        This function calculates one syllogistic input problem under the WCS, which is encoded
        as a string of two characters and a number (for instance: "AE1" encodes the syllogism
        "All a are b, No b are c").
        The optional attribute "enabled_principles" describes, which principles should be
        enabled when calculating the solution to the input problem.

        The problem solving process consists of the following steps:
        1. Given the figure of the problem, create two strings containing the objects of
           the premises ("a", "b", "c"), which will later be used as the names of the atoms.
        2. Create a logic program for each quantor (each quantor is encoded differently, leading
           to the insertion of 3-8 rules to the logic program, depending on the quantor).
           If advanced principles are enabled, depending on the problem, more rules are inserted
           (up to 8 more rules per quantor).
        3. Ground the logic program (all variables in the atoms ("a(None)") are replaced
           by the object instances ("a(o1)", "a(o2)" etc)).
        4. Compute the least model of the logic program (find a minimal assignment for the
           atoms, which satisfies all rules in the logic program).
        5. Compute the conclusion(s) of the problem (given the truth-assignments of the atoms,
           entail all possible conclusions).
        6. In case the conclusion is "NVC" and abduction is enabled, try to find alternative
           conclusions to "NVC" with the help of abduction.
        7. In case any filter is enabled, filter the conclusions (matching strategy and/or biased
           conclusions of figure 1).
        """
        if enabled_principles:
            self.enabled_principles = enabled_principles

        self.lp_ = LogicProgram() # Create new logic program.
        figure = self.calculate_figure(problem)                                              # (1)

        # Check if contraposition is applicable (only if one mood is A, the other E/O).
        reset_for_csv = False
        if self.enabled_principles[3]:
            if (problem[0] == "A" and (problem[1] == "E" or problem[1] == "O")):
                self.enabled_principles[3] = 1
            elif ((problem[0] == "E" or problem[0] == "O") and problem[1] == "A"):
                self.enabled_principles[3] = 1
            else:
                self.enabled_principles[3] = 0
                reset_for_csv = True

        self.create_quantor_program(problem[0], figure[0])                                   # (2)
        self.create_quantor_program(problem[1], figure[1])                                   # (2)

        # Enable principle converse interpretation for mood I / E (if applicable).           # (2)
        for index, quantor in enumerate([problem[0], problem[1]]):
            if quantor == "I" and self.enabled_principles[1]: # mood I
                self.create_quantor_program(quantor, figure[index][::-1], "_converse")
            if quantor == "E" and self.enabled_principles[2]: # mood E
                self.create_quantor_program(quantor, figure[index][::-1], "_converse")

        if self.print_output:
            print("\nSYLLOGISTIC PROBLEM: ", problem, "\n")
            self.print_syllogistic_problem(problem, figure)
            self.lp_.print_lp_truth("CREATION")

        self.lp_.ground_program()                                                            # (3)
        if self.print_output:
            self.lp_.print_lp_truth("GROUNDING")

        self.lp_.compute_least_model(self.print_output)                                      # (4)
        if self.print_output:
            self.lp_.print_lp_truth("LEAST MODEL")

        result = self.lp_.compute_conclusion("a", "c", self.enabled_principles)              # (5)
        if self.print_output:
            self.lp_.print_atoms()
            print("\nRESULT:", result, "\n")

        if result[0] == "NVC" and self.enabled_principles[4]:                                # (6)
            result = self.lp_.abduction("a", "c", self.enabled_principles)
            if self.print_output:
                print("RESULT AFTER ABDUCTION:", result, "\n")

        filtered_result = self.filter_conclusion(problem, result)                            # (7)
        if self.print_output:
            print(problem, "       Pre-filter:", result, "      Final:", filtered_result)

        if reset_for_csv:
            self.enabled_principles[3] = 1
        return filtered_result

    @staticmethod
    def print_syllogistic_problem(problem, figure):
        """
        Function prints out a human-readable syllogism task.
        """
        incl_not = ["", ""]
        quantors = ["", ""]
        for index, _ in enumerate(quantors):
            if problem[index] == "A":
                quantors[index] = "All"
            elif problem[index] == "I":
                quantors[index] = "Some"
            elif problem[index] == "E":
                quantors[index] = "No"
            else:
                quantors[index] = "Some"
                incl_not[index] = "not"
        print(quantors[0], " ", figure[0][0], " are ", incl_not[0], " ", figure[0][1])
        print(quantors[1], " ", figure[1][0], " are ", incl_not[1], " ", figure[1][1], "\n")

    @staticmethod
    def calculate_figure(problem):
        """
        This function translates a given problem-order into two premises, containing two objects
        each ("ab"/"ba" and "bc"/"cb" accordingly). These objects will be used as the names
        for the atoms in the logic program.
        """
        prem1 = ""
        prem2 = ""
        figure = problem[2]
        if figure == "1":
            prem1 = "ab"
            prem2 = "bc"
        elif figure == "2":
            prem1 = "ba"
            prem2 = "cb"
        elif figure == "3":
            prem1 = "ab"
            prem2 = "cb"
        else:
            prem1 = "ba"
            prem2 = "bc"
        return [prem1, prem2]

    def create_quantor_program(self, quantor, premise, converse=""):
        """
        This function creates the according logic program for each quantor, given the quantor
        (A = All, I = Some, E = No, O = Some not) and a premise.

        The WCS approach is applied to syllogisms with the help of some principles, which were
        developed to create a logical form for the representation of syllogisms.
        The basic principles (used for every problem) are:
        - Quantified Assertion as Conditional (all moods)
        - Licenses for Inferences (all moods)
        - Existential Import / Gricean Implicature (all moods)
        - Negation by Transformation (mood E, O)
        - Unknown Generalization (mood I, O)
        - No derivation by double negation (mood E, O)

        The advanced principles, which are not used for every problem, are:
        - converse interpretation (mood E, I)
        - Deliberate generalization (context operator) (mood A, I)
        - contraposition (mood A)
        - Search for alternative conclusions/ abduction (all moods)
        (the principles converse interpretation and abduction are handled in the function
        "compute_problem")
        """
        # Create atoms + object for main rule.
        atom1 = self.lp_.create_atom(premise[0], None, "atom_w_obj", False)
        atom2 = self.lp_.create_atom(premise[1], None, "atom_w_obj", False)
        obj = self.lp_.create_object("o")    # first object for existential import

        # Create fact: atom1 is True for a certain object o. (Principle Existential import)
        atom1b = self.lp_.create_atom(atom1.name, obj, "atom_w_obj")
        self.lp_.create_fact(atom1b, True, "fact_existential_import"+converse)  # a(o1) <-- T

        if quantor == "A":
            # Create Main rule + general false abnormality-fact.
            rule1 = self.lp_.create_rule("rule_conditionals_and_licenses_A",
                                         atom2, atom1, True, True, False)  # b <-- a and -ab

            # Enable contraposition principle.
            if self.enabled_principles[3]:
                not_atom2 = OperatorNot(atom2)
                atom1_neg = self.lp_.create_atom(atom1.name+"´´", atom1.obj, "atom_w_obj", False)
                not_atom1_neg = OperatorNot(atom1_neg)

                self.lp_.create_rule("rule_contraposition", atom1_neg, not_atom2, True, True, False)
                self.lp_.create_rule("rule_neg_transformation", atom1, not_atom1_neg, True, False)

                # integrity constraint for contraposition
                and_ = OperatorAnd(atom1_neg, atom1, "and")
                self.lp_.create_rule("ic_integrity_constraint", None, and_)  # U <-- b´ and b

        if quantor == "I":
            # Create Main rule + false abnormality-fact for obj1.
            rule1 = self.lp_.create_rule("rule_conditionals_and_licenses_I"+converse,
                                         atom2, atom1, True, True, False, obj)

        if quantor == "I" or quantor == "O":
            obj2 = self.lp_.create_object("o")   # create second object for unknown generalization

            # Create second fact. (Principle unknown generalization)
            atom1c = self.lp_.create_atom(atom1.name, obj2, "atom_w_obj")
            self.lp_.create_fact(atom1c, True, "fact_unknown_generalization"+converse)

        # Enable Deliberate generalization and add Context operator.
        if (quantor == "A" or quantor == "I") and self.enabled_principles[0]:
            atom2_neg = self.lp_.create_atom(atom2.name+"´´", atom2.obj, "atom_w_obj", False) #b´(X)
            ctxt = ContextOperator(atom2_neg)
            ab_atom = rule1.right.right.content
            self.lp_.create_rule("rule_deliberate_generalization"+converse, ab_atom, ctxt)

            if quantor == "I":
                ab_atom2 = self.lp_.create_atom(ab_atom.name, obj2, "atom_w_obj")
                self.lp_.create_fact(ab_atom2, None, "fact_deliberate_generalization"+converse)

        # Negation by Transformation: Create Main rule, which needs to be split
        # (since negative heads are not allowed) and add according abnormality-facts.
        if quantor == "E" or quantor == "O":
            atom2_neg = self.lp_.create_atom(atom2.name+"´´", atom2.obj, "atom_w_obj", False) #b´(X)
            not_atom2_neg = OperatorNot(atom2_neg)

            if quantor == "E":      # b´ <-- a and -ab1;    b <-- -b´ and -ab2
                self.lp_.create_rule("rule_conditionals_and_licenses_E"+converse,
                                     atom2_neg, atom1, True, True, False)
                self.lp_.create_rule("rule_neg_transformation_E"+converse,
                                     atom2, not_atom2_neg, True, True, False, obj)
            elif quantor == "O":    # b` <-- a and -ab1;    b <-- -b´ and -ab2
                self.lp_.create_rule("rule_conditionals_and_licenses_O",
                                     atom2_neg, atom1, True, True, False, obj)
                self.lp_.create_rule("rule_neg_transformation_O",
                                     atom2, not_atom2_neg, True, True, False, obj, obj2)

            # Create integrity constraint: atom2 & atom2_neg shouldn´t be true at the same time.
            and_ = OperatorAnd(atom2_neg, atom2, "and")
            self.lp_.create_rule("ic_integrity_constraint", None, and_)  # U <-- b´ and b

    def filter_conclusion(self, problem, conclusion):
        """
        This function applies two possible filters (the matching strategy and biased conclusions
        of figure 1), in case they are enabled in "enabled_filters", and returns the
        filtered conclusions.
        """
        filtered_result = conclusion
        if self.enabled_filters[0]:
            filtered_result = self.matching_strategy(problem, filtered_result)
        if self.enabled_filters[1]:
            filtered_result = self.biased_conclusions(problem, filtered_result)
        return filtered_result

    @staticmethod
    def matching_strategy(problem, conclusion):
        """
        This function implements a simple heuristic strategy called the matching strategy
        and filters the conclusions of the WCS-model according to the predictions of the
        matching strategy.
        The matching strategy postulates that the conclusion-premise cannot contain an operator
        with a lower "conservative level" than the conservative level of the quantors in the
        problem. The conservative rank on the quantors is defined as follows (left is least
        conservative): A < I <= O < E). In case all conclusions are filtered out by the matching
        strategy, the conclusion "NVC" is added.
        """
        ranking_order = ["A", "I", "O", "E"]
        border_index = ranking_order.index(problem[0])
        if border_index < ranking_order.index(problem[1]):
            border_index = ranking_order.index(problem[1])

        new_conclusion = []
        for answer in conclusion:
            if answer == "NVC":
                new_conclusion.append(answer)
                continue
            ans_index = ranking_order.index(answer[0])
            if ans_index >= border_index:
                new_conclusion.append(answer)
        if not new_conclusion:
            new_conclusion.append("NVC")
        return new_conclusion

    @staticmethod
    def biased_conclusions(problem, conclusion):
        """
        This function implements a heuristic strategy which filters out all unlikely conclusions
        for figure1-syllogisms. This heuristic strategy is based on the empirical observation,
        that for all figure1-problems, people tend to answer Xac (X el. of {A, E, I, O}) and almost
        never answer with Xca. In case Xac is an invalid syllogism, other conclusions
        are considered and not deleted by the heuristic.
        """
        if problem[2] != "1":
            return conclusion
        ranking_order = ["A", "I", "O", "E"]
        border_index = ranking_order.index(problem[0])
        if border_index < ranking_order.index(problem[1]):
            border_index = ranking_order.index(problem[1])
        new_conclusion = []
        for answer in conclusion:
            if (answer == "NVC" or (problem in INVALID_SYLLOGISMS and
                                    ranking_order[border_index]+"ac" not in conclusion)):
                new_conclusion.append(answer)
                continue
            ans_index = ranking_order.index(answer[0])
            if (ans_index == border_index and answer[1:] == "ac"):
                new_conclusion.append(answer)
        return new_conclusion



# ----------------------------------------- LOGIC OPERATOR CLASSES ---------------------------------

class Formula(object):
    """
    This class implements basic formula, which have a type and a boolean value.
    All formulas (and all child-classes of formulas) have an evaluation-function to
    calculate their current boolean value.
    """
    def __init__(self):
        self.boolean_val = None
        self.type = "formula"

    def evaluate(self):
        """
        Function for evaluating the own current boolean value.
        """
        return self.boolean_val

    def get_atom(self):
        """
        Function to return the atom (Atom is a subclass of Formula, so return None).
        """
        return None


class Atom(Formula):
    """
    This class implements atoms. All atoms have a name, a boolean value, and an
    evaluation-function to calculate their current boolean value.
    Moreover, atoms can contain objects. If an atom contains an object and has
    the boolean value "True", then this object has the feature which is
    encoded by the atom(name). For instance, "baker(o1)" with boolean_val = True
    encodes the fact that o1 is a baker.
    The attributes "type" and "subtype" are used to distinguish different kinds of formula
    and atoms (atoms can have the subtypes "atom_w_obj" (= "atoms with object"), "bool"
    (= atom which is a boolean) and "no_obj" (= "atom without object")).
    """
    def __init__(self, name, bool_val=None, obj=None, subtype="no_obj"):
        super().__init__()
        self.name = name
        self.type = "atom"
        self.subtype = subtype
        self.boolean_val = bool_val
        self.obj = obj

    def evaluate(self):
        """
        Function for evaluating the own current boolean value.
        In case the atom was instantiated with one of the boolean values "True",
        "False" or "None" (= Unknown) as the name, the boolean value is set accordingly.
        """
        if self.subtype == "bool" and(self.name is True or self.name is False or self.name is None):
            self.boolean_val = self.name
        return self.boolean_val

    def get_atom(self):
        """
        Function to return the atom.
        """
        if self.subtype == "atom_w_obj":
            return self
        return None

    def __str__(self):
        if self.subtype != "atom_w_obj":
            return str(self.name)
        return self.name + "(" + str(self.obj) + ")"


class ContextOperator(Formula):
    """
    This class implements the context operator. The context operator, wrapped around an atom,
    is evaluated to True, if the boolean value of the atom inside is True, else False.
    The context operator was introduced by Dietz, Hoelldobler and Pereira in order
    to solve a technical bug regarding syllogisms with "A"-quantors.
    (For further information, see E.-A. Dietz, S. Hoelldobler, L. M. Pereira.
    Contextual reasoning: Usually birds can abductively fly. 2017).
    """
    def __init__(self, content, name="ctxt"):
        super().__init__()
        self.type = "context"
        self.symbol = "ctxt"
        self.name = name
        self.content = content

    def evaluate(self):
        con_val = self.content.evaluate()
        if con_val is True:
            self.boolean_val = True
        else:
            self.boolean_val = False
        return self.boolean_val

    def get_atom(self):
        if self.content.type == "atom" and self.content.subtype == "atom_w_obj":
            return self.content
        return None

    def __str__(self):
        return self.symbol + "(" + str(self.content) + ")"


class OperatorNot(Formula):
    """
    This class implements the logic operator "not".
    """
    def __init__(self, content, name="not"):
        super().__init__()
        self.type = "not"
        self.name = name
        self.content = content

    def evaluate(self):
        con_val = self.content.evaluate()
        if con_val is None:
            self.boolean_val = None
        elif con_val is True:
            self.boolean_val = False
        else:
            self.boolean_val = True
        return self.boolean_val

    def get_atom(self):
        if self.content.type == "atom" and self.content.subtype == "atom_w_obj":
            return self.content
        return None

    def __str__(self):
        return self.type + "(" + str(self.content) + ")"


class OperatorBijective(Formula):
    """
    This class implements the parent class of bijective logic operators
    (for logic operator like "and" and "implication").
    """
    def __init__(self, left, right, name):
        super().__init__()
        self.left = left
        self.right = right
        self.name = name
        self.type = "operator_bijective"
        self.symbol = "***"

    def __str__(self):
        return "(" + str(self.left) + " " + self.symbol + " " + str(self.right) + ")"


class OperatorAnd(OperatorBijective):
    """
    This class implements the logic operator "and".
    """
    def __init__(self, left, right, name):
        super().__init__(left, right, name)
        self.type = "and"
        self.symbol = "and"

    def evaluate(self):
        left = self.left.evaluate()
        right = self.right.evaluate()

        if left is True and right is True:
            self.boolean_val = True
        elif left is False or right is False:
            self.boolean_val = False
        else:
            self.boolean_val = None
        return self.boolean_val

    def get_atom(self):
        at_left = self.left.get_atom()
        at_right = self.right.get_atom()
        return [at_left, at_right]


class OperatorImplication(OperatorBijective):
    """
    This class implements the logic operator "implication".
    """
    def __init__(self, left, right, name):
        super().__init__(left, right, name)
        self.type = "implication"
        self.symbol = "<--"

    def evaluate(self):
        left = self.left.evaluate()
        right = self.right.evaluate()

        if left is True or right is False:
            self.boolean_val = True
        elif left is None and right is None:
            self.boolean_val = True
        elif left is False and right is True:
            self.boolean_val = False
        else:
            self.boolean_val = None
        return self.boolean_val

    def get_atom(self):
        at_list = []
        at_left = self.left.get_atom()
        at_right = self.right.get_atom()
        if isinstance(at_left, list):
            for atom in at_left:
                at_list.append(atom)
        else: at_list.append(at_left)
        if isinstance(at_right, list):
            for atom in at_right:
                at_list.append(atom)
        else: at_list.append(at_right)
        return at_list



# ----------------------------------------- LOGIC PROGRAM CLASS ------------------------------------

class LogicProgram():
    """
    This class implements logic programs, which contain rules and positive / negative facts
    (which encode the truth-value of atoms).
    Logic Programs are the basic framework used in Weak-Completion-Semantics.
    The LogicProgram-class contains important functions which encode the main
    concepts used in Weak-Completion-Semantics, like the implementation of the semantic
    operator in "compute_least_model", abduction, as well as many helper-functions.
    """

    def __init__(self):
        """
        Initializes a LogicProgram-object.
        A logic program contains a list of atoms, objects and rules (implications).
        The attributes "X_counter" keep track of the current amount of objects and
        abnormalities in the logic program and are used to give the instances unique names.
        """
        self.atoms = []
        self.objects = []
        self.rules = []
        self.object_counter = 1
        self.ab_counter = 1

    def create_atom(self, name, obj, subtype, append=True):
        """
        Function creates an atom with the specified name and adds it to the atoms-list
        of the logic program (only if the attribute "append" is True).
        In case the atom already exists, the according atom is returned.
        """
        for i in range(0, len(self.atoms)):
            if self.atoms[i].name == name and self.atoms[i].obj == obj:
                return self.atoms[i]
        at_ = Atom(name, None, obj, subtype)
        if append:
            self.atoms.append(at_)
        return at_

    def create_object(self, name):
        """
        This function creates an object (String) and adds it to the internal list of objects
        of the logic program.
        Objects are used as instances of the "classes" representing certain features.
        For instance, "a(o1)" means that the object with name "o1" is an instance of the atom
        with name "a", which symbolizes that o1 has the feature which is encoded by "a"
        (a could stand for something like "astronaut", which would mean that o1 is an astronaut).
        """
        obj = name + str(self.object_counter)
        self.objects.append(obj)
        self.object_counter += 1
        return obj

    def create_rule(self, name, consequent, conditional, ab_=False, ab_fact=False,
                    ab_truth=False, ab_obj=None, ab_obj2=None):
        """
        This function creates a rule (implication), given an input name, a conditional and
        a consequent, and adds the rule to the logic program.
        In case the parameter "ab" was set to True, then a negated abnormality-atom is added
        to the conditional of the rule (cons <-- cond and -ab).
        The parameter "ab_truth" determines the boolean value of the abnormality-atom.
        Depending on this truth-value and optionally some defined objects "ab_obj" and "ab_obj2",
        according facts are created and added. If no object is given (ab_obj == None), then
        the general fact "ab(None) <-- ab_truth" is added. If one or two objects are given,
        then the facts are added with the according objects contained in the ab-atom:
        "ab(ab_obj) <-- ab_truth".
        """
        cond = conditional
        cons = consequent
        # In case the consequent is "None" due to an integrity constraint
        if consequent is None:
            cons = Atom(None, None)

        if ab_:
            ab_1 = self.create_atom("ab" + str(self.ab_counter), None, "atom_w_obj", False)
            ab_2 = ab_
            ab_3 = ab_
            self.ab_counter += 1
            not_ab = OperatorNot(ab_1, "not_ab" + str(self.ab_counter))
            and_ = OperatorAnd(conditional, not_ab, "and_ab")
            cond = and_

            # In case the abnormality-fact shouldn´t be generalized (holds just for one/two
            # object(s)), create ab-atoms which contain the specified objects.
            if ab_obj != None:
                ab_2 = self.create_atom(ab_1.name, ab_obj, "atom_w_obj")
            if ab_obj2 != None:
                ab_3 = self.create_atom(ab_1.name, ab_obj2, "atom_w_obj")

            # Create the facts (if no objects specified, add a general fact)
            if ab_obj != None and ab_obj2 != None:
                self.create_fact(ab_2, ab_truth, "fact_licenses")
                self.create_fact(ab_3, ab_truth, "fact_licenses")
            elif ab_obj != None:
                self.create_fact(ab_2, ab_truth, "fact_licenses")
            elif ab_fact:
                self.create_fact(ab_1, ab_truth, "fact_licenses")

        rule = OperatorImplication(cons, cond, name)
        self.rules.append(rule)
        return rule

    def create_fact(self, consequent, true_false, name="fact"):
        """
        Given a consequent and a boolean value "true_false", this function creates a fact
        of the form "consequent <-- true_false" and adds it to the logic program.
        """
        atom = Atom(true_false, None, None, "bool")
        fact = OperatorImplication(consequent, atom, name)
        self.rules.append(fact)
        return fact

    def get_atom(self, name, obj):
        """
        Function returns the atom which matches the specified name and object.
        """
        for i in range(0, len(self.atoms)):
            if self.atoms[i].name == name and self.atoms[i].obj == obj:
                return self.atoms[i]
        return None

    def get_truth_atoms(self, truth):
        """
        Function returns all atoms of the logic program that match the boolean value
        of the input parameter "truth".
        """
        truth_atoms = []
        for at_ in self.atoms:
            if at_.boolean_val is truth:
                truth_atoms.append(at_)
        return truth_atoms

    def get_facts(self):
        """
        Function returns all facts of the logic program.
        """
        facts = []
        for rule in self.rules:
            if rule.name[:4] == "fact":
                facts.append(rule)
        return facts

    def print_atoms(self):
        """
        Function prints all positve, negative and unknown atoms of the logic program.
        """
        true_atoms = self.get_truth_atoms(True)
        false_atoms = self.get_truth_atoms(False)
        unknown_atoms = self.get_truth_atoms(None)

        pos = ([str(at) for at in true_atoms])
        neg = ([str(at) for at in false_atoms])
        un_ = ([str(at) for at in unknown_atoms])
        print("=" * 127, "\nPositive atoms:\n", pos, "\n\nNegative atoms:\n", neg,
              "\n\nUnknown atoms:\n", un_, "\n", "=" * 127)

    def print_lp_truth(self, state):
        """
        Function prints all rules of the logic program with their according boolean values.
        """
        print("-" * 56, str(state), "-" * 56)
        self.rules.sort(key=attrgetter('name'))
        for rule in self.rules:
            string = rule.name
            blank = 45 - len(string)
            string = string + blank * " " + str(rule)
            blank2 = 100 - len(string)
            string = string + blank2 * " " + str(rule.boolean_val)
            blank3 = 105 - len(string)
            string = string + blank3 * " " + " <<   " + str(rule.left.boolean_val)
            blank4 = 118 - len(string)
            string = string + blank4 * " " + "<--  " + str(rule.right.boolean_val)
            print(string)
        print("-" * 127)

    def get_variable_rules(self):
        """
        Helper-function for the function "ground_program".
        This function retrieves and returns all rules containing variables in the atoms
        (For instance: b(None) <-- a(None)). They are used to ground the logic programm
        according to the existing objects.
        """
        variable_rules = []
        for rule in self.rules:
            atom_list = rule.get_atom()
            has_no_object = False
            for item in atom_list:
                if item != None and item.obj is None:
                    has_no_object = True
            if has_no_object:
                variable_rules.append(rule)
        return variable_rules

    def get_definite_rules(self):
        """
        Helper-function for the function "ground_program".
        This function retrieves all rules that contain only atoms with fixed objects
        (For instance: b(o1) <-- a(o1)).
        """
        variable_rules = self.get_variable_rules()
        non_var_rules = []
        for rule in self.rules:
            if rule not in variable_rules:
                non_var_rules.append(rule)
        return non_var_rules

    def copy_rule_with_obj(self, rule, obj):
        """
        Helper-function for the function "ground_program".
        This function creates a copy of a given rule and inserts the given input object "obj" as
        the object contained in all atoms of the rule. The resulting grounded rule is returned.
        (For instance: The input rule is: "c(None) <-- a(None) and b(None)" and the object "o1".
        The result of the function is the copied and grounded rule: "c(o1) <-- a(o1) and b(o1)").
        """
        head = rule.left
        body = rule.right

        if head.subtype == "atom_w_obj":
            grounded_head = self.create_atom(head.name, obj, head.subtype)
        else:
            grounded_head = head

        if body.type == "and":        # AND
            left_body = body.left
            right_body = body.right
            if left_body.type == "not":
                internal_atom = self.create_atom(left_body.content.name, obj,
                                                 left_body.content.subtype)
                grounded_body_left = OperatorNot(internal_atom)
            else:
                grounded_body_left = self.create_atom(left_body.name, obj, left_body.subtype)
            if right_body.type == "not":
                internal_atom = self.create_atom(right_body.content.name, obj,
                                                 right_body.content.subtype)
                grounded_body_right = OperatorNot(internal_atom)
            else:
                grounded_body_right = self.create_atom(right_body.name, obj, right_body.subtype)
            grounded_body = OperatorAnd(grounded_body_left, grounded_body_right, "and")

        elif body.type == "context":  # CTXT
            internal_atom = self.create_atom(body.content.name, obj, body.content.subtype)
            grounded_body = ContextOperator(internal_atom)
        elif body.type == "not":      # NEG
            internal_atom = self.create_atom(body.content.name, obj, body.content.subtype)
            grounded_body = OperatorNot(internal_atom)
        else:                         # T / F / U
            grounded_body = body

        copied_rule = OperatorImplication(grounded_head, grounded_body, rule.name)
        return copied_rule

    def ground_program(self):
        """
        This function grounds a logic program, meaning that all rules that contain variables
        in the atoms (like "a(None)") are grounded according to all existing objects
        in the logic program.
        The function first retrieves all rules that contain variables. Afterwards the function
        iterates trough all objects of the lp and creates a copy of the variable rules for each
        object. All atoms in the variable rules are set to the current object, in order to
        contain this object instead of the variable "None". This is done for all objects with
        their according copies of variable rules.
        At the end of the function, the internal list of rules in the lp is set to the already
        definite rules and the newly grounded rules.
        """
        var_rules = self.get_variable_rules()
        grounded_rules = []
        for obj in self.objects:
            for rule in var_rules:
                copied_rule = self.copy_rule_with_obj(rule, obj)
                grounded_rules.append(copied_rule)

        self.rules = self.get_definite_rules()
        for rule in grounded_rules:
            self.rules.append(rule)

    def evaluate(self):
        """
        Function evaluates all rules in the logic program by calling their evaluate-function.
        Since the logic-operators are implemented inductively, the evaluation-calls heads down
        from bijective-operators to monotonic operators to the atoms with their true/false/unknown-
        assignments.
        """
        for rule in self.rules:
            rule.evaluate()

    def same_head(self, rule_head):
        """
        Helper-function for "compute_least_model".
        This function searches for all rules in the logic program which have the same head
        as "rule_head". The function then checks whether these bodies all have the boolean value
        "False" (this is a necessary condition to set a head False with the semantic operator).
        If thats the case, return True, else return False.
        """
        result = True
        for rule in self.rules:
            if ((rule.left.subtype == "atom_w_obj") and(rule.left.name == rule_head.name)
                    and (rule.left.obj == rule_head.obj) and (rule.right.boolean_val != False)):
                result = False
        return result

    def atoms_to_assign(self):
        """
        Helper-function for "compute_least_model".
        Check whether the semantic operator is done or has still some atoms to assign.
        In the first case, return an empty list. In the last case, return a list with
        all rule-heads and their boolean values that still need to be assigned.
        """
        atoms_to_assign = []
        for rule in self.rules:
            if (rule.right.boolean_val != rule.left.boolean_val) and (rule.name[:2] != "ic"):
                if rule.right.boolean_val is True:
                    atoms_to_assign.append([rule.name, rule.left, True])
                elif rule.right.boolean_val is False and self.same_head(rule.left):
                    atoms_to_assign.append([rule.name, rule.left, False])
        return atoms_to_assign

    def compute_least_model(self, print_=False):
        """
        This function computes a least model of the logic program by setting rule heads (consequent
        of a rule), that contain objects, to True/False.
        The function applies the semantic operator, introduced by Stenning and Lambalgen, to the
        logic program: If the conditional of a rule is True/False, but it´s consequent has another
        boolean value (leading to a non-satisfied rule), then the consequent is set to True/False
        as well (note: in order to set a rule head to False, all rules that have the same rule head
        need to satisfy the condition, that the boolean values of all according bodies are False
        as well).
        In each iteration, the function "atoms_to_assign" searches for all rules that have heads
        that need to be assigned (according to the semantic operator). All of these heads are set,
        the logic program is re-evaluated and the next iteration starts. As soon as there are
        no more rule-heads to assign, the least fixed point is found and the iteration-process
        stops. The resulting interpretation is the least model of the weak completion of the
        logic program (wcP).
        """
        found_fixed_point = False
        self.evaluate()
        itr_counter = 0
        while found_fixed_point is False:
            if print_:
                print("\n", "-" * 25, "SEMANTIC OPERATOR ITERATION", str(itr_counter), "-" * 25)
            atoms_list = self.atoms_to_assign()
            if not atoms_list:
                found_fixed_point = True
                if print_:
                    print("DONE\n")
            printed_rule_heads = []
            for atom in atoms_list:
                if atom[1] not in printed_rule_heads and print_:
                    blank_len = 42 - len(atom[0])
                    print("RULE", atom[0], blank_len * " ", "set",
                          str(atom[1]), "  to", str(atom[2]))
                atom[1].boolean_val = atom[2]
                printed_rule_heads.append(atom[1])
            self.evaluate()
            itr_counter += 1

    def get_atom_truth(self, atom_name, true_false):
        """
        Helper-function for the function "compute_conclusion".
        Given the name of an atom and a boolean value "true_false", this function extracts all
        atoms that match the input name and have the specified boolean value.
        (For instance: get_atom_truth("a", True") returns all atoms named "a", that are True,
        like "a(o1)", "a(o4)" etc.)
        """
        # Get all genereal postive / negative atoms
        atom_list = self.get_truth_atoms(true_false)

        # Filter out atoms with the wrong name.
        result_list = []
        for atom in atom_list:
            if atom.name == atom_name:
                result_list.append(atom)
        return result_list

    @staticmethod
    def same_objects(list1, list2):
        """
        Helper-function for the function "compute_conclusion".
        This function calculates how many objects are contained in the atoms of both list1 and
        list2. The amount of same objects in both lists is then returned.
        """
        same_obj = 0
        for atom in list1:
            for atom2 in list2:
                if atom.obj == atom2.obj:
                    same_obj += 1
        return same_obj

    def compute_conclusion(self, atom1, atom2, enabled_principles):
        """
        This function calculates the final conclusion, which of the 9 possible answers can be
        derived as a solution to the input problem.
        The function first calculates four list, for each input atom one list with the true
        and one with the false atoms (atoms that hold and atoms that do not hold
        in the least model of the logic program).
        Depending on these four lists and the amount of same objects between these lists,
        eight conclusions can be derived. In case none of these conclusions could be derived,
        the conclusion "NVC" (no valid conclusion) is derived.
        """
        conclusion_list = []
        pos_a = self.get_atom_truth(atom1, True)
        neg_a = self.get_atom_truth(atom1, False)
        pos_c = self.get_atom_truth(atom2, True)
        neg_c = self.get_atom_truth(atom2, False)

        # Case All a are c
        same_obj_ac = self.same_objects(pos_a, pos_c)
        if same_obj_ac == len(pos_a) and pos_a:
            conclusion_list.append("A" + atom1 + atom2)

        # Case All c are a
        if same_obj_ac == len(pos_c) and pos_c:
            conclusion_list.append("A" + atom2 + atom1)

        # Case Some c are a
        if same_obj_ac >= 1 and same_obj_ac != len(pos_c):
            if (enabled_principles[1] or enabled_principles[2]) and same_obj_ac != len(pos_a):
                conclusion_list.append("I" + atom2 + atom1)
            elif (not enabled_principles[1] and not enabled_principles[2]):
                conclusion_list.append("I" + atom2 + atom1)

        # Case Some a are c
        if same_obj_ac >= 1 and same_obj_ac != len(pos_a):
            if (enabled_principles[1] or enabled_principles[2]) and same_obj_ac != len(pos_c):
                conclusion_list.append("I" + atom1 + atom2)
            elif (not enabled_principles[1] and not enabled_principles[2]):
                conclusion_list.append("I" + atom1 + atom2)

        # Case No c are a
        same_obj_nota_c = self.same_objects(pos_c, neg_a)
        same_obj_notc_a = self.same_objects(pos_a, neg_c)
        if (same_obj_nota_c == len(pos_c) and pos_c):
            if enabled_principles[2] and (same_obj_notc_a == len(pos_a) and pos_a):
                conclusion_list.append("E" + atom2 + atom1)
            elif not enabled_principles[2]:
                conclusion_list.append("E" + atom2 + atom1)

        # Case No a are c
        if (same_obj_notc_a == len(pos_a) and pos_a):
            if enabled_principles[2] and (same_obj_nota_c == len(pos_c) and pos_c):
                conclusion_list.append("E" + atom1 + atom2)
            elif not enabled_principles[2]:
                conclusion_list.append("E" + atom1 + atom2)

        # Case Some a are not c
        if same_obj_notc_a >= 1 and same_obj_notc_a != len(pos_a):
            conclusion_list.append("O" + atom1 + atom2)

        # Case Some c are not a
        if same_obj_nota_c >= 1 and same_obj_nota_c != len(pos_c):
            conclusion_list.append("O" + atom2 + atom1)

        # If none of the above entailment holds (empty list), conclude "NVC".
        if not conclusion_list:
            conclusion_list.append("NVC")
        return conclusion_list

    @staticmethod
    def not_in(list_, atom):
        """
        Helper-function for the function "abduction".
        Given a list of atoms and a certain atom, this function checks whether the input atom
        is already contained in the list. Returns False if thats the case, otherwise True.
        """
        not_in_list = True
        for item in list_:
            if item.name == atom.name and item.obj == atom.obj:
                not_in_list = False
        return not_in_list

    def get_observations(self):
        """
        Helper-function for the function "abduction".
        This function collects and returns all rules in the logic program that can be used
        as observations for the abductive framework.
        Observations are facts, which heads are both head of a fact and head of another rule
        which is not a fact. In this case, only facts that come from the principle "existential
        import" are used as observations (according to the paper "Monadic reasoning using Weak
        Completion Semantics; A. Costa, E.A. Dietz, S. Hoelldobler; 2017").
        """
        observations = []
        for fact in self.get_facts():
            for rule in self.rules:
                # Special case for integrity constraints.
                if rule.left.type == "no_obj":
                    pass
                elif (fact.left.name == rule.left.name and fact.left.obj == rule.left.obj
                      and fact.name[0:23] == "fact_existential_import" and rule.name[:4] == "rule"
                      and fact not in observations):
                    observations.append(fact)
        return observations

    def get_abducibles(self):
        """
        Helper-function for the function "abduction".
        This function gets and returns all atoms with their according truth values, which
        can be used as abducibles in the abductive framework.
        Abducibles are facts that can be added to the logic program in order to find an explanation
        (= a set of facts out of the set of abducibles) for an observation-fact.
        All atoms, that are either undefined in the logic program (meaning there exists no rule
        in the lp which has the according atom as it´s head), or that are assumed to be False
        (meaning there exists a negative fact with the according atom as it´s head),
        are used as the heads of the abducibles. In case the atom is undefined, a positive and
        negative fact with the atom as its head are added. If the atom is assumed to be False,
        only a positive fact is added.
        """
        defined_heads = []
        for atom in self.atoms:
            for rule in self.rules:
                if (rule.left.subtype != "no_obj" and atom.name == rule.left.name
                        and atom.obj == rule.left.obj):
                    defined_heads.append(atom)
        undefined_heads = []
        for atom in self.atoms:
            if self.not_in(defined_heads, atom):
                undefined_heads.append(atom)
        abducibles = []
        for atom in undefined_heads:
            abducibles.append((atom, True))
            abducibles.append((atom, False))
        for fact in self.get_facts():
            if fact.right is False:
                abducibles.append((fact.left, True))
        return abducibles

    @staticmethod
    def get_relevant_abducibles(obs_atom, abd_list):
        """
        Helper-function for the function "abduction".
        This function filters out all abducibles that are irrelevant for the current
        processed observation (obs_atom) and returns the filtered abducibles.
        """
        rel_abd = []
        for abd in abd_list:
            if abd[0].obj == obs_atom.obj:
                rel_abd.append(abd)
        return rel_abd

    def reset_lp(self, rules):
        """
        Helper-function for the function "abduction".
        This function resets the logic program: All boolean-values of the atoms
        are set to None, all rules that should be excluded from the lp are deleted,
        and afterwards the logic program is re-evaluated. This needs to be done after
        each processed observation or explanation in order to preserve the original lp
        to avoid wrong truth-values.
        """
        for rule in rules:
            self.rules.remove(rule)
        for atom in self.atoms:
            atom.boolean_val = None
        self.evaluate()

    def abduction(self, atom1, atom2, enabled_principles):
        """
        This function implements the abduction-process (backward reasoning).
        Backward reasoning is used when people conclude "NVC" after the usual deduction
        process but then decide to search for alternative conclusions to "NVC".
        The idea is that observations, which are facts that are created from the principle
        "existential import", are deleted from the logic program in order to find an alternative
        explanation on why the atom in the observation-fact is True.
        For each observation there is a set of abducibles (positive and negative facts), which
        is used to find an explanation. If a (sub)set of the abducibles, together with the logic
        program, leads to a least model which includes the atom in the observation-fact as
        a true atom, then this (sub)set of abducibles is an explanation for the observation.
        Minimal explanations are preferred.
        After finding a minimal explanation for each observation, the function computes the
        conclusions, based on the least model for each explanation.
        In the last step, the function checks whether all explanations leaded to the same
        conclusions (in case they were not empty), only in that case, a new (credulous) conclusion
        is entailed and returned (otherwise, the function returns "NVC").
        """
        # 1. Search for facts that can be used as observations.
        observations = self.get_observations()

        # 2. Search for a minimal explanation for each observation.
        abducibles = self.get_abducibles()
        explanation_list = [[] for _ in range(len(observations))]
        for index, obs in enumerate(observations):
            relevant_abd = self.get_relevant_abducibles(obs.left, abducibles)
            self.reset_lp([obs])
            for index2, ab_ in enumerate(relevant_abd+relevant_abd):
                # If a 1-fact-explanation was already found, stop the loop.
                if explanation_list[index] and index2 >= len(abducibles):
                    break
                new_fact = self.create_fact(ab_[0], ab_[1], "fact_abduction")
                self.compute_least_model()
                if self.get_atom(obs.left.name, obs.left.obj).boolean_val == obs.right.boolean_val:
                    explanation_list[index].append(new_fact)
                # Check all explanations, which include only one fact, first.
                if index2 >= len(relevant_abd) and not explanation_list[index]:
                    for ab2 in relevant_abd:
                        if ab2 == ab_:
                            continue
                        new_fact2 = self.create_fact(ab2[0], ab2[1], "fact_abduction2")
                        self.compute_least_model()
                        if (self.get_atom(obs.left.name, obs.left.obj).boolean_val
                                == obs.right.boolean_val):
                            explanation_list[index].append(new_fact)
                            explanation_list[index].append(new_fact2)
                        self.reset_lp([new_fact2])
                self.reset_lp([new_fact])
            self.rules.append(obs)
            self.evaluate()
        explanation_list = [x for x in explanation_list if x != []] # filter our empty explanations

        # 3. Compute the least model and corresponding conclusion for each explanation.
        conclusion_list = [[] for x in range(len(observations))]
        for index, expl in enumerate(explanation_list):
            for rule in expl:
                self.rules.append(rule)
            self.compute_least_model()
            result = self.compute_conclusion(atom1, atom2, enabled_principles)
            conclusion_list[index].append(result)
            self.reset_lp(expl) # Erase current explanation from lp to compute the next conclusion

        # 4. The final result is the intersection of all computed conclusions. If empty: Return NVC.
        conclusion_empty = True
        for expl in conclusion_list:
            if expl:
                conclusion_empty = False
        if conclusion_empty:
            return ["NVC"]
        for result in conclusion_list:
            if result != conclusion_list[0]:
                return ["NVC"]
        return conclusion_list[0][0]



# ------ FUNCTIONS TO COMPUTE ALL 64 SYLLOGISM SOLUTIONS FOR SELECTED PRINCIPLE COMBINATIONS -------

def compute_all_variations():
    """
    Function calculates all possible combinations of the five advanced, enabled principles.
    Afterwards, the result of the 64 syllogistic problems for each of the combinations is
    calculated. All results for all combinations are stored in a csv-file, to later look up
    values when predicting human responses (since the calculation of the logic program is
    too expensive).
    """
    all_variations = []
    ass = [0, 1]
    for a__ in ass:
        for b__ in ass:
            for c__ in ass:
                for d__ in ass:
                    for e__ in ass:
                        all_variations.append([a__, b__, c__, d__, e__])

    model = WCSSyllogistic()
    model.print_output = False

    id_list = []
    result_list = []

    for var in all_variations:
        for _, problem in enumerate(PROBLEM_LIST):
            id_list.append(problem+str(var[0])+str(var[1])+str(var[2])+str(var[3])+str(var[4]))
            result = model.compute_problem_from_scratch(problem, var)
            result_list.append(result)

    df_ = pd.DataFrame({"problem" : id_list, "result" : result_list})
    df_.to_csv("all_variations.csv")


def compute_one_trial(enabled_principles, name):
    """
    Function calculates one trial of all 64 syllogistic problems for a given assignment of
    enabled principles. The results are saved in a csv-file with the specified input name.
    """
    model = WCSSyllogistic()
    model.print_output = False

    id_list = []
    result_list = []
    for _, problem in enumerate(PROBLEM_LIST):
        id_list.append(problem)
        result = model.compute_problem_from_scratch(problem, enabled_principles)
        result_list.append(result)

    df_ = pd.DataFrame({"problem" : id_list, "result" : result_list})
    df_.to_csv(str(name)+".csv")



PROBLEM_LIST = ["AA1", "AA2", "AA3", "AA4", "AI1", "AI2", "AI3", "AI4",
                "AE1", "AE2", "AE3", "AE4", "AO1", "AO2", "AO3", "AO4",
                "IA1", "IA2", "IA3", "IA4", "II1", "II2", "II3", "II4",
                "IE1", "IE2", "IE3", "IE4", "IO1", "IO2", "IO3", "IO4",
                "EA1", "EA2", "EA3", "EA4", "EI1", "EI2", "EI3", "EI4",
                "EE1", "EE2", "EE3", "EE4", "EO1", "EO2", "EO3", "EO4",
                "OA1", "OA2", "OA3", "OA4", "OI1", "OI2", "OI3", "OI4",
                "OE1", "OE2", "OE3", "OE4", "OO1", "OO2", "OO3", "OO4"]


INVALID_SYLLOGISMS = ["AA3", "AI1", "AI3", "AO1", "AO2", "IA2", "IA3", "II1", "II2",
                      "II3", "II4", "IO1", "IO2", "IO3", "IO4", "EE1", "EE2", "EE3",
                      "EE4", "EO1", "EO2", "EO3", "EO4", "OA1", "OA2", "OI1", "OI2",
                      "OI3", "OI4", "OE1", "OE2", "OE3", "OE4", "OO1", "OO2", "OO3",
                      "OO4"]



# --------------------------------------------- MAIN -----------------------------------------------

def main():
    """
    The WCS Model can be either run as a CCOBRA-Model in a predefined benchmark to test
    the performance compared to other syllogistic CCOBRA-Models, or separately to check
    the WCS answers to different principle combinations.

    1. To calculate the solutions to the 64 syllogisms for one specified principle assignment
       (f. i. [0, 0, 0, 0, 1]), run the following function. The results of the computation
       are saved as a csv file in the folder, in which this python file is located.
       - compute_one_trial([0, 0, 0, 0, 1], "csv_table_name")

    2. To calculate the solutions to the 64 syllogisms for all possible principle combinations
       (32 in total), run the following function (this needs to be done before using the WCS-model
       in a CCOBRA benchmark, so that the answers can be looked up in the csv-file).
       - compute_all_variations()

    3. To see what the WCS model is calculating in detail for one given problem and principle
       assignment, run the following (here: problem "AA1" and principles [1, 1, 1, 1, 1]):
       - model = WCSSyllogistic()
       - model.print_output = True
       - model.compute_problem_from_scratch("AA1", [1, 1, 1, 1, 1])

    Some in the most recent paper (Reasoning Principles and Heuristic Strategies in Modeling Human
    Clusters; Dietz, Hoelldobler, Moerbitz; 2017) used, promising principle combinations are:
    1. Basic + converse(I) + abduction --> [0, 1, 0, 0, 1]
    2. Basic + converse(I) + Deliberate generalization --> [1, 1, 0, 0, 0]
    3. Basic + converse(I, E) + Contraposition (A) (contraposition needs the context operator)
       --> [1, 1, 1, 1, 0]
    """

    #compute_all_variations()
    model = WCSSyllogistic()
    model.print_output = True
    model.compute_problem_from_scratch("AA1", [1, 1, 1, 1, 1])



if __name__ == "__main__":
    main()
