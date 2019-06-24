"""
Weak-Completion-Semantics model for the Wason Selection Task.

Created on 22.11.2018

@author: Julia Mertesdorf <julia.mertesdorf@gmail.com>
"""

import random
import scipy.optimize
import numpy



#------------------------------------------ SCORING RULES ------------------------------------------

def calc_rmse(parameters, model, target_data):
    """
    Calculate Root Mean Squared Error between parameterized model and target data.
    """
    predictions = model(*parameters)
    error = numpy.sqrt(numpy.mean((numpy.array(predictions) - numpy.array(target_data))**2))
    return error



#------------------------------------------ MODEL CLASS --------------------------------------------

class WCSWason():
    """
    This class implements the Weak-Completion-Semantics model for the Wason Selection Task.
    The class incorporates functions to set up and run one trial of the Wason Selection Task, which
    uses the internal logic program and the Weak Completion Semantics framework to solve it,
    as well as a function to run it for "num"-times.
    """
    probability_guess = [0.5, 0.5] # initial guess for the principle-probabilities.

    def __init__(self):
        """
        The model class contains a logic program-object, which is used to do all calculation
        processes to find a solution for the Wason Selection Task.
        With the variable "print_output" enabled, the problem solving process is printed
        for visualization (prints the logic program, as well as the truth-assignments of the atoms).
        The list "enabled_principles" determines, which additional principles are used when
        solving an instance of the Wason Selection Task. The first boolean value encodes, whether
        abduction should be used, and the second whether an additional rule ("-D <-- 7") should
        be added to the logic program for advanced reasoning.
        The list "cards" contains all cards that are used in one trial of the Wason Selection Task.
        The dictionary "result_pattern_dict" saves the result for each computed trial of the Wason
        Selection Task with the according set boolean-values for the additional principles.
        Since the result, given the boolean values for the principles, is always the same for the
        same pattern, the program only has to calculate each pattern once, which is more efficient
        than calculating the logic program again for every new call.
        """
        self.lp_ = []
        self.print_output = False
        self.enabled_principles = [0, 0]
        self.principle_probabilities = self.probability_guess
        self.cards = ["D", "K", "3", "7"]
        self.result_pattern_dict = {}

    def set_enabled_principles(self):
        """
        Function sets truth values for the additional principles to be True and enabled,
        depending on a given input probability list and a random number.
        """
        for index, value in enumerate(self.principle_probabilities):
            self.enabled_principles[index] = value > random.random()

    def create_program(self):
        """
        This function creates the base logic program which is used to solve the decision,
        whether a given card should be turned or not (the observed card is added in another step).
        The logic program, in case the second additional pricinples is not enabled, is:
        ---------------------------
        3   <-- D and not(ab1)
        ab1 <-- False
        ---------------------------
        In case the second additional principle is enabled, more rules for advanced reasoning
        are added. The logic program is:
        ---------------------------
        3   <-- D and not(ab1)
        D`  <-- 7 and not(ab2)
        D   <-- not(D`)
        ab1 <-- False
        ab2 <-- False
        ---------------------------
        """
        # Create necessary atoms and operators for the main rule and fact.
        atom_d = self.lp_.create_atom("D")
        atom_3 = self.lp_.create_atom("3")
        ab_main = self.lp_.create_atom("ab1")
        not_ab_main = OperatorNot(ab_main)
        and_main = OperatorAnd(atom_d, not_ab_main, "and")

        # Create main rule: "3 <-- D and not(ab1)" and the according abnormality fact.
        self.lp_.create_rule("rule_main", atom_3, and_main)
        self.lp_.create_fact(ab_main, False)

        # Create additional rule: "D` <-- 7 and not(ab2)"; "D <-- not(D`)" (encodes modus tollens)
        if self.enabled_principles[1]:
            neg_atom_d = self.lp_.create_atom("D`")
            atom_7 = self.lp_.create_atom("7")
            ab_adv = self.lp_.create_atom("ab2")
            not_neg_d = OperatorNot(neg_atom_d)
            not_ab_adv = OperatorNot(ab_adv)
            and_adv = OperatorAnd(atom_7, not_ab_adv, "and")
            self.lp_.create_rule("rule_mt", neg_atom_d, and_adv)
            self.lp_.create_rule("rule_mt2", atom_d, not_neg_d)
            self.lp_.create_fact(ab_adv, False)

    def add_card(self, card):
        """
        This function adds a given input-card as a positive fact to the logic program
        and returns it (in order to later delete it from the logic program before
        observing the next card).
        """
        card_atom = self.lp_.create_atom(card)
        card_fact = self.lp_.create_fact(card_atom, True)
        return card_fact

    def compute_one_trial(self):
        """
        This function computes one trial of the Wason Selection Task and returns the pattern
        which encodes the turned cards in this trial ([p, -p, q, -q] or [D, K, 3, 7]).

        The function first sets the boolean values for the additional enabled principles,
        given the internal probabilities ("self.principle_probabilities").
        Before creating and calculating the logic program to solve the task, the function checks
        whether there already exists a solution, given the enabled additional principles.
        If thats the case, the solution is directly returned.
        Otherwise, the following procedure is executed:
        First, the basic logic program, which is used for all cards, is created. For each card,
        the function adds the card as a positive fact to the logic program, computes the least
        model, calls a function to decide whether to turn the currently observed card or not and
        then deletes the card-fact from the logic program before processing the next card.
        If "print_output" is enabled, the problem solving process for every card is visualized.
        (Note: The result of the function depends on the current probabilities for the
        additional principles to be enabled, given in "self.principle_probabilities")
        """
        self.set_enabled_principles()
        result = self.search_pattern_dict(self.enabled_principles)
        if result != None:
            return result
        self.lp_ = LogicProgram()
        self.create_program()
        turn_pattern = [0, 0, 0, 0]
        for index, card in enumerate(self.cards):
            if self.print_output:
                print("\n==============================", card, self.enabled_principles,
                      "=================================")
            card_fact = self.add_card(card)
            if self.print_output:
                self.lp_.print_lp_truth("CREATION")

            self.lp_.compute_least_model()
            if self.print_output:
                self.lp_.print_lp_truth("LEAST MODEL")
                self.lp_.print_atoms()

            turn = self.lp_.decide_turn()
            if self.enabled_principles[0]:
                abd_turn = self.lp_.abduction(self.print_output)
                if abd_turn:
                    turn = abd_turn
            turn_pattern[index] += turn
            self.lp_.reset_lp([card_fact])
        self.update_pattern_dict(self.enabled_principles, turn_pattern)
        if self.print_output:
            print("\nTurn-pattern is [D, K, 3, 7]:", turn_pattern, "\n\n")
        return turn_pattern

    def compute_every_variation_once(self):
        """
        This function exists due to test purposes, to show how every pattern (p/ pq/ pq-q/ p-q)
        can be obtained by setting the enabled-principles accordingly.
        """
        blank = " "
        output_str = "\nEnabled Principles:" + blank * 13 + "Result:\n"
        output_str = output_str + "[Abduction, Additional Rule]" + blank * 3 + "[D, K, 3, 7]\n\n"
        self.principle_probabilities = [0, 0]
        solution = self.compute_one_trial()
        output_str = output_str + str(self.enabled_principles) + blank * 17 + str(solution) + "\n"
        self.principle_probabilities = [1, 0]
        solution = self.compute_one_trial()
        output_str = output_str + str(self.enabled_principles) + blank * 18 + str(solution) + "\n"
        self.principle_probabilities = [0, 1]
        solution = self.compute_one_trial()
        output_str = output_str + str(self.enabled_principles) + blank * 18 + str(solution) + "\n"
        self.principle_probabilities = [1, 1]
        solution = self.compute_one_trial()
        output_str = output_str + str(self.enabled_principles) + blank * 19 + str(solution) + "\n"
        print(output_str)

    def compute_case_xtimes(self, case, num, manual_probs=None):
        """
        This function computes the Wason Selection Task with type "case" (abstract/ social/ deontic)
        "num" times, specified by the two input parameters.
        If instead of a string encoding a specific case-type, the string "manual" is given, as well
        as an additional list "manual_probs", this list is set as the principle_probabilities.
        The result of the function is the percentage, how often each pattern (p/ pq/ pq-q/ p-q)
        was computed, after running the Wason Selection Task "num"-times.
        (Note: the probability parameters specified for the three case-types were calculated
        by the function "optimize")
        """
        turn_pattern = [0, 0, 0, 0]
        if case == "abstract":
            self.principle_probabilities = [0.475, 0.256]
        elif case == "everyday":
            self.principle_probabilities = [0.49, 0.424]
        elif case == "deontic":
            self.principle_probabilities = [0.151, 0.785]
        elif case == "manual" and manual_probs is not None:
            self.principle_probabilities = manual_probs
        for _ in range(0, num):
            current_turn_pattern = self.compute_one_trial()
            turn_pattern = self.add_turn_counter(turn_pattern, current_turn_pattern)
        for index, _ in enumerate(turn_pattern):
            turn_pattern[index] = round((turn_pattern[index] / num) * 100)
        print("Percentages after", num, "runs:", turn_pattern)
        return turn_pattern

    @staticmethod
    def add_turn_counter(old_counter, turned_cards):
        """
        Given the old counter for the amount of turned cards-patterns so far, the function
        adds the new pattern ("turned_cards") to the old pattern counter and returns it
        (the turned-cards-pattern is: [p, pq, pq-q, p-q]).
        """
        if turned_cards == [1, 0, 0, 0]:     # case p
            old_counter[0] += 1
        elif turned_cards == [1, 0, 1, 0]:   # case pq
            old_counter[1] += 1
        elif turned_cards == [1, 0, 1, 1]:   # case pq-q
            old_counter[2] += 1
        elif turned_cards == [1, 0, 0, 1]:   # case p-q
            old_counter[3] += 1
        return old_counter

    def search_pattern_dict(self, prin_list):
        """
        This function constructs a specific key-string, containing the problem and the boolean
        values of the two additional principles (encoding whether they are enabled or not).
        Afterwards, the function checks whether this key-string is already contained in the
        result_pattern-dictionary (meaning this pattern was calculated before).
        If thats the case, the function will return the result for this pattern,
        otherwise it returns None.
        """
        key_str = "wason" + str(int(prin_list[0])) + str(int(prin_list[1]))
        result = self.result_pattern_dict.get(key_str)
        return result

    def update_pattern_dict(self, prin_list, return_val):
        """
        Likewise to the function "search_pattern_dict", this function constructs a key-string,
        containing the problem and the boolean values of the two additional principles.
        Afterwards, the function adds a new dictionary-entry to the dictionary, taking "key-str"
        as the key and the input-parameter "return_val", which was calculated by the procedure
        in the function "compute_one_trial", as solution for this pattern.
        """
        key_str = "wason" + str(int(prin_list[0])) + str(int(prin_list[1]))
        self.result_pattern_dict[key_str] = return_val

    def optimize_run_10000times(self, prin1_prob, prin2_prob):
        """
        This function is a helper function for the function "optimize".
        The function sets the given input probabilities as the internal probabilities (which
        encode whether the additional principles are enabled). Then, "compute_one_trial" is called
        10.000 times, to get a high resolution and accurate prediction for the distribution of the
        four canonical cases p, pq, pq-q and p-q with regards to the input probabilities.
        The normalized distribution is returned.
        """
        self.principle_probabilities = [prin1_prob, prin2_prob]
        turn_pattern = [0, 0, 0, 0]
        for _ in range(0, 10000):
            current_turn_pattern = self.compute_one_trial()
            turn_pattern = self.add_turn_counter(turn_pattern, current_turn_pattern)
        for index, _ in enumerate(turn_pattern):
            turn_pattern[index] = round(turn_pattern[index] / 100)
        return turn_pattern

    def optimize(self, target_data, loss_function=calc_rmse):
        """
        This function optimizes the probability-parameters to get results, that
        match the input target_data as good as possible. The resulting best values
        are set as the new principle_probabilities.
        The method "Cobyla" is used for optimization, since it is one of the few methods that
        can handle non-deterministic, probability-depending functions.
        (Note: Since the model is non-deterministic, the resulting best values for the
        principle probabilities can vary (the extent of variance depends on the target data))
        """
        # Since the optimization-method "Cobyla" doesn´t support bounds, formulate
        # them as inequality constraints.
        self.principle_probabilities = self.probability_guess
        cons = []
        lower_bound = {"type": "ineq", "fun": lambda x: x}   # x > 0
        upper_bound = {"type": "ineq", "fun": lambda x: 1-x} # x < 1
        cons.append(lower_bound)
        cons.append(upper_bound)

        res = scipy.optimize.minimize(loss_function, self.principle_probabilities,
                                      method="Cobyla",
                                      constraints=cons,
                                      options={'disp': False},
                                      args=(self.optimize_run_10000times,
                                            target_data))
        self.principle_probabilities = res.x
        print("Optimized probabilities:", self.principle_probabilities)
        return self.principle_probabilities

    def average_results(self, case, num):
        """
        This function calculates and returns the average result (probabilities for each of the four
        canonical cases) of running a specific case "num" times.
        For the given input case, the function optimizes the probability-parameters with regards
        to the input case and after that, calls "compute_case_xtimes" to run the specific case
        10.000 times in order to get high resolution-results.
        These two steps (parameter optimization + 10.000 runs with the calculated
        probability-values) are executed "num" times and afterwards divided by "num" to obtain
        the average result over all runs.
        """
        if case == "abstract":
            target_data = [36, 39, 5, 19]
        elif case == "everyday":
            target_data = [23, 37, 11, 29]
        elif case == "deontic":
            target_data = [13, 19, 4, 64]
        average_result = [0, 0, 0, 0]
        for _ in range(0, num):
            prob = self.optimize(target_data)
            result = self.compute_case_xtimes("manual", 10000, prob)
            average_result = [sum(_) for _ in zip(average_result, result)]
        for index, _ in enumerate(average_result):
            average_result[index] = round(average_result[index] / num)
        print("Average result:", average_result)
        return average_result



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


class Atom(Formula):
    """
    This class implements basic atoms. Atoms have a name and a boolean value.
    All atoms have an evaluation-function to calculate their current boolean value.
    """
    def __init__(self, name, bool_val=None):
        super().__init__()
        self.name = name
        self.type = "atom"
        self.boolean_val = bool_val

    def evaluate(self):
        """
        Function for evaluating the own current boolean value.
        In case the atom was instantiated with one of the boolean values "True",
        "False" or "None" (= Unknown) as the name, the boolean value is set accordingly.
        """
        if (self.name is True or self.name is False or self.name is None):
            self.boolean_val = self.name
        return self.boolean_val

    def __str__(self):
        return str(self.name)


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

    def __str__(self):
        return self.type + "(" + str(self.content) + ")"


class OperatorBijective(Formula):
    """
    This class implements the parent class of bijective logic operators
    (for logic operator like "and", and "implication").
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



# ----------------------------------------- LOGIC PROGRAM CLASS ------------------------------------

class LogicProgram():
    """
    This class implements logic programs, which contain rules and positive / negative facts
    (to encode the truth-value of atoms).
    Logic Programs are the basic framework used in the Weak-Completion-Semantics.
    The LogicProgram-class contains important functions which encode the main
    concepts used in Weak-Completion-Semantics, like the implementation of the semantic
    operator in "compute_least_model", abduction, as well as many helper-functions.
    """
    def __init__(self):
        """
        Initializes a LocicProgram-object.
        A logic program contains a list of atoms and rules (implications).
        """
        self.atoms = []
        self.rules = []

    def create_atom(self, atom_name):
        """
        Function creates an atom with the specified name and adds it to the atoms-list of the
        logic program. In case the atom already exists, the function returns this atom.
        """
        for i in range(0, len(self.atoms)):
            if self.atoms[i].name == atom_name:
                return self.atoms[i]
        at_ = Atom(atom_name)
        self.atoms.append(at_)
        return at_

    def get_atom(self, atom_name):
        """
        Function returns the atom which matches the specified name.
        """
        for i in range(0, len(self.atoms)):
            if self.atoms[i].name == atom_name:
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

    def create_rule(self, name, consequent, conditional):
        """
        This function creates a rule (implication), given an input name, a conditional and
        a consequent, and adds the rule to the logic program.
        """
        rule = OperatorImplication(consequent, conditional, name)
        self.rules.append(rule)
        return rule

    def create_fact(self, consequent, true_false, name="fact"):
        """
        Given a truth-value "true_false" and a consequent, this function creates a fact
        of the form "consequent <-- true_false" and adds it to the logic program.
        """
        atom = Atom(true_false)
        fact = OperatorImplication(consequent, atom, name)
        self.rules.append(fact)
        return fact

    def print_atoms(self):
        """
        Function prints all positive, negative and unknown atoms of the logic program.
        """
        true_atoms = self.get_truth_atoms(True)
        false_atoms = self.get_truth_atoms(False)
        unknown_atoms = self.get_truth_atoms(None)
        pos = ([str(at) for at in true_atoms])
        neg = ([str(at) for at in false_atoms])
        un_ = ([str(at) for at in unknown_atoms])
        print("Positive Atoms:", pos, "\nNegative Atoms:", neg, "\nUnknown Atoms: ", un_)

    def print_lp_truth(self, state):
        """
        Function prints all rules in the logic program with their according boolean values.
        """
        print("------------------------------ AFTER", str(state), "-------------------------------")
        for rule in self.rules:
            string = rule.name
            blank = 18 - len(string)
            string = string + blank * " " + str(rule)
            blank2 = 60 - len(string)
            string = (string + blank2 * " " + str(rule.boolean_val) + " <-- "
                      + str(rule.left.boolean_val) + " " + str(rule.right.boolean_val))
            print(string)
        print("--------------------------------------------------------------------------------")

    def evaluate(self):
        """
        Function evaluates all rules in the logic program by calling their evaluate-function.
        Since the logic-operators are implemented inductively, the evaluation-processes heads down
        from bijective-operators to monotonic operators to the atoms with their
        true/false/unknown-assignments.
        """
        for rule in self.rules:
            rule.evaluate()

    def same_head(self, rule_head):
        """
        This function searches for all rules in the logic program which have the same head
        as "rule_head". The function then checks whether these bodies all have the boolean value
        "False" (this is a necessary condition to set a head False with the semantic operator).
        If thats the case, return True, else return False.
        """
        result = True
        for rule in self.rules:
            if ((rule.right.boolean_val != False) and (rule.left.name == rule_head.name)):
                result = False
        return result

    def atoms_to_assign(self):
        """
        Helper-function for "compute_least_model".
        Check whether the semantic operator is done or has still some atoms to assign.
        In the first case, return an empty list. Otherwise return a list with all rules
        that still need to be assigned.
        """
        atoms_to_assign = []
        for rule in self.rules:
            if rule.right.boolean_val != rule.left.boolean_val:
                if rule.right.boolean_val is True:
                    atoms_to_assign.append([rule.name, rule.left, True])
                elif rule.right.boolean_val is False and self.same_head(rule.left):
                    atoms_to_assign.append([rule.name, rule.left, False])
        return atoms_to_assign

    def compute_least_model(self):
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
        while found_fixed_point is False:
            atoms_list = self.atoms_to_assign()
            if not atoms_list:
                found_fixed_point = True
            for atom in atoms_list:
                atom[1].boolean_val = atom[2]
            self.evaluate()

    def get_observations(self):
        """
        Helper-function for the function "abduction".
        This function collects and returns all rules in the logic program that can
        be used as observations for the abductive framework.
        Observations are facts, which heads are both head of a fact and head of another rule
        which is not a fact.
        """
        observations = []
        for fact in self.get_facts():
            for rule in self.rules:
                if (rule != fact and fact.left.name == rule.left.name and rule.name[:4] == "rule"):
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
        which has the according atom as it´s head), or that are assumed to be False (meaning there
        exists a negative fact with the according atom as it´s head), are used as the heads of
        the abducibles. In case the atom is undefined, a positive and negative fact with the atom
        as its head are added. If the atom is assumed to be False, only a positive fact is added.
        """
        defined_heads = []
        for atom in self.atoms:
            for rule in self.rules:
                if atom.name == rule.left.name:
                    defined_heads.append(atom)
        undefined_heads = []
        for atom in self.atoms:
            if atom not in defined_heads:
                undefined_heads.append(atom)
        abducibles = []
        for atom in undefined_heads:
            abducibles.append((atom, True))
            abducibles.append((atom, False))
        for fact in self.get_facts():
            if fact.left.boolean_val is False:
                abducibles.append((fact.left, True))
        return abducibles

    def reset_lp(self, rules):
        """
        Helper-function for the function "abduction".
        This function resets the logic program: All boolean-values of the atoms are set to None,
        all rules that should be excluded from the logic program are deleted, and afterwards
        the logic program is re-evaluated.
        This needs to be done after each processed observation or explanation in order to preserve
        the original logic program to avoid wrong truth-values.
        """
        for rule in rules:
            self.rules.remove(rule)
        for atom in self.atoms:
            atom.boolean_val = None
        self.evaluate()

    def abduction(self, print_):
        """
        This function implements the abduction-process (backward reasoning).
        The idea is that observations, which are facts that encode the current observed card (only
        if the card-atom is also head of another rule in the logic program, which is not a fact),
        are deleted from the logic program in order to find an alternative explanation on why
        the atom in the observation-fact is True.
        For an observation there is a set of abducibles (positive and negative facts), which
        is used to find an explanation. If a (sub)set of the abducibles, together with the logic
        program, leads to a least model which includes the atom in the observation-fact as
        a true atom, then this (sub)set of abducibles is an explanation for the observation.
        Minimal explanations are preferred.
        After a minimal explanation for the observation is found, the function calls "decide_turn"
        again in order to determine whether the card should be turned, considering the new
        information as a result of the abduction process.
        """
        # 1. Search for facts that can be used as observations.
        observations = self.get_observations()
        if not observations:
            return None
        observation = observations[0] # There should be only one observation, so take first element.

        # 2. Search for a minimal explanation for the observation.
        abducibles = self.get_abducibles()
        self.reset_lp([observation])
        explanation = []
        for index, ab1 in enumerate(abducibles+abducibles):
            # If a 1-fact-explanation was already found, stop the loop.
            if index >= len(abducibles):
                break
            new_fact = self.create_fact(ab1[0], ab1[1])
            self.compute_least_model()
            if self.get_atom(observation.left.name).boolean_val == observation.right.boolean_val:
                explanation.append(new_fact)
            # Check all explanations, which include only one fact, first.
            if index >= len(abducibles) and not explanation:
                for ab2 in abducibles:
                    if ab2 == ab1:
                        continue
                    new_fact2 = self.create_fact(ab2[0], ab2[1])
                    self.compute_least_model()
                    if self.get_atom(observation.left).boolean_val == observation.right.boolean_val:
                        explanation.append(new_fact)
                        explanation.append(new_fact2)
                    self.reset_lp([new_fact2])
            self.reset_lp([new_fact])
        self.rules.append(observation)
        self.evaluate()
        if print_:
            print("\nExplanation:", [str(exp) for exp in explanation], "\n")

        # 3. Compute the least model and corresponding conclusion for the explanation.
        for rule in explanation:
            self.rules.append(rule)
        self.compute_least_model()
        if print_:
            self.print_lp_truth("ABDUCTION")
            self.print_atoms()
        result = self.decide_turn()
        self.reset_lp(explanation)
        return result

    def decide_turn(self):
        """
        This function decides, regarding the truth-assignments of the two atoms in the main rule
        ("D" and "3"), whether the currently observed card should be turned or not.
        Basically, a card should be turned if the given main rule "3 <-- D" evaluates to True,
        given the truth assignments of "D" and "3" after computing the least model (and abduction).
        The idea is, that in these cases, the main rule needs to hold, so the person has to turn
        the card in order to verify if the main rule actually holds.
        A two-valued evaluation is used, as in case both atoms are unknown, the card shouldn`t be
        turned, since one cannot conclude anything with the current card about the two atoms, if
        both atoms are unknown (if a three-valued (Lukasiewicz) evaluation would be used, then
        almost every card would have to be turned in most cases, since U <-- U evaluates to True).
        """
        atom_d = self.get_atom("D")
        atom_3 = self.get_atom("3")
        if atom_3.boolean_val is True and atom_d.boolean_val is True:
            return 1
        elif atom_3.boolean_val is False and atom_d.boolean_val is False:
            return 1
        elif atom_3.boolean_val is True and atom_d.boolean_val is False:
            return 1
        return 0



# ---------------------------------------------- MAIN ----------------------------------------------

def main():
    """
    Usage of the WCS-model for the Wason Selection Task:
    1) To try out one run with manually set principle_probabilities:
    - model.principle_probabilities = [0.5, 0.5]
    - model.compute_one_trial()

    2) To see how every pattern of the four canonical cases is derived:
    - model.compute_every_variation_once()

    3) To compute a certain case-type for a variabel number of times (result is in percentage):
    - model.compute_case_xtimes("abstract", 10000)

    4) To optimize the probabilities for the additional principles for certain target-data
       (returns the optimal probability-values for the given target-data):
    - target_abstract = [36, 39, 5, 19]
    - target_everyday = [23, 37, 11, 29]
    - target_deontic = [13, 19, 4, 64]
    - model.optimize(target_deontic)

    5) To calculate the average result for a specific case over many runs (result in percentage):
    - model.average_results("abstract", 50)

    If the problem solving process should be printed:
    - model.print_output = True
    """
    model = WCSWason()
    model.print_output = True
    model.compute_every_variation_once()


    # Note:
    # The optimal parameters to match the empirical results of the wason selection task are:
    # - abstract: [0.44-0.51, 0.22-0.29] --> average: [0.475, 0.256]
    # - everyday: [0.42-0.56, 0.37-0.47] --> average: [0.49,  0.424]
    # - deontic:  [0.12-0.18, 0.75-0.82] --> average: [0.151, 0.785]
    #
    # Average result (~50 * 10.000 runs):
    # - abstract: [39, 37, 12, 13]
    # - everyday: [29, 31, 21, 20]
    # - deontic: [19, 4, 12, 65]


if __name__ == "__main__":
    main()
