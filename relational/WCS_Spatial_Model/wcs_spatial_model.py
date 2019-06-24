"""
Weak-Completion-Semantics model for spatial reasoning.

Created on 28.10.2018

@author: Julia Mertesdorf<julia.mertesdorf@gmail.com>
"""

from operator import attrgetter
from itertools import permutations
import ccobra



#------------------------------------------ MODEL CLASS --------------------------------------------

class WCSSpatial(ccobra.CCobraModel):
    """
    This class implements the Weak-Completion-Semantics approach for solving spatial
    reasoning tasks.
    The function "compute_problem" solves a single spatial task by creating a logic program
    founded on the input premises and reason based on the least model of the according
    logic program. The class "LogicProgram" is a helper-class which contains all functions
    to create and compute logic programs.
    """

    def __init__(self, name="WCSSpatial"):
        super(WCSSpatial, self).__init__(name, ['spatial-relational'], ['verify', 'single-choice'])
        self.lp_ = LogicProgram()        # Internal logic program of the model
        self.print_output = False        # Print the problem solving process
        self.perform_variation = False
        self.variation_search_depth = 3

    def predict(self, item, **kwargs):
        """
        Predict and return a response for a given item.
        """
        if item.response_type == 'verify':
            result = self.compute_problem(item.task, item.choices[0][0])
            return result

    def compute_problem(self, problem, query):
        """
        This function computes a spatial problem consisting of several premises and a query.
        The function first creates a logic program which comprises rules encoding the computation
        process to obtain the preferred mental model. After that, all problem premises are added
        to the program. Then the least model of the program is computed, resulting in the preferred
        mental model (model construction phase). This model entails whether the problem query holds
        or does not hold, by searching for the relation encoding this query with the help of the
        function "compute_conclusion" (model inspection phase).
        If the query holds in the preferred model and variation is enabled (via the boolean
        "perform_variation"), the third and last phase (model variation phase) is processed,
        which may lead to alternative models and conclusions.
        Finally, a dictionary of all valid models with their according swap distance (from the
        preferred model) and the answer to the query is printed. Based on this dictionary, a final
        conclusion can be drawn - if the query relation does not hold in at least one of the
        resulting models, then the query is rejected.
        """
        model_dict = {} # Dictionary containing each model (string) with the minimal swap distance
                        # and the answer, whether the query relation holds in the model or not.

        # 1. Create logic program (LP), collect all objects and create the initial rules for LP.
        self.lp_ = LogicProgram()
        objects = self.collect_objects(problem)
        phases = len(problem)
        self.create_program(objects, phases)
        self.add_premise_requests(problem)

        if self.print_output:
            string = "=" * 100
            print(string, "\n\nSpatial problem:\n", problem, "\n")
            self.lp_.print_lp_truth("PREFERRED MODEL")

        # 2. Compute initial (preferred) model & conclusion and add it to the dictionary of models.
        self.lp_.compute_least_model(self.print_output)
        if self.print_output:
            self.lp_.print_atoms()
        result = self.lp_.compute_conclusion(query, self.print_output)
        if self.print_output:
            print("The preferred model is:")
        preferred_model = self.visualize_model(phases, self.print_output)
        model_dict[preferred_model] = [0, result]

        # 3. If applicable, start the variation phase and search for alternative models.
        if result is True and self.perform_variation:
            relevant_atoms = self.extract_relevant_atoms(phases)
            self.setup_variation_phase_program(relevant_atoms, objects, phases)
            amb_list = self.get_ambiguous_combinations(relevant_atoms[1])
            if self.print_output:
                print("\n\nPermutations:")
                for sublist in amb_list:
                    print([str(at) for at in sublist])
                print("\nLength of permutation list: ", len(amb_list))

            # Construct one alternative model per iteration (amb-combination), if possible.
            for _, subset in enumerate(amb_list):
                if self.variation_search_depth < len(subset):
                    break
                current_additional_facts = []
                for item in subset:
                    rule = self.lp_.create_fact(item, True, "15_Object_swap_request")
                    adjust_facts = self.add_adjust_facts(item, phases)
                    current_additional_facts.append(rule)
                    current_additional_facts += adjust_facts

                if self.print_output:
                    print("\nCurrent permutation:", [str(at) for at in subset])
                    self.lp_.print_lp_truth("MODEL-VARIATION")
                self.lp_.compute_least_model(self.print_output) # Try to compute alternative model
                if self.print_output:
                    self.lp_.print_atoms()

                # Check if the model is complete, only if that´s the case calculate a conclusion.
                if self.model_complete():
                    result = self.lp_.compute_conclusion(query, self.print_output)
                    if self.print_output:
                        print("Alternative model is:")
                    model = self.visualize_model(phases+len(relevant_atoms[1]), self.print_output)
                    if model not in model_dict:
                        model_dict[model] = [len(subset), result]

                # Reset the variation program in order to process the next amb-combination.
                self.reset_lp_for_variation(current_additional_facts)

        # 4. Determine the final conclusion.
        final_result = True
        for val in model_dict.values():
            if val[1] is False:
                final_result = False

        if self.print_output:
            print("\n", "-" * 100, "\n\nAll models:  ", ', '.join(model_dict))
            print("\nDictionary - All models with minimal transformation distance and",
                  "according response:\n", model_dict, "\n")
            print("Final Conclusion: Does the relation hold? - ", final_result)

        return final_result



#------------------------------------------ MODEL CONSTRUCTION -------------------------------------

    @staticmethod
    def collect_objects(problem):
        """
        This function collects all objects that are mentioned in the premises of the problem
        and returns the according list of objects.
        """
        objects = []
        for premise in problem:
            if premise[1] not in objects:
                objects.append(premise[1])
            if premise[2] not in objects:
                objects.append(premise[2])
        return objects

    def add_premise_requests(self, problem):
        """
        This function adds the premises of the problem in form of placement requests
        "l(o1, o2, i)" to the logic program.
        Moreover, the program adds the same placement requests a second time as initial requests
        "il(o1, o2, i)" to the logic program to signalize that these were the original premise
        requests. They are not needed for the construction of the preferred mental model, but later
        for the construction of alternative models in the variation phase. These initial placement
        requests will then be used as constraints to prevent the implementation from constructing
        models that would violate the premises of the problem.
        """
        for index, premise in enumerate(problem):
            if premise[0] == "Left" or premise[0] == "left":
                rel = self.lp_.create_atom("l", premise[1], premise[2], index+1, "atom_ob2")
                rel2 = self.lp_.create_atom("il", premise[1], premise[2], None, "atom_ob2")
                self.lp_.create_fact(rel, True, "01_Placement_request")                       # (1a)
                self.lp_.create_fact(rel2, True, "14_Initial_placement_request")              # (14)
            elif premise[0] == "Right" or premise[0] == "right":
                rel = self.lp_.create_atom("l", premise[2], premise[1], index+1, "atom_ob2")
                rel2 = self.lp_.create_atom("il", premise[2], premise[1], None, "atom_ob2")
                self.lp_.create_fact(rel, True, "01_Placement_request")                       # (1b)
                self.lp_.create_fact(rel2, True, "14_Initial_placement_request")              # (14)

    def create_program(self, objects, phases):
        """
        This function creates the initial logic program for solving spatial relations, according
        to the specified objects and phases in the input parameters. The model created by applying
        these rules is the preferred mental model, which is obtained by using the first-free-fit-
        strategy(fff) for insertions of new objects.
        Altogether, there are 11 rules necessary to create the preferred model, summarized in the
        following. In order to mark ambiguities while constructing the models, 5 additional rules
        are needed (12a-14). Many of the rules need to be created for every object combination and
        time index, resulting in an exponential growth of rules, depending on the amount of
        objects for the problem. Some of the following rules are added in the function
        "add_premise_requests" (1, 14).

        The rules are:
        1.  Initial placement request. i is premise-number in problem
            - l(o1, o2, i) <-- True (left);    l(o2, o1, i) <-- True (right)
        2.  Closed world assumption for l-atom (relation)
            - l(o1, o2, 1) <-- False
        3.  All positions are not occupied in the beginning
            - ol(o, 1) <-- False;    or(o, 1) <-- False
        4.  Set direct left neighbour
            - ln(o1, o2, i) <-- l(o1, o2, i) and -ol(o2, i) and -or(o1, i)
        5.  ln-atom holds in the next phase:
            - ln(o1, o2, i+1) <-- ln(o1, o2, i)
        6.  Set occupied positions
            - ol(o2, i+1) <-- ln(o1, o2, i);   or(o1, i+1) <-- ln(o1, o2, i)
        7.  Adapted placement request:
            - l(o3, o1, i+1) <-- l(o3, o2, i+1) and ln(o1, 02, i)
        8.  Adapted placement request II
            - l(o2, o3, i+1) <-- l(o1, o3, i+1) and ln(o1, o2, i)
        9.  Direct left neighbourhood implies general left-relation. n is the last premise number
            - left(o1, o2) <-- ln(o1, o2, n)
        10. Transitivity of left-relation
            - left(o1, o3) <-- left(o1, o2) and left(o2, o3)
        11. Inverse of left-relation is right-relation
            - right(o1, o2) <-- left(o2, o1)

        Additional rules to enable the amb-atoms and il-constraints, which are later needed for the
        variation program:
        12a. Ambiguous objects found - mark ambiguity trough ambiguous-atom
             - amb(o3, o1, i+1) <-- l(o3, o2, i+1) and ln(o1, o2, i)
        12b. Ambiguous objects found - mark ambiguity trough ambiguous-atom II
             - amb(o3, o2, i+1) <-- l(o1, o3, i+1) and ln(o1, o2, i)
        13a. Pass on ambiguity-annotations (when a new object o3 should be inserted next to object
             o2, which is already marked as ambiguous, then the new object o3 is ambiguos as well).
             - amb(o3, o1, i+1) <-- l(o3, o2, i+1) and amb(o1, o2, i)
        13b. Pass on ambiguity-annotations II
             - amb(o3, o2, i+1) <-- l(o1, o3, i+1) and amb(o1, o2, i)
        14.  Initial placement requests. i is premise-number of the problem
             - il(o1, o2, i) <-- True (left);    il(o2, o1, i) <-- True (right)

        (Note: The rules 1-11 are taken from "A Computational Logic Approach to Human Spatial
        Reasoning. Dietz, Hölldobler, Höps. 2015")
        """
        for obj in objects:
            ol_ = self.lp_.create_atom("ol", obj, None, 1, "atom_ob")
            or_ = self.lp_.create_atom("or", obj, None, 1, "atom_ob")
            self.lp_.create_fact(ol_, False, "03_Free_position")                             # (3a)
            self.lp_.create_fact(or_, False, "03_Free_position")                             # (3b)

        for obj in objects:
            for obj2 in objects:
                if obj == obj2:
                    continue
                l_rel = self.lp_.create_atom("l", obj, obj2, 1, "atom_ob2")
                self.lp_.create_fact(l_rel, False, "02_CWA_placement_requests")              # (2)

                left = self.lp_.create_atom("left", obj, obj2, None, "atom_ob2")
                ln_last = self.lp_.create_atom("ln", obj, obj2, phases, "atom_ob2")
                self.lp_.create_rule("09_ln_left", left, ln_last)                            # (9)

                right = self.lp_.create_atom("right", obj2, obj, None, "atom_ob2")
                self.lp_.create_rule("11_Inverse_left", right, left)                         # (11)

                for time in range(1, phases+1):
                    l_rel2 = self.lp_.create_atom("l", obj, obj2, time, "atom_ob2")
                    ol_ = self.lp_.create_atom("ol", obj2, None, time, "atom_ob")
                    or_ = self.lp_.create_atom("or", obj, None, time, "atom_ob")
                    ln_ = self.lp_.create_atom("ln", obj, obj2, time, "atom_ob2")
                    and_ = OperatorAnd(l_rel2, OperatorNot(ol_), "and")
                    and2 = OperatorAnd(and_, OperatorNot(or_), "and2")
                    self.lp_.create_rule("04_Placement_left_neighbour", ln_, and2)           # (4)

                for time in range(1, phases):
                    ln_ = self.lp_.create_atom("ln", obj, obj2, time, "atom_ob2")
                    ln_next = self.lp_.create_atom("ln", obj, obj2, time+1, "atom_ob2")
                    self.lp_.create_rule("05_Left_neighbour_remains", ln_next, ln_)          # (5)

                    ol_next = self.lp_.create_atom("ol", obj2, None, time+1, "atom_ob")
                    or_next = self.lp_.create_atom("or", obj, None, time+1, "atom_ob")
                    self.lp_.create_rule("06_Occupied_position", ol_next, ln_)               # (6a)
                    self.lp_.create_rule("06_Occupied_position", or_next, ln_)               # (6b)

        for obj in objects:
            for obj2 in objects:
                if obj == obj2:
                    continue

                for time in range(1, phases+1):
                    for obj3 in objects:
                        if obj2 != obj3 and obj != obj3:
                            if time < phases:
                                ln_rel = self.lp_.create_atom("ln", obj, obj2, time, "atom_ob2")
                                lb_rel = self.lp_.create_atom("l", obj, obj3, time+1, "atom_ob2")
                                l3_rel = self.lp_.create_atom("l", obj2, obj3, time+1, "atom_ob2")
                                and2 = OperatorAnd(lb_rel, ln_rel, "and")
                                self.lp_.create_rule("08_Adapted_placement_request",         # (8)
                                                     l3_rel, and2)
                                amb2 = self.lp_.create_atom("amb", obj3, obj2, time+1, "atom_ob2")
                                self.lp_.create_rule("12a_Mark_ambiguity", amb2, and2)       # (12a)

                                l_rel = self.lp_.create_atom("l", obj3, obj2, time+1, "atom_ob2")
                                l2_rel = self.lp_.create_atom("l", obj3, obj, time+1, "atom_ob2")
                                and_ = OperatorAnd(l_rel, ln_rel, "and")
                                self.lp_.create_rule("07_Adapted_placement_request",         # (7)
                                                     l2_rel, and_)
                                amb = self.lp_.create_atom("amb", obj3, obj, time+1, "atom_ob2")
                                self.lp_.create_rule("12b_Mark_ambiguity", amb, and_)        # (12b)

                            if time >= 2:
                                for index in range(time, phases):
                                    l_rel_an = self.lp_.create_atom("l", obj3, obj2,
                                                                    index+1, "atom_ob2")
                                    l_rel_anb = self.lp_.create_atom("l", obj, obj3,
                                                                     index+1, "atom_ob2")
                                    amb_1 = self.lp_.create_atom("amb", obj, obj2,
                                                                 time, "atom_ob2")
                                    amb_3 = self.lp_.create_atom("amb", obj3, obj,
                                                                 index+1, "atom_ob2")
                                    amb_4 = self.lp_.create_atom("amb", obj3, obj2,
                                                                 index+1, "atom_ob2")
                                    and3 = OperatorAnd(l_rel_an, amb_1, "and")
                                    and4 = OperatorAnd(l_rel_anb, amb_1, "and")
                                    self.lp_.create_rule("13b_Inherited ambiguity",          # (13a)
                                                         amb_3, and3)
                                    self.lp_.create_rule("13a_Inherited ambiguity",          # (13b)
                                                         amb_4, and4)

                for obj3 in objects:
                    if obj2 != obj3 and obj != obj3:
                        left1 = self.lp_.create_atom("left", obj, obj2, None, "atom_ob2")
                        left2 = self.lp_.create_atom("left", obj2, obj3, None, "atom_ob2")
                        left3 = self.lp_.create_atom("left", obj, obj3, None, "atom_ob2")
                        self.lp_.create_rule("10_Transitivity_left", left3,
                                             OperatorAnd(left1, left2, "and"))                # (10)


#------------------------------------------ MODEL VARIATION ----------------------------------------

    def extract_relevant_atoms(self, phases):
        """
        This function extracts all relevant atoms which were computed by the logic program
        for the preferred mental model. These atoms are necessary to later construct all
        alternative models starting from the preferred mental model.
        The returned lists comprise one of all il- and ln-atoms which were True in the preferred
        mental model and one list of all swap-request-atoms (amb-atoms) which were True.
        """
        true_atoms = self.lp_.get_truth_atoms(True)
        ln_atoms_pos = []
        swap_requests = []
        for atom in true_atoms:
            if (atom.name[:2] == "ln" and atom.time_index == phases) or atom.name[:2] == "il":
                ln_atoms_pos.append(atom)
            if atom.name[:9] == "amb":
                swap_requests.append(atom)

        # Remove equivalent requests like "amb(a, b), amb(b, a)"
        for item in swap_requests:
            for item2 in swap_requests.copy():
                if item.obj == item2.obj2 and item.obj2 == item2.obj:
                    swap_requests.remove(item)
        return [ln_atoms_pos, swap_requests]

    def get_ambiguous_combinations(self, amb_list):
        """
        Given an input list of ambiguous-atoms (swap requests), this function computes all possible
        permutations of all lengths of these ambiguous-atoms. A list of sublists containing
        all permutation orders is returned.
        By computing all possible permutations of swap-requests, the implementation can compute all
        alternative models of the problem by considering all possible orders of swapping two objects
        in the model. Moreover, since each sublist with a permutation contains the enumerated
        ambiguous-atoms, the order of swapping objects can be considered in the construction-process
        of the alternative model.
        Example:
        The input list [amb(a, b), amb(b, c)] would result in the returned list:
        [[amb(a, b, 1)], [amb(b, c, 1)], [amb(a, b, 1), amb(b, c, 2)], [amb(b, c, 1), amb(a, b, 2)]]
        """
        all_combos = []
        for i in range(0, len(amb_list)+1):    # Compute all permutations of swap-requests.
            all_combos += list(permutations(amb_list, i))
        enumerated_combos = []
        for subset in all_combos:   # Enumerate the swap-request-permutations with a time index.
            sublist = []
            for index, atom in enumerate(subset, 1):
                sublist.append(self.lp_.create_atom(atom.name, atom.obj,
                                                    atom.obj2, index, "atom_ob2"))
            enumerated_combos.append(sublist)
        enumerated_combos.pop(0)    # Delete first element, which is always the empty list [].
        return enumerated_combos

    def setup_variation_phase_program(self, relevant_atoms, objects, phases):
        """
        This function initializes the logic program responsible for calculating all alternative
        mental models.
        The variation program consist altogether of 4 different kind of facts and 11 different
        kind of rules. The facts are added in this function and the functions "add_adjust_facts"
        and "compute_problem"; the rules ared added in the function "create_variation_program".
        First, the function creates a new logic program and calls the function "create_variation_
        program", which will add all rules necessary to construct alternative models.
        Next, positive atoms extracted from the least model of the logic program for the preferred
        mental model are added as facts accordingly to the current logic program for the variation
        process: Since the construction of alternative models always starts by alternating the
        preferred mental model, the ln-atoms which were True in the last phase of the preferred
        model construction are added as positive facts to the current logic program. Moreover, the
        initial requests (il-atoms) need to be added as positive facts as well, since the variation
        program uses them as constraints to prevent constructing models which would violate the
        premises of the problem.
        In the last step, all atoms in the logic program are set to Unknown.
        """
        self.lp_ = LogicProgram()
        self.create_variation_program(objects, phases, len(relevant_atoms[1]))

        for atom in relevant_atoms[0]: # True ln & il-atoms
            at_ = self.lp_.create_atom(atom.name, atom.obj, atom.obj2, atom.time_index, "atom_ob2")
            if atom.name == "il":
                self.lp_.create_fact(at_, True, "17_il_constraints")                          # (17)
            else:
                self.lp_.create_fact(at_, True, "18_Neihbours_preferred_model")               # (18)

        for atom in self.lp_.atoms:
            atom.boolean_val = None

    def add_adjust_facts(self, amb_atom, phases):
        """
        This function adds the object-adjustment-facts which are necessary to construct
        alternative mental models.
        The adjustment-facts are created according to the input swap-request-atom (amb-atom).
        For instance, the input atom "amb(a, b, 1)" would result in adding the facts:
        adj(a, 3) <-- True, adj(b, 3) <-- True
        in case the problem consists of 3 premises and a query (for 2 premises, the adjust-atoms
        would have the time index 2, for 4 premises the time-index would be 4 etc.)
        Afterwards, the two new facts are returned as a list in order to refer to these facts to
        later delete them by the function "reset_lp_for_variation" after the construction
        process of the current alternative model is done.
        """
        adjust = self.lp_.create_atom("adj", amb_atom.obj, None,
                                      amb_atom.time_index+phases-1, "atom_ob")
        adjust2 = self.lp_.create_atom("adj", amb_atom.obj2, None,
                                       amb_atom.time_index+phases-1, "atom_ob")
        fact1 = self.lp_.create_fact(adjust, True, "16_Adapt_object_position")               # (16a)
        fact2 = self.lp_.create_fact(adjust2, True, "16_Adapt_object_position")              # (16b)
        return [fact1, fact2]

    def create_variation_program(self, objects, phases, swap_phases):
        """
        This function creates the logic program which computes alternative mental models
        in the variation phase.
        Only rules are added in this function, the facts are added in different functions.
        The 11 rules incorporate three that were already used in the computation of the preferred
        mental model (rules 9, 10 and 11), one rule to preserve unchanged ln-atoms, three rules
        to swap objects and some rules to detect incomplete or violating models.

        objects in the model:
        19.  The object order within the ambiguous-atom is irrelevant. Not enabling this rule would
             lead to having to add three more rules of the kind 9, 10 and 11 with a changed object
             order in the amb-atom.
             - amb(o2, o1, i) <-- amb(o1, o2, i)
        20.  ln-atoms remain unchanged in the next phase, if none of the concerning objects
             is supposed to be adjusted.
             - ln(o1, o2, i+1) <-- ln(o1, o2, i) and not(ctxt(adj(o1, i))) and not(ctxt(adj(o2, i))
        21a. Swap two objects in the same ln-atom (left neighbourhood relation).
             - ln(o2, o1, phases-1+i+1) <-- ctxt(amb(o1, o2, i)) and ln(o1, o2, phases-1+i)
                                            and not(ctxt(il(o1, o2)))
        21b. Swap two objects - left object in ln-atom (left neighbourhood relation) is swapped.
             - ln(o3, o2, phases-1+i+1) <-- ctxt(amb(o1, o3, i)) and ln(o1, o2, phases-1+i)
                                            and not(ctxt(il(o2, o3)))
        21c. Swap two objects - right object in ln-atom (left neighbourhood relation) is swapped.
             - ln(o1, o3, phases-1+i+1) <-- ctxt(amb(o3, o2, i)) and ln(o1, o2, phases-1+i)
                                            and not(ctxt(il(o3, o1)))
        22.  In case the constructed model violates a constraint, abnormality (ab) is enabled.
             - ab <-- left(o2, o1) and ctxt(il(o1, o2))
        23a. Construct a complete chain, considering all objects, with the help of the
             left-relations (here: example-rule for 4 objects).
             - chain <-- left(o1, o2) and left(o2, o3) and left(o3, o4)
        23b. If no complete chain exists, the model is abnormal (not fully constructed due
             to constraints).
             - ab <-- not(chain)
        """
        self.add_chain_rule(objects)                                                         # (23a)
        ab_ = self.lp_.create_atom("ab")
        chain = self.lp_.create_atom("chain")
        self.lp_.create_rule("23b_Incomplete_model", ab_, OperatorNot(chain))                # (23b)

        for obj in objects:
            for obj2 in objects:
                if obj == obj2:
                    continue

                left = self.lp_.create_atom("left", obj, obj2, None, "atom_ob2")
                ln_last = self.lp_.create_atom("ln", obj, obj2, phases+swap_phases, "atom_ob2")
                self.lp_.create_rule("09_ln_left", left, ln_last)                            # (9)

                right = self.lp_.create_atom("right", obj2, obj, None, "atom_ob2")
                self.lp_.create_rule("11_Inverse_left", right, left)                         # (11)

                il_ = self.lp_.create_atom("il", obj2, obj, None, "atom_ob2")
                self.lp_.create_rule("22_Violated_constraints", ab_,
                                     OperatorAnd(left, ContextOperator(il_), "and"))         # (22)

                for obj3 in objects:
                    if obj != obj3 and obj2 != obj3:
                        left1 = self.lp_.create_atom("left", obj, obj2, None, "atom_ob2")
                        left2 = self.lp_.create_atom("left", obj2, obj3, None, "atom_ob2")
                        left3 = self.lp_.create_atom("left", obj, obj3, None, "atom_ob2")
                        self.lp_.create_rule("10_Transitivity_left", left3,
                                             OperatorAnd(ContextOperator(left1),
                                                         ContextOperator(left2), "and"))     # (10)

                for i in range(1, swap_phases+1):
                    amb = self.lp_.create_atom("amb", obj, obj2, i, "atom_ob2")
                    amb2 = self.lp_.create_atom("amb", obj2, obj, i, "atom_ob2")
                    self.lp_.create_rule("19_amb_order_irrelevant", amb2, amb)               # (19)

                    adjust_ol = self.lp_.create_atom("adj", obj, None, phases-1+i, "atom_ob")
                    adjust_or = self.lp_.create_atom("adj", obj2, None, phases-1+i, "atom_ob")
                    and_ = OperatorAnd(OperatorNot(ContextOperator(adjust_ol)),
                                       OperatorNot(ContextOperator(adjust_or)), "and")
                    and_2 = OperatorAnd(self.lp_.create_atom(
                        "ln", obj, obj2, phases-1+i, "atom_ob2"), and_, "and")
                    ln_next = self.lp_.create_atom("ln", obj, obj2, phases+i, "atom_ob2")
                    self.lp_.create_rule("20_Unchanged_neighbourhood", ln_next, and_2)       # (20)

                    ln_ = self.lp_.create_atom("ln", obj, obj2, phases-1+i, "atom_ob2")
                    ln2 = self.lp_.create_atom("ln", obj2, obj, phases-1+i+1, "atom_ob2")
                    not_il = OperatorNot(ContextOperator(
                        self.lp_.create_atom("il", obj, obj2, None, "atom_ob2")))
                    and_a = OperatorAnd(ContextOperator(amb), ln_, "and")
                    and_a2 = OperatorAnd(and_a, not_il, "and")
                    self.lp_.create_rule("21a_Object_swap_a", ln2, and_a2)                   # (21a)

                    for obj3 in objects:
                        if obj != obj3 and obj2 != obj3:
                            amb_c = self.lp_.create_atom("amb", obj, obj3, i, "atom_ob2")
                            ln_c = self.lp_.create_atom("ln", obj, obj2, phases-1+i, "atom_ob2")
                            ln2_c = self.lp_.create_atom("ln", obj3, obj2,
                                                         phases-1+i+1, "atom_ob2")
                            not_il_c = OperatorNot(ContextOperator(
                                self.lp_.create_atom("il", obj2, obj3, None, "atom_ob2")))
                            and_c = OperatorAnd(ContextOperator(amb_c), ln_c, "and")
                            and_c2 = OperatorAnd(and_c, not_il_c, "and")
                            self.lp_.create_rule("21b_Object_swap_b", ln2_c, and_c2)         # (21b)

                            amb = self.lp_.create_atom("amb", obj3, obj2, i, "atom_ob2")
                            ln_b = self.lp_.create_atom("ln", obj, obj2, phases-1+i, "atom_ob2")
                            ln2_b = self.lp_.create_atom("ln", obj, obj3, phases-1+i+1, "atom_ob2")
                            not_il_b = OperatorNot(ContextOperator(
                                self.lp_.create_atom("il", obj3, obj, None, "atom_ob2")))
                            and_b = OperatorAnd(ContextOperator(amb), ln_b, "and")
                            and_b2 = OperatorAnd(and_b, not_il_b, "and")
                            self.lp_.create_rule("21c_Object_swap_c", ln2_b, and_b2)         # (21c)


    def add_chain_rule(self, objects):
        """
        This function adds rule no. 23a to the variation program.
        Since this rule depends on the amount of objects and needs to consider every object
        in each rule instance, the function makes use of nested for-loops.
        The goal of the function is to create a rule which links all objects in all possible
        ways to a full length-chain, containing all objects.
        For example, consider 3 different objects, then this function will create the following
        6 rules:
        chain <-- left(o1, o2) and left(o2, o3) .... chain <-- left(o3, o2) and left(o2, o1)

        (Note: This function is limited to a maximum of 5 objects)
        """
        for obj in objects:
            for obj2 in objects:
                if obj == obj2:
                    continue
                if len(objects) == 2:
                    left = self.lp_.create_atom("left", obj, obj2, None, "atom_ob2")
                    chain = self.lp_.create_atom("chain")
                    self.lp_.create_rule("23a_chain_construction", chain, left)
                else:
                    for obj3 in objects:
                        if obj3 == obj or obj3 == obj2:
                            continue
                        if len(objects) == 3:
                            left1 = self.lp_.create_atom("left", obj, obj2, None, "atom_ob2")
                            left2 = self.lp_.create_atom("left", obj2, obj3, None, "atom_ob2")
                            chain = self.lp_.create_atom("chain")
                            self.lp_.create_rule("23a_chain_construction", chain,
                                                 OperatorAnd(left1, left2, "and"))
                        else:
                            for obj4 in objects:
                                if obj4 == obj or obj4 == obj2 or obj4 == obj3:
                                    continue
                                if len(objects) == 4:
                                    left1 = self.lp_.create_atom("left", obj, obj2,
                                                                 None, "atom_ob2")
                                    left2 = self.lp_.create_atom("left", obj2, obj3,
                                                                 None, "atom_ob2")
                                    left3 = self.lp_.create_atom("left", obj3, obj4,
                                                                 None, "atom_ob2")
                                    chain = self.lp_.create_atom("chain")
                                    and1 = OperatorAnd(left1, left2, "and")
                                    self.lp_.create_rule("23a_chain_construction", chain,
                                                         OperatorAnd(and1, left3, "and"))
                                else:
                                    for obj5 in objects:
                                        if (obj5 == obj or obj5 == obj2
                                                or obj5 == obj3 or obj5 == obj4):
                                            continue
                                        if len(objects) == 5:
                                            left1 = self.lp_.create_atom("left", obj, obj2,
                                                                         None, "atom_ob2")
                                            left2 = self.lp_.create_atom("left", obj2, obj3,
                                                                         None, "atom_ob2")
                                            left3 = self.lp_.create_atom("left", obj3, obj4,
                                                                         None, "atom_ob2")
                                            left4 = self.lp_.create_atom("left", obj4, obj5,
                                                                         None, "atom_ob2")
                                            chain = self.lp_.create_atom("chain")
                                            and1 = OperatorAnd(left1, left2, "and")
                                            and2 = OperatorAnd(and1, left3, "and")
                                            self.lp_.create_rule("23a_chain_construction", chain,
                                                                 OperatorAnd(and2, left4, "and"))

    def reset_lp_for_variation(self, facts_to_delete):
        """
        This function resets the logic program for the variation-phase, so that all
        alternative models can be computed (starting from the logic program which resulted
        in the preferred mental model).
        In the first step, all additional facts regarding the ambiguous- and adjustment-atoms
        which were necessary to build the previous alternative model, are deleted from lp.
        Afterwards, all atoms in the logic program are reset to None.
        The logic program is then in its initial variation-phase-state and ready to compute
        the next alternative model.
        """
        for fact in facts_to_delete:
            self.lp_.rules.remove(fact)
        for at_ in self.lp_.atoms:
            at_.boolean_val = None


# -------------------------------------- ADDITIONAL MODEL FUNCTIONS --------------------------------

    def model_complete(self):
        """
        This function checks whether a model constructed in the variation phase is complete.
        If that´s the case, the function returns True, else False.
        """
        ab_val = self.lp_.get_atom("ab", None, None).boolean_val
        if ab_val is False:
            return True
        return False

    @staticmethod
    def count_references(ln_atoms):
        """
        Helper-function for "visualize_model".
        This function counts, how often each object is referred to in the ln-atoms. A dictionary
        containing all objects as keys and the counters as the according values is returned.
        """
        count_dict = {}
        for at_ in ln_atoms:
            if str(at_.obj) in count_dict:
                count_dict[str(at_.obj)] += 1
            else:
                count_dict[str(at_.obj)] = 1
            if str(at_.obj2) in count_dict:
                count_dict[str(at_.obj2)] += 1
            else:
                count_dict[str(at_.obj2)] = 1
        return count_dict

    def visualize_model(self, phases, print_):
        """
        This function visualizes a model.
        First, all ln-atoms, that are true in the last phase, are collected.
        Next, a dictionary counts, how often each object is referred to - in a complete model
        consisting of all objects, each object would need to be referred to twice, except the
        start and end object, which is only referred to once.
        The function then selectes the ln-atom which starts the model from the left side
        by picking the ln-atom where the left object is only referred to once.
        Afterwards, a visual model is constructed object by object, by searching for the next
        ln-atom in the true ln-atoms-list, which has the current last object as the left object,
        matching to the right object of the previous ln-atom.
        """
        # Get all positive left-neighbour (ln) relations
        true_atoms = ([at for at in self.lp_.get_truth_atoms(True)])
        ln_atoms = []
        for at_ in true_atoms:
            if at_.name == "ln" and at_.time_index == phases:
                ln_atoms.append(at_)

        # Count, how often each objects is referred to.
        count_dict = self.count_references(ln_atoms)

        # If each object is only referred to once, certainly no full model can be constructed.
        referred_once = []
        for key, value in count_dict.items():
            if value == 1:
                referred_once.append(key)
        if len(referred_once) >= 3:
            return ""

        # Determine start atom.
        start_atom = ""
        for at_ in ln_atoms:
            for key, value in count_dict.items():
                if at_.obj == key and value == 1:
                    start_atom = at_

        # Start constructing model
        model = ""
        if start_atom:
            model += str(start_atom.obj)+" "+str(start_atom.obj2)+" "
            ln_atoms.remove(start_atom)
        else:
            return ""
        counter = 0
        while ln_atoms:
            for at_ in ln_atoms:
                if at_.obj == model[-2]:
                    model += str(at_.obj2)+" "
                    ln_atoms.remove(at_)
                counter += 1
        if print_:
            print(model)
        return model



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
    This class implements atoms. All atoms have a name, a boolean value and an evaluation-function
    to calculate their current boolean value.
    Moreover, atoms can contain objects. If an atom contains an object and has the boolean value
    "True", then this object has the feature which is encoded by the atom(name). For instance,
    "baker(o1)" with boolean_val = True encodes the fact that o1 is a baker.
    If an atom contains two objects, it encodes a relation that holds or does not hold between
    the two referred objects, according to the boolean value (f.i. "left(a, b)" means that object
    a is to the left of object b, if the boolean value of that atom is True).
    Additionally, atoms can have a time index, meaning that the encoded feature/relation does only
    hold at the specified time step.
    The attributes "type" and "subtype" are used to distinguish different kinds of formula
    and atoms (atoms can have the subtypes "atom_ob" (= "atoms with one object"), "atom_ob2"
    (= "atoms with two objects (relation)"), "bool" (= atom which is a boolean) and
    "no_obj" (= "atom without object")).
    """
    def __init__(self, name, obj=None, obj2=None, time_index=None, subtype="no_obj", bool_val=None):
        super().__init__()
        self.name = name
        self.obj = obj
        self.obj2 = obj2
        self.time_index = time_index
        self.type = "atom"
        self.subtype = subtype
        self.boolean_val = bool_val

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
        if self.subtype == "atom_ob" or self.subtype == "atom_ob2":
            return self
        return None

    def __str__(self):
        """
        Function returns a string representation of the atom.
        """
        if self.subtype == "atom_ob":
            if self.time_index is None:
                return self.name + "(" + str(self.obj) + ")"
            return self.name + "(" + str(self.obj) + ", " + str(self.time_index) + ")"
        elif self.subtype == "atom_ob2":
            if self.time_index is None:
                return self.name + "(" + str(self.obj) + ", " + str(self.obj2) + ")"
            return (self.name + "(" + str(self.obj) + ", " + str(self.obj2)
                    + ", " + str(self.time_index) + ")")
        return str(self.name)


class ContextOperator(Formula):
    """
    This class implements the context operator. The context operator, wrapped around an atom,
    is evaluated to True, if the boolean value of the atom inside is True, else False.
    The context operator was originally introduced by Dietz, Hoelldobler and Pereira in order
    to solve a technical bug. (For further information, see E.-A. Dietz, S. Hoelldobler,
    L. M. Pereira. Contextual reasoning: Usually birds can abductively fly. 2017).
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
        if self.content.type == "atom" and (self.content.subtype == "atom_ob" or
                                            self.content.subtype == "atom_ob2"):
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
        if self.content.type == "atom" and (self.content.subtype == "atom_ob" or
                                            self.content.subtype == "atom_ob2"):
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
    This class implements logic programs, which contain atoms, rules and positive / negative facts
    (to encode the truth-value of atoms).
    Logic Programs are the basic framework used in the Weak-Completion-Semantics.
    The LogicProgram class contains important functions which encode the main
    concepts used in Weak-Completion-Semantics, like the implementation of the semantic operator
    in "compute_least_model" as well as many helper-functions.
    """
    def __init__(self):
        """
        Initializes a LogicProgram-object.
        A logic program contains a list of atoms (that can contain one or two objects)
        and rules (implications).
        """
        self.atoms = []
        self.rules = []

    def create_atom(self, name, obj=None, obj2=None, time_index=None,
                    subtype="no_obj", append=True):
        """
        Function creates an atom with the specified name (and optionally objects and a time index)
        and adds it to the atoms-list of the logic program (only if the attribute "append" is True).
        In case the atom already exists, the according atom is returned.
        """
        for i in range(0, len(self.atoms)):
            if (self.atoms[i].name == name and self.atoms[i].obj == obj
                    and self.atoms[i].obj2 == obj2 and self.atoms[i].time_index == time_index):
                return self.atoms[i]
        at_ = Atom(name, obj, obj2, time_index, subtype, None)
        if append:
            self.atoms.append(at_)
        return at_

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
        atom = Atom(true_false, None, None, None, "bool")
        fact = OperatorImplication(consequent, atom, name)
        self.rules.append(fact)
        return fact

    def get_atom(self, name, obj, obj2, time_index=None):
        """
        Function returns the atom which matches the specified name and objects.
        """
        for i in range(0, len(self.atoms)):
            if (self.atoms[i].name == name and self.atoms[i].obj == obj
                    and self.atoms[i].obj2 == obj2 and self.atoms[i].time_index == time_index):
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
        print("\n")
        print("=" * 100, "\nPositive atoms:\n", pos, "\n\nNegative atoms:\n", neg,
              "\n\nUnknown atoms:\n", un_, "\n")
        print("=" * 100, "\n")

    def print_lp_truth(self, state):
        """
        Function prints all rules of the logic program.
        """
        print("-" * 40, str(state), "-" * 40)
        self.rules.sort(key=attrgetter('name'))
        for rule in self.rules:
            string = rule.name
            blank = 35 - len(string)
            string = string + blank * " " + str(rule)
            print(string)
        print("-" * 100)

    def evaluate(self):
        """
        Function evaluates all rules in the logic program by calling their evaluate-function.
        Since the logic-operators are implemented inductively, the evaluation-processes heads down
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
            if (rule.left.subtype != "bool"
                    and (rule.left.obj == rule_head.obj and rule.left.obj2 == rule_head.obj2)
                    and rule.left.name == rule_head.name and rule.right.boolean_val != False
                    and rule.left.time_index == rule_head.time_index):
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
            if rule.right.boolean_val != rule.left.boolean_val:
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
        itr_counter = 0
        self.evaluate()
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
                    print("RULE", atom[0], blank_len * " ", "set", str(atom[1]),
                          "  to", str(atom[2]))
                atom[1].boolean_val = atom[2]
                printed_rule_heads.append(atom[1])
            self.evaluate()
            itr_counter += 1

    def compute_conclusion(self, conclusion_prem, print_):
        """
        This function computes the conclusion by checking whether the input premise,
        which is a query (the last premise of the problem) holds in the least model.
        This is the case if the according atom encoding a relation is found to be true
        in the least model, otherwise (atom is None or False), the query does not hold.
        """
        if print_:
            print("\nDoes the relation", conclusion_prem, " hold?")
        rel = self.get_atom(conclusion_prem[0].lower(), conclusion_prem[1], conclusion_prem[2])
        if rel.boolean_val is True:
            if print_:
                print("Yes!\n")
            return True
        if print_:
            print("No!\n")
        return False



# ---------------------------------------------- MAIN ----------------------------------------------

def main():
    """
    Main-function, containing some sample-calls of the model.
    """
    model = WCSSpatial()
    model.print_output = True

    spatial_task = [["left", "A", "B"], ["left", "M", "B"]]
    query = ["left", "M", "A"]
    model.compute_problem(spatial_task, query)

    model.compute_problem([["left", "A", "B"], ["left", "A", "C"],
                           ["left", "C", "D"]], ["left", "B", "D"])

if __name__ == "__main__":
    main()
