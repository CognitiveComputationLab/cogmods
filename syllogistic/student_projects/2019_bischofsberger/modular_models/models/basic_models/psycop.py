# coding=utf-8

import os
import random
import sys
from collections import namedtuple
from enum import Enum

import ccobra
from anytree import AnyNode, LevelOrderIter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from modular_models.util import sylutil
from modular_models.models.basic_models.interface import SyllogisticReasoningModel


class PSYCOP(SyllogisticReasoningModel):
    """ PSYCOP model according to Rips (1994). """

    def __init__(self):
        SyllogisticReasoningModel.__init__(self)

        # Prospensity to guess instead of replying NVC if no conclusion is found
        self.params["guess"] = 0.0

        # Whether or not existential implicatures are added to the forward propositions
        self.params["premise_implicatures_existential"] = True

        # Whether or not gricean implicatures are added to the forward propositions
        self.params["premise_implicatures_grice"] = True

        # Whether or not proving conclusion implicatures is required to prove a conclusion
        self.params["conclusion_implicatures"] = False

        # Availability of rules
        self.params["rule_transitivity"] = True
        self.params["rule_exclusivity"] = True
        self.params["rule_conversion"] = True
        self.params["rule_fw_and_elimination"] = True
        self.params["rule_bw_and_introduction"] = True
        self.params["rule_bw_conjunctive_syllogism"] = True
        self.params["rule_bw_if_elimination"] = True
        self.params["rule_bw_not_introduction"] = True

        self.param_grid["guess"] = [0.0, 1.0]

        self.param_grid["premise_implicatures_existential"] = [True, False]
        self.param_grid["premise_implicatures_grice"] = [True, False]
        self.param_grid["conclusion_implicatures"] = [False, True]

        self.param_grid["rule_transitivity"] = [True, False]
        self.param_grid["rule_exclusivity"] = [True, False]
        self.param_grid["rule_conversion"] = [True, False]
        self.param_grid["rule_fw_and_elimination"] = [True, False]
        self.param_grid["rule_bw_and_introduction"] = [True, False]
        self.param_grid["rule_bw_conjunctive_syllogism"] = [True, False]
        self.param_grid["rule_bw_if_elimination"] = [True, False]
        self.param_grid["rule_bw_not_introduction"] = [True, False]


    class Prop:
        """ abstract representation of a categorical proposition like a syllogistic premise or
        conclusion.

        Example:
        All A are B = Prop(PT.implies, Prop(PT.atomic, Atom("A", 936), None), Prop(PT.atomic, Atom("B", 936), None))
        """

        def __init__(self, type, arg1, arg2):
            # proposition type like atom or conjunction
            self.type = type

            self.v1 = arg1
            self.v2 = arg2

        def __repr__(self):
            if self.type == PSYCOP.PT.atomic:
                if self.v1.is_name:
                    if self.v1.hat:
                        var = "â"
                    else:
                        var = "a"
                else:
                    var = "x"
                return self.v1.predicate + "(" + var + "_" + str(self.v1.arg_id) + ")"
            elif self.type == PSYCOP.PT.negation:
                return "NOT (" + self.v1.__repr__() + ")"
            elif self.type == PSYCOP.PT.implies:
                return "(" + self.v1.__repr__() + " -> " + self.v2.__repr__() + ")"
            elif self.type == PSYCOP.PT.conjunction:
                return "(" + self.v1.__repr__() + " AND " + self.v2.__repr__() + ")"

    # proposition type
    PT = Enum("PT", "atomic negation implies conjunction")

    """ representation of an atom = predicate + argument. Additional info (hat, name) is required by
    PSYCOP, example:
        Red(â) = Atom("Red", i, True, True) where i identifies â """
    Atom = namedtuple("Atom", "predicate arg_id is_name hat")

    # unique identifier for objects
    max_id = -1

    def get_fresh_id(self):
        self.max_id = self.max_id + 1
        return self.max_id

    def get_atomic_proposition(self, predicate, arg_id, is_name, hat):
        return self.Prop(self.PT.atomic, self.Atom(predicate, arg_id, is_name, hat), None)

    def encode_proposition(self, p, hat=False):
        """
        >>> m = PSYCOP()
        >>> m.encode_proposition("Aac")
        (A(x_0) -> C(x_0))
        >>> m.encode_proposition("Iac")
        (A(a_1) AND C(a_1))
        """

        i = self.get_fresh_id()

        if p[0] == "A":
            # A(x) -> B(x)
            return self.Prop(self.PT.implies,
                             self.get_atomic_proposition(p[1].upper(), i, False, hat),
                             self.get_atomic_proposition(p[2].upper(), i, False, hat))
        elif p[0] == "E":
            # not (A(x) and B(x))
            return self.Prop(self.PT.negation,
                             self.Prop(self.PT.conjunction,
                                       self.get_atomic_proposition(p[1].upper(), i, False, hat),
                                       self.get_atomic_proposition(p[2].upper(), i, False, hat)),
                             None)
        elif p[0] == "I":
            # A(a) and B(a)
            return self.Prop(self.PT.conjunction,
                             self.get_atomic_proposition(p[1].upper(), i, True, hat),
                             self.get_atomic_proposition(p[2].upper(), i, True, hat))
        else:
            # A(a) and not B(a)
            return self.Prop(self.PT.conjunction,
                             self.get_atomic_proposition(p[1].upper(), i, True, hat),
                             self.Prop(self.PT.negation,
                                       self.get_atomic_proposition(p[2].upper(), i, True, hat),
                                       None))

    def encode_premises(self, syllogism, ex_implicatures=True, grice_implicatures=False):
        """ Encode premises as propositions, possibly adding implicatures """
        to = sylutil.term_order(syllogism[2])
        premises = []
        pr = []
        for i in [0, 1]:
            pr.append(syllogism[i] + to[i])
        pr = sylutil.add_implicatures(pr, existential=ex_implicatures, gricean=grice_implicatures)
        for p in pr:
            premises.append(self.encode_proposition(p, True))
        return premises

    def isomorphic(self, p1, p2, same_nameness=False):
        """ same_nameness = True <-> "notational variant", see p. 197

        >>> m = PSYCOP()
        >>> a0 = m.Prop(m.PT.atomic, m.Atom("A", 0, False, False), None)
        >>> a1 = m.Prop(m.PT.atomic, m.Atom("A", 1, False, False), None)
        >>> b = m.Prop(m.PT.atomic, m.Atom("B", 2, False, False), None)
        >>> p1 = m.Prop(m.PT.implies, a0, b)
        >>> p2 = m.Prop(m.PT.implies, a1, b)
        >>> m.isomorphic(p1,p2)
        True
        >>> m.isomorphic(m.Prop(m.PT.negation, p1, None),m.Prop(m.PT.negation, p2, None))
        True
        >>> m.isomorphic(p1,m.Prop(m.PT.negation, p2, None))
        False
        >>> p3 = m.Prop(m.PT.conjunction, a1, b)
        >>> m.isomorphic(p1,p3)
        False
        """

        if p1 is None and p2 is None:
            return True
        if p1 is None or p2 is None:
            return False

        if type(p1) is self.Atom and type(p2) is self.Atom:
            if p1.predicate == p2.predicate:
                if same_nameness:
                    if p1.is_name == p2.is_name:
                        return True
                    return False
                return True
            return False
        if type(p1) is self.Atom or type(p2) is self.Atom:
            return False

        if p1.type == p2.type:
            return self.isomorphic(p1.v1, p2.v1) and self.isomorphic(p1.v2, p2.v2)
        return False

    def contains_isomorphic_proposition(self, domain, p):
        for pd in domain:
            if self.isomorphic(pd, p):
                return True
        return False

    def atom_prop_replace_properties(self, p, new_arg_id=None, new_is_name=None, new_hat=None):
        if new_arg_id is None:
            new_arg_id = p.v1.arg_id
        if new_is_name is None:
            new_is_name = p.v1.is_name
        if new_hat is None:
            new_hat = p.v1.hat
        return self.Prop(self.PT.atomic,
                         self.Atom(p.v1.predicate, new_arg_id, new_is_name, new_hat), None)

    def prop_replace_properties(self, p, new_arg_id=None, new_is_name=None, new_hat=None):
        if p.type == self.PT.negation:
            return self.Prop(self.PT.negation,
                             self.atom_prop_replace_properties(p.v1, new_arg_id, new_is_name,
                                                               new_hat), None)
        return self.atom_prop_replace_properties(p, new_arg_id, new_is_name, new_hat)

    def rule_transitivity(self, p1, p2, domain):
        """ PSYCOP transitivity rule

        >>> m = PSYCOP()
        >>> i = m.get_fresh_id()
        >>> a = m.Prop(m.PT.atomic, m.Atom("A", i, False, False), None)
        >>> b = m.Prop(m.PT.atomic, m.Atom("B", i, False, False), None)
        >>> c = m.Prop(m.PT.atomic, m.Atom("C", i, False, False), None)
        >>> p1 = m.Prop(m.PT.implies, a, b)
        >>> p2 = m.Prop(m.PT.implies, b, c)
        >>> m.rule_transitivity(p1, p2, set())
        [(A(x_1) -> C(x_1))]
        """
        if p1.type == self.PT.implies and p2.type == self.PT.implies:
            if p1.v1.type == self.PT.atomic and p1.v2.type == self.PT.atomic and \
                    p2.v1.type == self.PT.atomic and p2.v2.type == self.PT.atomic:
                if p1.v1.v1.arg_id == p1.v2.v1.arg_id and p2.v1.v1.arg_id == p2.v2.v1.arg_id:
                    if not p1.v1.v1.is_name and not p1.v2.v1.is_name and not p2.v1.v1.is_name and not p2.v2.v1.is_name:
                        if p1.v2.v1.predicate == p2.v1.v1.predicate:
                            i = self.get_fresh_id()
                            p = self.Prop(self.PT.implies,
                                          self.atom_prop_replace_properties(p1.v1, i),
                                          self.atom_prop_replace_properties(p2.v2, i))
                            if not self.contains_isomorphic_proposition(domain, p):
                                return [p]
        return []

    def rule_exclusivity(self, p1, p2, domain):
        """ PSYCOP exclusivity rule

        >>> m = PSYCOP()
        >>> i = m.get_fresh_id()
        >>> j = m.get_fresh_id()
        >>> ai = m.Prop(m.PT.atomic, m.Atom("A", i, False, False), None)
        >>> bi = m.Prop(m.PT.atomic, m.Atom("B", i, False, False), None)
        >>> bj = m.Prop(m.PT.atomic, m.Atom("B", j, False, False), None)
        >>> cj = m.Prop(m.PT.atomic, m.Atom("C", j, False, False), None)
        >>> p1 = m.Prop(m.PT.implies, ai, bi)
        >>> p2 = m.Prop(m.PT.negation, m.Prop(m.PT.conjunction, bj, cj), None)
        >>> m.rule_exclusivity(p1, p2, set())
        [NOT ((A(x_2) AND C(x_2)))]
        """

        if p1.type == self.PT.implies and p2.type == self.PT.negation:
            if p2.v1.type == self.PT.conjunction:
                if p1.v1.type == self.PT.atomic and p1.v2.type == self.PT.atomic:
                    if p2.v1.v1.type == self.PT.atomic and p2.v1.v2.type == self.PT.atomic:
                        if p1.v1.v1.arg_id == p1.v2.v1.arg_id and p2.v1.v1.v1.arg_id == p2.v1.v2.v1.arg_id:
                            if not p1.v1.v1.is_name and not p1.v2.v1.is_name and not p2.v1.v1.v1.is_name and not p2.v1.v2.v1.is_name:
                                if p1.v2.v1.predicate == p2.v1.v1.v1.predicate:
                                    i = self.get_fresh_id()
                                    p = self.Prop(self.PT.negation,
                                                  self.Prop(self.PT.conjunction,
                                                            self.atom_prop_replace_properties(p1.v1,
                                                                                              i),
                                                            self.atom_prop_replace_properties(
                                                                p2.v1.v2, i)),
                                                  None)
                                    if not self.contains_isomorphic_proposition(domain, p):
                                        return [p]
        return []

    def rule_conversion(self, p, domain):
        """ PSYCOP conversion rule

        >>> m = PSYCOP()
        >>> i = m.get_fresh_id()
        >>> a = m.Prop(m.PT.atomic, m.Atom("A", i, False, False), None)
        >>> b = m.Prop(m.PT.atomic, m.Atom("B", i, False, False), None)
        >>> p = m.Prop(m.PT.negation, m.Prop(m.PT.conjunction, a, b), None)
        >>> m.rule_conversion(p, set())
        [NOT ((B(x_1) AND A(x_1)))]
        """

        if p.type == self.PT.negation:
            if p.v1.type == self.PT.conjunction:
                if p.v1.v1.type == self.PT.atomic and p.v1.v2.type == self.PT.atomic:
                    i = self.get_fresh_id()
                    p_new = self.Prop(self.PT.negation,
                                      self.Prop(self.PT.conjunction,
                                                self.atom_prop_replace_properties(p.v1.v2, i),
                                                self.atom_prop_replace_properties(p.v1.v1, i)),
                                      None)
                    if not self.contains_isomorphic_proposition(domain, p_new):
                        return [p_new]
        return []

    def get_leftmost_atom(self, p):
        """ Returns leftmost atom in p. """

        if p.type == self.PT.atomic:
            return p.v1
        else:
            return self.get_leftmost_atom(p.v1)

    def matching(self, p, g):
        if self.isomorphic(p, g):
            # note: the leftmost atom is equal to any atom in the proposition
            pa, ga = self.get_leftmost_atom(p), self.get_leftmost_atom(g)
            if pa == ga:
                # Propositions are equal
                return True
            if not pa.is_name and not ga.is_name:
                # Matching 1
                return True
            if pa.is_name and ga.is_name and not ga.hat:
                # Matching 2
                return True
            if not pa.is_name and ga.is_name:
                # Matching 4
                return True  # ?
        return False

    def rule_forward_and_elimination(self, p):
        if p.type == self.PT.conjunction:
            return [p.v1, p.v2]
        return []

    def rule_backward_and_introduction(self, g):
        return self.rule_forward_and_elimination(g)

    def rule_backward_conjunctive_syllogism(self, p, g):
        """
        a = m.Prop(m.PT.atomic, v1='a', v2=None)
        b = m.Prop(m.PT.atomic, v1='b', v2=None)

        >>> m = PSYCOP()
        >>> i = m.get_fresh_id()
        >>> a = m.Prop(m.PT.atomic, m.Atom("A", i, False, False), None)
        >>> b = m.Prop(m.PT.atomic, m.Atom("B", i, False, False), None)
        >>> prop = m.Prop(m.PT.negation, m.Prop(m.PT.conjunction, a, b), None)
        >>> m.rule_backward_conjunctive_syllogism(prop, m.Prop(m.PT.negation, a, None))
        [B(x_0)]
        """

        if g.type == self.PT.negation and p.type == self.PT.negation:
            # g = NOT(A(x))
            if p.v1.type == self.PT.conjunction:
                # p = NOT(A(x) AND B(x))
                if self.matching(p.v1.v1, g.v1):
                    return [self.atom_prop_replace_properties(p.v1.v2, new_arg_id=g.v1.v1.arg_id,
                                                              new_is_name=g.v1.v1.is_name,
                                                              new_hat=g.v1.v1.hat)]
                elif self.matching(p.v1.v2, g.v1):
                    return [self.atom_prop_replace_properties(p.v1.v1, new_arg_id=g.v1.v1.arg_id,
                                                              new_is_name=g.v1.v1.is_name,
                                                              new_hat=g.v1.v1.hat)]
        return []

    def rule_backward_if_elimination(self, p, g):
        """
        >>> m = PSYCOP()
        >>> i = m.get_fresh_id()
        >>> a = m.Prop(m.PT.atomic, m.Atom("A", i, False, False), None)
        >>> b = m.Prop(m.PT.atomic, m.Atom("B", i, False, False), None)
        >>> m.rule_backward_if_elimination(m.Prop(m.PT.implies, a, b), b)
        [A(x_0)]
        """

        if p.type == self.PT.implies:
            # p = IF A(x) THEN B(x)
            if self.matching(p.v2, g):
                return [self.atom_prop_replace_properties(p.v1, new_arg_id=g.v1.arg_id,
                                                          new_is_name=g.v1.is_name,
                                                          new_hat=g.v1.hat)]
        return None

    def rule_backward_not_introduction(self, g):
        new_subgoals = []

        if g.type == self.PT.negation:
            if any(self.isomorphic(g.v1, s, True) for s in self.subformulas):
                for s in self.subformulas:
                    new_subgoals.append(
                        self.Prop(self.PT.conjunction, s, self.Prop(self.PT.negation, s, None)))
                new_subgoals = self.remove_duplicates(new_subgoals)
                return g.v1, new_subgoals
        return None, None

    def tentative_conclusion_mood(self, syllogism):
        if "E" in syllogism:
            return "E"
        elif "O" in syllogism:
            return "O"
        elif "I" in syllogism:
            return "I"
        return "A"

    def flatten(self, list):
        return [element for sublist in list for element in sublist]

    def apply_backward_rules(self, fw_propositions, g_node):
        g = g_node.goal
        new_subgoals = []
        by_which_rule = []
        matched_propositions = []
        suppositions = []
        for p in fw_propositions:
            if self.matching(p, g):
                matched_propositions.append(p)
                new_subgoals.append(p)
                by_which_rule.append("by-match")
                suppositions.append(None)

        if self.params["rule_bw_and_introduction"]:
            r = self.rule_backward_and_introduction(g)  # applies iff g = P AND Q
            if r:
                new_subgoals.extend(r)
                matched_propositions.extend([None] * len(r))
                by_which_rule.extend(["by-ai"] * len(r))
                suppositions.extend([None] * len(r))

        for p in fw_propositions:
            if self.params["rule_bw_conjunctive_syllogism"]:
                r = self.rule_backward_conjunctive_syllogism(p, g)  # applies iff g = NOT P
                if r:
                    new_subgoals.extend(r)
                    matched_propositions.extend([None] * len(r))
                    by_which_rule.extend(["by-cs"] * len(r))
                    suppositions.extend([None] * len(r))

            if self.params["rule_bw_if_elimination"]:
                r = self.rule_backward_if_elimination(p, g)  # applies iff p = IF P THEN g => g = A(x)
                if r:
                    new_subgoals.extend(r)
                    matched_propositions.extend([None] * len(r))
                    by_which_rule.extend(["by-ie"] * len(r))
                    suppositions.extend([None] * len(r))

        if not g_node.suppositions:
            if self.params["rule_bw_not_introduction"]:
                supposition, r = self.rule_backward_not_introduction(g)  # g = NOT P
                if r:
                    new_subgoals.extend(r)
                    matched_propositions.extend([None] * len(r))
                    by_which_rule.extend(["by-ni"] * len(r))
                    suppositions.extend([supposition] * len(r))

        return new_subgoals, matched_propositions, by_which_rule, suppositions

    def solve_disjunctive_tree(self, root_node, fw_propositions, right_conjunct=None):
        right_conjunct_alternatives = None
        if right_conjunct is not None:
            right_conjunct_alternatives = []

        root = AnyNode(goal=root_node.goal, exhausted=False, suppositions=root_node.suppositions)
        current_node = root
        branch_sat = False
        while True:
            if current_node.goal.type == self.PT.conjunction:
                current_node.exhausted = True
                if self.solve_conjunction_tree(current_node, fw_propositions):
                    branch_sat = True
            else:
                new_subgoals, matched_props, b, suppositions = self.apply_backward_rules(
                    fw_propositions + current_node.suppositions, current_node)
                current_node.exhausted = True
                for i, sg in enumerate(new_subgoals):
                    mp = matched_props[i]
                    supp = suppositions[i]
                    if mp == current_node.goal:
                        pa = self.get_leftmost_atom(mp)
                        if right_conjunct is not None:
                            right_conjunct_alternatives.append(
                                self.prop_replace_properties(right_conjunct.goal, pa.arg_id,
                                                             pa.is_name, pa.hat))
                        branch_sat = True
                    elif all(m != current_node.goal for m in matched_props):
                        if supp is None:
                            supp = []
                        else:
                            supp = [supp]
                        AnyNode(goal=sg, parent=current_node, exhausted=False,
                                suppositions=supp + current_node.suppositions)

            for c in LevelOrderIter(root):
                if not c.exhausted:
                    current_node = c
                    break
            if current_node.exhausted:
                return branch_sat, right_conjunct_alternatives

    def solve_conjunction_tree(self, conjunction_node, fw_propositions):
        root = AnyNode(goal=conjunction_node.goal, exhausted=True,
                       suppositions=conjunction_node.suppositions)
        current_node = root
        # the arguments of root.goal are either atomic or negation
        new_subgoals, matched_props, _, _ = self.apply_backward_rules(fw_propositions, current_node)
        if any(p is not None for p in matched_props):
            # direct match of the conjunction
            return True
        if len(new_subgoals) != 2:
            return False  # ?
        left_conjunct = AnyNode(goal=new_subgoals[0], parent=root, exhausted=False,
                                suppositions=root.suppositions)
        right_conjunct = AnyNode(goal=new_subgoals[1], parent=root, exhausted=True,
                                 suppositions=root.suppositions)

        left_branch_sat, conjunct2_alternatives = self.solve_disjunctive_tree(left_conjunct,
                                                                              fw_propositions,
                                                                              right_conjunct)
        if not left_branch_sat:
            return False
        for c in conjunct2_alternatives:
            alternative_node = AnyNode(goal=c, suppositions=root.suppositions)
            right_branch_sat, _ = self.solve_disjunctive_tree(alternative_node, fw_propositions)
            if right_branch_sat:
                return True
        return False

    def run_backward_rules(self, fw_propositions, conclusion):
        ret, _ = self.solve_disjunctive_tree(AnyNode(goal=conclusion, suppositions=[]),
                                             fw_propositions, None)
        return ret

    def remove_duplicates(self, propositions):
        """ Removes isomorphic propositions where both involve variables """
        propositions_copy = list(propositions)
        uniques = []
        while True:
            duplicates = []
            if len(propositions_copy) == 0:
                return uniques
            p1 = propositions_copy[0]
            for p2 in propositions_copy:
                if self.isomorphic(p1, p2):
                    if not (self.get_leftmost_atom(p1).is_name or self.get_leftmost_atom(
                            p2).is_name):
                        duplicates.append(p2)
            uniques.append(p1)
            propositions_copy.remove(p1)
            [propositions_copy.remove(x) for x in duplicates if x in propositions_copy]

    def run_forward_rules(self, fw_propositions):
        while True:
            new_propositions = []
            for p1 in fw_propositions:
                for p2 in fw_propositions:
                    if self.params["rule_fw_and_elimination"]:
                        new_propositions.extend(self.rule_forward_and_elimination(p1))
                    if self.params["rule_transitivity"]:
                        new_propositions.extend(self.rule_transitivity(p1, p2, fw_propositions))
                    if self.params["rule_exclusivity"]:
                        new_propositions.extend(self.rule_exclusivity(p1, p2, fw_propositions))
                    if self.params["rule_exclusivity"]:
                        new_propositions.extend(self.rule_conversion(p1, fw_propositions))
            if set(fw_propositions) == set(fw_propositions + new_propositions):
                # exhausted all possibilities: no more rules apply.
                break
            fw_propositions = sylutil.uniquify_keep_order(fw_propositions + new_propositions)

        return self.remove_duplicates(fw_propositions)

    def proposition_to_string(self, p):
        if p.type == self.PT.negation:
            if p.v1.type == self.PT.conjunction:
                if p.v1.v1.type == self.PT.atomic and p.v1.v2.type == self.PT.atomic:
                    if not p.v1.v1.v1.is_name and not p.v1.v2.v1.is_name:
                        return "E" + p.v1.v1.v1.predicate.lower() + p.v1.v2.v1.predicate.lower()
        elif p.type == self.PT.conjunction:
            if p.v1.type == self.PT.atomic:
                if p.v2.type == self.PT.atomic:
                    if p.v1.v1.is_name and p.v2.v1.is_name:
                        return "I" + p.v1.v1.predicate.lower() + p.v2.v1.predicate.lower()
                elif p.v2.type == self.PT.negation:
                    if p.v2.v1.type == self.PT.atomic:
                        if p.v1.v1.is_name and p.v2.v1.v1.is_name:
                            return "O" + p.v1.v1.predicate.lower() + p.v2.v1.v1.predicate.lower()
        elif p.type == self.PT.implies:
            if p.v1.type == self.PT.atomic and p.v2.type == self.PT.atomic:
                if not p.v1.v1.is_name and not p.v2.v1.is_name:
                    return "A" + p.v1.v1.predicate.lower() + p.v2.v1.predicate.lower()
        return None

    def extract_ac_conclusions(self, propositions):
        """
        >>> m = PSYCOP()
        >>> i = m.get_fresh_id()
        >>> a = m.Prop(m.PT.atomic, m.Atom("A", i, False, False), None)
        >>> b = m.Prop(m.PT.atomic, m.Atom("B", i, False, False), None)
        >>> c = m.Prop(m.PT.atomic, m.Atom("C", i, False, False), None)
        >>> p1 = m.Prop(m.PT.implies, a, b)
        >>> p2 = m.Prop(m.PT.implies, b, c)
        >>> p3 = m.Prop(m.PT.implies, a, c)
        >>> m.extract_ac_conclusions({p1, p2, p3})
        ['Aac']
        >>> m.extract_ac_conclusions({p1, p2})
        []
        """

        prop_ac = []
        for p in propositions:
            s = self.proposition_to_string(p)
            if s is not None:
                if {s[1], s[2]} == {"a", "c"}:
                    prop_ac.append(s)
        return prop_ac

    def extract_atomic_subformulas(self, p):
        if p.type == self.PT.atomic:
            return [p]
        elif p.type == self.PT.negation:
            return self.extract_atomic_subformulas(p.v1)
        else:
            return self.extract_atomic_subformulas(p.v1) + self.extract_atomic_subformulas(p.v2)

    def extract_all_atomic_subformulas(self, propositions):
        subformulas = []
        for p in propositions:
            subformulas.extend(self.extract_atomic_subformulas(p))
        return subformulas

    def heuristic(self, syllogism):
        return {"AA": "A",
                "AI": "I",
                "AE": "E",
                "AO": "O",
                "EI": "E",
                "EE": "E",
                "EO": "E",
                "II": "I",
                "IO": "O",
                "OO": "O",
                }[''.join(sorted(syllogism[:2]))]

    def conclusions_positive_checks(self, syllogism, additional_premises=[]):
        premises = self.encode_premises(syllogism,
                                        ex_implicatures=self.params["premise_implicatures_existential"],
                                        grice_implicatures=self.params["premise_implicatures_grice"])

        for p in additional_premises:
            premises.append(self.encode_proposition(p, True))

        # 1. Try to get conclusions by applying forward rules
        fw_propositions = self.run_forward_rules(premises)
        fw_conclusions = []
        for prop in fw_propositions:
            for c in ccobra.syllogistic.RESPONSES:
                conclusion = self.encode_proposition(c, hat=False)
                if self.proposition_to_string(conclusion) == self.proposition_to_string(prop):
                    fw_conclusions.append(c)

        checked_conclusions = fw_conclusions
        for concl in ccobra.syllogistic.RESPONSES:
            tc_enc = self.encode_proposition(concl, hat=False)

            self.subformulas = self.extract_all_atomic_subformulas(premises + [tc_enc])
            success = self.run_backward_rules(fw_propositions, tc_enc)
            if success:
                checked_conclusions.append(concl)

        checked_conclusions = checked_conclusions if len(checked_conclusions) != 0 else ["NVC"]
        return checked_conclusions

    def predict(self, syllogism):
        premises = self.encode_premises(syllogism,
                                        ex_implicatures=self.params["premise_implicatures_existential"],
                                        grice_implicatures=self.params["premise_implicatures_grice"])

        # 1. Try to get conclusions by applying forward rules
        fw_propositions = self.run_forward_rules(premises)
        fw_conclusions = []
        for prop in fw_propositions:
            for c in ccobra.syllogistic.RESPONSES:
                conclusion = self.encode_proposition(c, hat=False)
                if self.proposition_to_string(conclusion) == self.proposition_to_string(prop):
                    fw_conclusions.append(c)
        if len(fw_conclusions) != 0:
            return fw_conclusions

        ac = "ac" if random.random() < 0.5 else "ca"
        tentative_conclusion = self.heuristic(syllogism) + ac
        tc_enc = self.encode_proposition(tentative_conclusion, hat=False)

        self.subformulas = self.extract_all_atomic_subformulas(premises + [tc_enc])
        success = self.run_backward_rules(fw_propositions, tc_enc)
        if success:
            if self.params["conclusion_implicatures"]:
                c_impl = sylutil.add_implicatures([tentative_conclusion], True, True)[1]
                conclusion_impl = self.encode_proposition(c_impl, hat=False)
                self.subformulas = self.extract_all_atomic_subformulas(premises + [conclusion_impl])
                success_impl = self.run_backward_rules(fw_propositions, conclusion_impl)
                if success_impl:
                    return [tentative_conclusion]
            else:
                return [tentative_conclusion]

        if random.random() < self.params["guess"]:
            return ["Aac", "Aca", "Iac", "Ica", "Eac", "Eca", "Oac", "Oca"]
        return ["NVC"]
