"""
Ordinal Condition Function model for the Suppression Task.
Based on Belief Revision

@author: Francine Wagner <wagner.francine94@gmail.com>
"""

import collections

import numpy as np

import ccobra

class OCFModel(ccobra.CCobraModel):
    def __init__(self, name='OCFModel', k=1):
        super(OCFModel, self).__init__(name, ["nonmonotonic"], ["single-choice"])

    def predict(self, item, **kwargs):
        variables_fact = []
        variables = []
        kappa_w = {}
        knowledge = item.task[0]
        fact = item.task[1]
        variables = self.build_variables(knowledge)
        variables_fact = self.build_fact_var(fact)
        neg = False
        if "Not" in fact:
            neg = True
        for el in variables_fact:
            if el not in variables:
                variables.append(el)
        worlds = self.build_worlds(variables)
        worlds_dict = {}
        for i in range(len(worlds)):
            worlds_dict.update({"w"+str(i):worlds[i]})
        conditionals_incl_fact = self.encode_conditionals(knowledge) + [("T",variables_fact[0])]
        kappa_w.update(self.compute_kappa(worlds_dict,conditionals_incl_fact))
        plausible_worlds = []
        for w,kappa in kappa_w.items():
            if kappa == 0:
                plausible_worlds.append(w)
        choice_pos = self.encode(item.choices[0][0][1])
        choose = False
        p_w = plausible_worlds[:]
        for w in p_w:
            assignment = worlds_dict[w]
            if neg:
               if (variables_fact[0],1) in assignment:
                    plausible_worlds.remove(w)
            else:
               if (variables_fact[0],0) in assignment:
                    plausible_worlds.remove(w)
        for w in plausible_worlds:
            assignment = worlds_dict[w]
            if (choice_pos,1) in assignment:
                choose = True
            else:
                choose = False
        if choose:
            return item.choices[0]
        else:
            return item.choices[1]


    def compute_kappa(self, worlds_dict, conditionals):
        """
        computes the rank of all possible worlds given a list of conditionals.
        :param worlds_dict: all possible worlds and their assignments
        :param conditionals: list with conditionals of the form: [(A,B),(B,C)]
        :return:
        """
        kappa_w = {}
        # start all worlds has kappa(w) = 0, looks like: ([(f, 0), (m, 0)], 0)
        for w in worlds_dict:
            kappa_w.update({w:0})
        for condi in conditionals:
            kappa_AB = 0
            kappa_AnotB = 0
            kappa_notA = 0
            verifying_worlds = []
            falsifying_worlds = []
            inapplicable_worlds = []
            pre, cons = condi
            if pre == "T":
                for w,assignment in worlds_dict.items():
                    if (cons,1) in assignment:
                        verifying_worlds.append(w)
                    if (cons,0) in assignment:
                        falsifying_worlds.append(w)
            else:
                pass
                # Annahme: es gibt keine negative precondition!
                neg_cons = False
                if "not" in cons:
                    neg_cons = True
                    cons = cons[4:]
                for w,assignment in worlds_dict.items():
                    if neg_cons:
                        if (pre,1) in assignment and (cons,0) in assignment:
                            verifying_worlds.append(w)
                        if (pre,1) in assignment and (cons,1) in assignment:
                            falsifying_worlds.append(w)
                    else:
                        if (pre,1) in assignment and (cons,1) in assignment:
                            verifying_worlds.append(w)
                        if (pre,1) in assignment and (cons,0) in assignment:
                            falsifying_worlds.append(w)
                    if (pre,0) in assignment:
                        inapplicable_worlds.append(w)
            mini = 10000000000000000000 # smallest kappa = 0 for most plausible world, place holder
            for w in verifying_worlds:
                if kappa_w[w] <= mini:
                    mini = kappa_w[w]
            kappa_AB = mini
            mini = 10000000000000000000 # smallest kappa = 0 for most plausible world, place
            for w in falsifying_worlds:
                if kappa_w[w] <= mini:
                    mini = kappa_w[w]
            kappa_AnotB = mini
            mini = 10000000000000000000 # smallest kappa = 0 for most plausible world, place
            for w in inapplicable_worlds:
                if kappa_w[w] <= mini:
                    mini = kappa_w[w]
            kappa_notA = mini
            gamma_minus = 1 # werden die echt immer neu iniuialisiert?
            gamma_plus = -1
            while gamma_minus - gamma_plus <= kappa_AB - kappa_AnotB:
                gamma_minus += 1
                gamma_plus -= 1
            kappa0 = min(gamma_plus + kappa_AB, kappa_notA)
            for w in verifying_worlds:
                kappa_w[w] = kappa_w[w] - kappa0 + gamma_plus
            for w in falsifying_worlds:
                kappa_w[w] = kappa_w[w] - kappa0 + gamma_minus
            for w in inapplicable_worlds:
                kappa_w[w] = kappa_w[w] - kappa0
        return kappa_w

    def build_variables(self, knowledge):
        """
        Builds from knowledge (conditionals) logical variables
        :param knowledge: List with strings which contains the conditionals as a sentence
        :return: a list with variables (strings)
        """
        knowledge_var = []
        a = 0
        for i in knowledge:
            if (i == "Implies"):
                k = self.encode(knowledge[a + 2])
            if (i == "Holds"):
                if (knowledge != "Implies"): # hier wurde vorher unterschieden ob "Not" drin stand!
                    if (a + 1) < len(knowledge):
                        k = self.encode(knowledge[a + 1])
            a += 1
            if k not in knowledge_var:
                knowledge_var.append(k)
        return knowledge_var

    def build_fact_var(self, fact):
        """
        Builds from the fact (task) logical vairables (here only one)
        :param fact: List with fact as a string
        :return: a list with one single String --> logical variable
        """
        fact_var = []
        if ("Rarely" in fact or "Mostly" in fact or "Not" in fact):
            f = self.encode(fact[2])
        else:
            f = self.encode(fact[1])
        if f not in fact_var:
            fact_var.append(f)
        return fact_var

    def build_worlds(self,task):
        """
        Builds all possible assignment for the variables in task
        :param task: list containing all logical variables.
        :return: a list with lists(worlds) containg tuples (varible, assignemnt) pairs
        """
        w_0 = []
        all_worlds_list = []
        all_worlds_tuple = []
        insert = task[:]
        n = len(task)
        while len(w_0) != n:
            w_0.append(0)
        all_worlds_list.append(w_0)
        while True:
            w_x = self.bitshifting(all_worlds_list[-1])
            all_worlds_list.append(w_x)
            if len(all_worlds_list) == 2 ** n:
                break
        for element in all_worlds_list:
            new_world = []
            index = 0
            for assignment in element:
                new_world.append((insert[index],assignment))
                index += 1
            all_worlds_tuple.append(new_world)
        return all_worlds_tuple

    def bitshifting(self, bitlist):
        """
        bitwise shifts left one bit
        :param bitlist: a list with 0 or 1 integer
        :return:shifted list
        """
        n = len(bitlist) - 1
        index = 0
        found_zero = False
        new_bitlist = bitlist[:]
        if 0 not in new_bitlist:
            return new_bitlist
        while not found_zero:
            if new_bitlist[n - index] == 0:
                found_zero = True
                new_bitlist[n - index] = 1
            else:
                if n - index == 0:
                    break
                new_bitlist[n - index] = 0
                index += 1
        return new_bitlist

    def encode_conditionals(self, knowledge):
        """
        Encode conditionals into list with tupels
        :param knowledge: list with strings:
        Example:
        ['Implies', 'Holds', 'There is excess of food for her species', 'Holds', 'Kira will mate',
        'Implies', 'Holds', 'It is the 7th month of the solar year', 'Holds', 'Kira will mate',
        'Implies', 'Holds', 'The temperature falls below 10 Celsius', 'Not', 'Holds', 'Kira will mate']
        [('f', 'm'), ('s', 'm'), ('c', 'not m')]
        :return: List with tupels representing the conditionals
        """
        conditional = []
        a = 0
        for i in knowledge:
            if (i == "Implies"):
                pre = self.encode(knowledge[a + 2])
                cons = ""

            if (i == "Holds"):
                if (knowledge[a - 1] != "Not" and knowledge[a - 1] != "Implies"):
                    if (a + 1) < len(knowledge):
                        cons = self.encode(knowledge[a + 1])
                if (knowledge[a - 1] == "Not" and knowledge[a - 2] != "Implies"):
                    if (a + 1) < len(knowledge):
                        cons = "not " + self.encode(knowledge[a + 1])
            if (pre != "" and cons != "") and ((pre,cons) not in conditional):
                    conditional.append((pre, cons))
            a += 1
        return conditional


    def encode(self,sentence):
        """
        Encodes the sentence from the database into variables
        :param sentence: String from the database
        :return: single char String
        """
        # library
        if (sentence == "Lisa has an essay to finish"):
            return "e"
        elif (sentence == "She will study late in the library" or sentence == "Lisa will study late in the library"):
            return "l"
        elif (sentence == "She has some textbooks to read"):
            return "t"
        elif (sentence == "The library stays open"):
            return "o"
        # alien
        elif (sentence == "There is excess of food for her species" or sentence == "there is excess of food for her species" or sentence == "There is excess of food for Kira's species"):
            return "f"
        elif (sentence == "Kira will mate" or sentence == "Kira mated"):
            return "m"
        elif (sentence == "It is the 7th month of the solar year"):
            return "s"
        elif (sentence == "The temperature falls below 10 Celsius"):
            return "c"
