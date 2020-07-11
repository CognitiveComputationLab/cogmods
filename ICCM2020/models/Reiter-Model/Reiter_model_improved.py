"""
Reiter model for the Suppression Task.
Based on Reiter Default Logic
incl Modus Tollens and affirmation of the consequence.

@author: Francine Wagner <wagner.francine94@gmail.com>
"""

import numpy as np
import ccobra
import random

class ReiterModelImproved(ccobra.CCobraModel):
    def __init__(self, name='ReiterModelImproved'):
        super(ReiterModelImproved,self).__init__(name, ["nonmonotonic"], ["single-choice"])
        self.counter = 0

    def predict(self, item, **kwargs):
        global given_answer
        given_answer = []
        W = []
        knowledge = item.task[0]
        fact = item.task[1]
        if ("Rarely" in fact or "Mostly" in fact):
            f = self.encode(fact[2])
            W.append(f)
        elif ("Not" in fact):
            f = "not " + self.encode(fact[2])
            W.append(f)
        else:
            f = self.encode(fact[1])
            W.append(f)

        pre = ""
        cons = ""
        a = 0
        for i in knowledge:
            # new default
            if (i == "Implies"):
                pre = self.encode(knowledge[a+2])
                cons = ""

            if (i == "Holds"):
                if (knowledge[a-1] != "Not" and knowledge[a-1] != "Implies"):
                    if (a+1) < len(knowledge):
                        cons = self.encode(knowledge[a+1])
                if (knowledge[a-1] == "Not" and knowledge[a-2] != "Implies"):
                    if (a+1) < len(knowledge):
                        cons = "not " + self.encode(knowledge[a+1])

            if (cons == "e" or cons == "l" or cons == "t" or cons == "o" or cons == "not e" or cons == "not l" or cons == "not t" or cons == "not o"):
                # library scenario
                neg_just = "not o"
            else:
                # alien scenario
                neg_just = ""

            for el in W:
                if el == pre and neg_just not in W and cons != "" and cons not in W:
                    W.append(cons)
                # print(self.counter)
                if self.counter >= 0:
                    if cons !="":
                        if el == cons and pre!= "" and pre not in W:
                            W.append(pre)

            a += 1
        choices = item.choices
        if self.encode(choices[0][0][1]) in W:
            given_answer = choices[0]
            return choices[0]
        else:
            given_answer = choices[1]
            return choices[1]

    def adapt(self, item, response, **kwargs):
        # print("Richtige antwort wäre", response)
        if response != given_answer:
            self.counter += 1

    def encode(self, sentence):
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
