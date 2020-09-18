import numpy as np
import random
import ccobra

class WCS_noAb(ccobra.CCobraModel):
    def __init__(self, name='WCS_noAb'):
        super(WCS_noAb,self).__init__(name, ["nonmonotonic"], ["single-choice"])
       
    def wcs_predict(self, item, **kwargs):
        W = []
        # major premise (conditional)
        knowledge = item.task[0]
        # minor premise
        fact = item.task[1]
        
        rarmostbool = False

        if ("Rarely" in fact or "Mostly" in fact):
            f = self.encode(fact[2])
            W.append(f)
            rarmostbool = True
        elif ("Not" in fact):
            f = "not " + self.encode(fact[2])
            W.append(f)
        else:
            f = self.encode(fact[1])
            W.append(f)
        
        p = None
        q = None
        
        
        if len(knowledge) > 6 or rarmostbool == True:
            ab1 = True
        else:
            ab1 = False

        # MP
        if "e" in W or "f" in W:
            p = True
            q = p and not(ab1) 
        # DA
        elif "not e" in W or "not f" in W:
            p = False
            q = p and not(ab1)
        
        if "e" in W or "not e" in W:
            if q:
                W.append("l")
            else:
                W.append("not l")
        elif "f" in W or "not f" in W:
            if q:
                W.append("m")
            else:
                W.append("not m")
        
        choices = item.choices
        if self.encode(choices[0][0][1]) in W:
            return choices[0]
        else:
            return choices[1]
        
    def predict(self, item, **kwargs):
        global given_answer
        given_answer = self.wcs_predict(item)
        return given_answer
        
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
