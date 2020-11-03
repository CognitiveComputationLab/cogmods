""" Implementation of the model based on the Atmosphere Hypothesis.

    Quantifiers:
    A - All
    E - None
    I - Some
    O - Some ... not
    T - Most
    D - Most ... not
    B - Few
    G - Few ... not
    
"""

import collections

import ccobra
import numpy as np


class Atmosphere(ccobra.CCobraModel):
    def __init__(self, name='Atmosphere'):
        super(Atmosphere, self).__init__(name, ['syllogistic-generalized'], ['single-choice'])

        # Initialize and declare member variables
        self.representation = {"A":["++", "+"], "E":["++", "-"], "I":["--", "+"], "O":["--", "-"], "T":["+", "+"], "D":["+", "-"], "B":["-", "+"], "G":["-", "-"]}
        self.quantity_order = ["++", "+", "-", "--"]
        self.figure_concl = {"1":0, "2":0, "3":0, "4":0}
        self.figure_nvc = {"1":0, "2":0, "3":0, "4":0}
        self.figure_counter = {"1":0, "2":0, "3":0, "4":0}

    def pre_train(self, dataset, **kwargs):
        """ The Atmosphere model is not pre-trained.
    
        """

        pass

    def pre_train_person(self, dataset, **kwargs):
        """ The Atmosphere model is not pre-trained per person.

        """
        
        pass
        

    def predict(self, item, **kwargs):
        """ Generate prediction based on the Atmosphere Hypothesis.

        """
        ## Encode the task information
        syl = ccobra.syllogistic_generalized.GeneralizedSyllogism(item)
        task_enc = syl.encoded_task
        enc_choices = [syl.encode_response(x) for x in item.choices]
                
        pred = self.get_answer(task_enc)
        
        return syl.decode_response(pred)
        
    def adapt(self, item, truth, **kwargs):
        """ The Atmosphere model cannot adapt.

        """
        syl = ccobra.syllogistic_generalized.GeneralizedSyllogism(item)
        task_enc = syl.encoded_task
        true = syl.encode_response(truth)

        ## Count the occurrences of figures for the individual, to determine the NVC ratio
        ## for each figure
        self.figure_counter[task_enc[2]] += 1
        
        if true == "NVC":
            self.figure_nvc[task_enc[2]] += 1
        ## If "ac" then +1, if "ca" then -1. When determining the order if the number is positive go with "ac"
        ## otherwise go with "ca". If 0, then random.
        elif true[1:] == "ac":
            self.figure_concl[task_enc[2]] += 1
        else:
            self.figure_concl[task_enc[2]] -= 1
                    
    def get_representation(self, syl):
        """ Given a syllogism, return the representation of its premises based on the Atmosphere Hypothesis.
        
        """
        
        first = syl[0]
        second = syl[1]
        
        return [self.representation[first], self.representation[second]]
    
    def determine_quantifier(self, rep):
        """ Given a syllogism, give the quantifier of the conclusion based on the Atmosphere Hypothesis.
        
        """
                
        quant1 = rep[0][0]
        quant2 = rep[1][0]
        
        quant = self.quantity_order[max(self.quantity_order.index(quant1), self.quantity_order.index(quant2))]

        pol1 = rep[0][1]
        pol2 = rep[1][1]
        
        if pol1 != pol2:
            pol = "-"
        else:
            pol = pol1
            
        ## Replace Few with Most...not
        if quant == "-" and pol == "+":
            quant = "+"
            pol = "-"
        
        ## Replace Few...not with Most
        if quant == "-" and pol == "-":
            quant = "+"
            pol = "+"
    
        return [quant, pol]
    
    def get_answer(self, syl):
        """ Given a syllogism, return an answer

        """
                
        if (self.figure_counter[syl[2]] != 0) and (np.random.random() < self.figure_nvc[syl[2]] / self.figure_counter[syl[2]]):
            answer = "NVC"
        else:
            rep = self.get_representation(syl)
            concl_quant = self.determine_quantifier(rep)
            quant = list(self.representation.keys())[list(self.representation.values()).index(concl_quant)]
            if self.figure_concl[syl[2]] > 0:
                order = "ac"
            elif self.figure_concl[syl[2]] < 0:
                order = "ca"
            else:
                order = np.random.choice(["ac", "ca"])
            answer = quant + order
                
        return answer
