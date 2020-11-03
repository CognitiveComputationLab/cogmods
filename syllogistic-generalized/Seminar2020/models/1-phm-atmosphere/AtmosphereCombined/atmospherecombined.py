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


class AtmosphereCombinedModel(ccobra.CCobraModel):
    def __init__(self, name='AtmosphereCombined'):
        super(AtmosphereCombinedModel, self).__init__(name, ['syllogistic-generalized'], ['single-choice'])

        # Initialize and declare member variables
        self.representation = {"A":["++", "+"], "E":["++", "-"], "I":["--", "+"], "O":["--", "-"], "T":["+", "+"], "D":["+", "-"], "B":["-", "+"], "G":["-", "-"]}
        self.quantity_order = ["++", "+", "-", "--"]
        
        self.pick_O = 0

    def pre_train(self, dataset, **kwargs):
        """ The Atmosphere model is not pre-trained.
    
        """
        
        picked_O = 0
        total = 0
        
        # Iterate over subjects
        for subject_data in dataset:
            # Iterate over tasks
            for task_data in subject_data:
                # Encode the task
                syl = ccobra.syllogistic_generalized.GeneralizedSyllogism(task_data['item'])
                enc_task = syl.encoded_task
                enc_resp = syl.encode_response(task_data['response'])
                if enc_resp[0] == "O":
                    picked_O += 1

                total += 1
                
        self.pick_O = picked_O / total   

    def pre_train_person(self, dataset, **kwargs):
        """ The Atmosphere model is not supposed to be person-trained.

        """

        pass

    def predict(self, item, **kwargs):
        """ Generate prediction based on the Atmosphere Hypothesis.

        """

        ## Encode the task information
        syl = ccobra.syllogistic_generalized.GeneralizedSyllogism(item)
        task_enc = syl.encoded_task
        enc_choices = [syl.encode_response(x) for x in item.choices]
        
        ## Get the Atmosphere representation of the premises
        rep = self.get_representation(task_enc)
        
        ## Get the order of the terms in the conclusion
        term_order = enc_choices[0][1:]
        
        ## Determine the mood of the conclusion
        concl_quant = self.determine_answer(rep)
                
        ## Encode the conclusion
        pred = self.encode_representation(concl_quant, term_order)
        
        return syl.decode_response(pred)
        
    def adapt(self, item, truth, **kwargs):
        """ The Atmosphere model cannot adapt.

        """

        pass

    def get_representation(self, syl):
        """ Given a syllogism, return the representation of its premises based on the Atmosphere Hypothesis.
        
        """
        
        first = syl[0]
        second = syl[1]
        
        return [self.representation[first], self.representation[second]]
    
    def determine_answer(self, rep):
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
            
        ## Replace Most...not either with Few or with Some...not
        if quant == "+" and pol == "-":
            if np.random.rand() < self.pick_O:
                quant = "--"
                pol = "-"
            else:
                quant = "-"
                pol = "+"
        
        ## Replace Few...not with Most
        if quant == "-" and pol == "-":
            if np.random.rand() < self.pick_O:
                quant = "--"
                pol = "-"
            else:
                quant = "+"
                pol = "+"
            
            
        return [quant, pol]
    
    def encode_representation(self, rep, order):
        """ Given a representation, and conclusion terms order, based on Atmosphere, return the encoded syllogism.

        """
        quant = list(self.representation.keys())[list(self.representation.values()).index(rep)]
        answer = quant + order
                
        return answer
