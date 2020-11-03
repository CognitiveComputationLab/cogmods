""" Implementation of PHM.

"""

import collections

import ccobra
import numpy as np


class PHMNoAttachment(ccobra.CCobraModel):
    def __init__(self, name='PHMNoAttachment'):
        super(PHMNoAttachment, self).__init__(name, ['syllogistic-generalized'], ['single-choice'])

        # Initialize and declare member variables
        self.informativeness = ["A", "T", "B", "I", "E", "O"]
        self.p_entailments = {"A":["I"], "T":["I", "O"], "B":["I", "O"], "E":["O"], "I":["O"], "O":["I"]}

        self.pick_O = 0.0

    def pre_train(self, dataset, **kwargs):
        """ ---
    
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
        """ The PHM will not be person-trained.

        """

        pass

    def predict(self, item, **kwargs):
        """ Generate prediction based on the PHM.

        """
        
        conclusion_quantifiers = []

        ## Encode the task information
        syl = ccobra.syllogistic_generalized.GeneralizedSyllogism(item)
        task_enc = syl.encoded_task
        enc_choices = [syl.encode_response(x) for x in item.choices]
                
        ## Min-heuristic
        conclusion_quantifiers.append(self.min_heuristic(task_enc))

        ## P-entailment
        if conclusion_quantifiers[0] in self.p_entailments:
            conclusion_quantifiers.extend(self.p_entailment(conclusion_quantifiers))

        term_order = enc_choices[0][1:]
                
        pred = self.conclusion(conclusion_quantifiers, term_order)
        
        return syl.decode_response(pred)
        
    def adapt(self, item, truth, **kwargs):
        """ The PHM will not be adapted.

        """

        pass

    def min_heuristic(self, syl):
        """ Min-heuristic: The quantifier of the conclusion is the same as the one 
            in the least informative premise
            
        """
        
        quant1 = syl[0]
        quant2 = syl[1]

        quant = self.informativeness[max(self.informativeness.index(quant1), self.informativeness.index(quant2))]
        
        return quant
    
    def p_entailment(self, concl):
        """ p-entailment: The next most preferred conclusion will be the p-entailment
            of the conclusion predicted by the min-heuristic
        """
        
        return self.p_entailments[concl[0]]
            
    def conclusion(self, concl, term_order):
        """ Determine conclusion
        
        """
        
        conclusions = []
        for c in concl:
            conclusions.append(c + term_order)
            
        if conclusions[0][0] == "O" and np.random.rand() > self.pick_O:
            final = np.random.choice(conclusions[1:])
        else:
            final = conclusions[0]

        return final
