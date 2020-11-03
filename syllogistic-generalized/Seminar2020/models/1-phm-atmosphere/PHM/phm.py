""" Implementation of PHM.

"""

import collections

import ccobra
import numpy as np


class PHM(ccobra.CCobraModel):
    def __init__(self, name='PHM'):
        super(PHM, self).__init__(name, ['syllogistic-generalized'], ['single-choice'])

        # Initialize and declare member variables
        self.informativeness = ["A", "T", "I", "E", "D", "O"]
        self.p_entailments = {"A":["I"], "T":["I", "O"], "D":["I", "O"], "E":["O"], "I":["O"], "O":["I"]}

        ## Confidence of "Most...not" taken as "Few"
        self.confidence = {"A":1.99, "T":1.41, "D":1.02, "I":0.76, "E":0, "O":0.05}
        
        self.pick_O = 0.0

        ## For when quantifiers are the same
        self.figure_concl = {"1":0, "2":0, "3":0, "4":0}

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

        pred = self.conclusion(task_enc, conclusion_quantifiers)
        return syl.decode_response(pred)
        
    def adapt(self, item, truth, **kwargs):
        """ The PHM will not be adapted.

        """
        syl = ccobra.syllogistic_generalized.GeneralizedSyllogism(item)
        task_enc = syl.encoded_task
        true = syl.encode_response(truth)

        ## Adapt conclusion term order for when quantifiers are the same
        if task_enc[0] == task_enc[1]:
            ## If "ac" then +1, if "ca" then -1. When determining the order if the number is positive go with "ac"
            ## otherwise go with "ca". If 0, then random.
            if true[1:] == "ac":
                self.figure_concl[task_enc[2]] += 1
            else:
                self.figure_concl[task_enc[2]] -= 1
        
        ## Adapt confidence (max-heuristic)
        if true == "NVC":
            max_quant = self.informativeness[min(self.informativeness.index(task_enc[0]), self.informativeness.index(task_enc[1]))]
            self.confidence[max_quant] -= 0.2

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
    
    def attachment(self, syl, concl_quant):
        """ Attachemnt heuristic: If the min-premise has an end-term as a subject, use this as
            the subject of the conclusion. Otherwise, use the end term of the max-premise 
            as the subject of the conclusion. When both
            moods are the same, there is no preferred order in the conclusion.

        """
        
        quant1 = syl[0]
        quant2 = syl[1]
        fig = syl[2]
        
        if quant1 == quant2:
            if self.figure_concl[fig] > 0:
                return "ca"
            elif self.figure_concl[fig] < 0:
                return "ac"
            else:
                return np.random.choice(["ac", "ca"])
        
        if fig == "1":
            return "ac"
        elif fig == "2":
            return "ca"
        elif fig == "3":
            if quant1 == concl_quant:
                return "ac"
            else:
                return "ca"
        else:
            if quant1 == concl_quant:
                return "ca"
            else:
                return "ac"
            
    def max_heuristic(self, syl):
        """ The confidence in the generated conclusion is proportional to the informativeness of the max-premise.
        
        """
        quant1 = syl[0]
        quant2 = syl[1]
        
        max_quant = self.informativeness[min(self.informativeness.index(quant1), self.informativeness.index(quant2))]
        
        return self.confidence[max_quant]
            
    def conclusion(self, syl, concl):
        """ Determine conclusion
        
        """
        
        conclusions = []
        for c in concl:
            ## Attachment heuristic
            term_order = self.attachment(syl, c)
            conclusions.append(c + term_order)
        
        ## O-heuristic
        if conclusions[0][0] == "O" and np.random.rand() > self.pick_O:
            final = np.random.choice(conclusions[1:])
        else:
            ## Max-heuristic
            if np.random.random() < self.max_heuristic(syl)/4.0:
                final = conclusions[0]
            else:
                final = "NVC"

        return final
