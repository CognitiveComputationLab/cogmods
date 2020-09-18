import numpy as np
import random
import ccobra

class WCSPretrain(ccobra.CCobraModel):
    def __init__(self, name='WCSPretrain'):
        super(WCSPretrain,self).__init__(name, ["nonmonotonic"], ["single-choice"])
    
        # Set to 1 -- always do abduction, uncomment below if pretraining is desired
        self.abduction_prob = 1       
        self.abnormality_prob_BC = 0
        self.abnormality_prob_D = 0
    
    def pre_train(self, dataset):
        abduct = []
        abnorm = []
        abnormD = []
        for subj_train_data in dataset:
            for seq_train_data in subj_train_data:
                count = 0
                total_count = 0.0000001
                group_count = 0
                total_group_count = 0.0000001
                group_count_D = 0
                total_group_count_D = 0.0000001
                item = seq_train_data['item']
                predicted = self.wcs_predict(item)
                if seq_train_data['inffig'] == 'MT' or seq_train_data['inffig'] == 'AC':
                    if predicted != seq_train_data['response']:
                        count += 1
                    total_count += 1
                if 'A' not in seq_train_data['Group']:
                    if 'B' in seq_train_data['Group'] or 'C' in seq_train_data['Group']:
                        if predicted != seq_train_data['response']:
                            group_count += 1
                        total_group_count += 1
                    else:
                        if predicted != seq_train_data['response']:
                            group_count_D += 1
                        total_group_count_D += 1
                abduct.append(count/total_count)
                abnorm.append(group_count/total_group_count)
                abnormD.append(group_count_D/total_group_count_D)
        self.abnormality_prob_BC = np.mean(abnorm)
        self.abnormality_prob_D = np.mean(abnormD)

        # Uncomment this if you want to pretrain the abduction probability
        # self.abduction_prob = np.mean(abduct)
                        
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
        
        
        if len(knowledge) > 6:
            if random.random() <= self.abnormality_prob_D:
                ab1 = True
            else:
                ab1 = False
        elif rarmostbool == True:
            if random.random() <= self.abnormality_prob_BC:
                ab1 = True
            else:
                ab1 = False
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
    
        # AC
        if "l" in W or "m" in W:
            # abduction
            if random.random() <= self.abduction_prob:
                q = True
        #MT
        elif "not l" in W or "not m" in W:
            # abduction
            if random.random() <= self.abduction_prob:
                q = False
        
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
        elif "l" in W or "not l" in W:
            if q:
                W.append("e")
            else:
                W.append("not e")
        elif "m" in W or "not m" in W:
            if q:
                W.append("f")
            else:
                W.append("not f")
        
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
