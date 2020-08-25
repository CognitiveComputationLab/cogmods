import numpy as np
import ccobra
from sklearn.naive_bayes import BernoulliNB

# For short notation, a = false, b = true
a = ['A']
b = ['B']
w = 0

class CRT_conditional(ccobra.CCobraModel):
    def __init__(self, name='Conditional'):
        super(CRT_conditional, self).__init__(name, ["crt"], ["single-choice"])

        # self.mfa = [sequence, [#false, #true]]
        self.mfa = []
        self.cond_data = [[],[],[],[]]


    def pre_train(self, dataset, **kwargs):
        # Iterate over subjects
        for subject_data in dataset:
            # Iterate over tasks
            for task_data in subject_data:

                # Assumes sequence starts from 0 incrementing to n
                seq = task_data['item'].sequence_number
                resp = task_data['response']


                # ***** MFA (System 1) *****:
                # Create new mfa class if existing
                len_mfa = len(self.mfa)
                if len_mfa <= seq:
                    self.mfa.append([seq, [0, 0]])
                
                # Count the true/false frequency
                if resp == [a]:
                    self.mfa[seq][1][0] += 1
                if resp == [b]:
                    self.mfa[seq][1][1] += 1

                # ***** Conditional (System 2) *****:
                # Feature variables
                # Only add the feats for for 1 dependent, after that it repeats
                if seq < 1:
                    tsk = task_data['item'].task
                    len_tsk = len(tsk)
                    enc_vars = [0]*len_tsk

                    for i in range(len_tsk):
                        if tsk[i] == b:
                            enc_vars[i] = 1
                    self.cond_data[0].append(enc_vars)

                # Dependent variables
                if resp == [a]:
                    self.cond_data[seq+1].append(0)
                if resp == [b]:
                    self.cond_data[seq+1].append(1)


    def predict(self, item,  **kwargs):
        seq = item.sequence_number
        tsk = item.task
        len_tsk = len(tsk)
        enc_vars = [0]*len_tsk
        x = self.cond_data[0]
        y = self.cond_data[seq+1]

        # Task info
        for i in range(len_tsk):
            if tsk[i] == b:
                enc_vars[i] = 1

        # Condtional
        # System 1
        temp_sum = self.mfa[seq][1][0] + self.mfa[seq][1][1]
        p_sys1 = self.mfa[seq][1][1] / temp_sum
        # System 2
        cond_model = BernoulliNB()
        cond_model.fit(x, y)
        p_sys2 = cond_model.predict_proba([enc_vars])[:,1][0]

        # Apply weighting
        p_val = w*p_sys1 + (1-w)*p_sys2

        if p_val < 0.5:
            p_resp = str("A")
        else:
            p_resp = str("B")
        
        return p_resp


    def adapt(self, item, truth, **kwargs):
        # Update each model
        seq = item.sequence_number
        tsk = item.task
        len_tsk = len(tsk)
        enc_vars = [0]*len_tsk

        # Count the true/false frequency
        if truth == [a]:
            # MFA:
            self.mfa[seq][1][0] += 1
            # Conditional, dependent:
            self.cond_data[seq+1].append(0)
        if truth == [b]:
            # MFA:
            self.mfa[seq][1][1] += 1
            # Conditional:
            self.cond_data[seq+1].append(1)

        # Conditional, features:
        for i in range(len_tsk):
            if tsk[i] == b:
                enc_vars[i] = 1
        self.cond_data[0].append(enc_vars)

        #print("Conditional:", self.cond_data)
        #print("Lens:", len(self.cond_data[0]),len(self.cond_data[1]),len(self.cond_data[2]),len(self.cond_data[3]))

