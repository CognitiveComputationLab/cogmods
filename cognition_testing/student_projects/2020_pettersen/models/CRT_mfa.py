import numpy as np
import ccobra

# For short notation, a = false, b = true
a = ['A']
b = ['B']

class CRT_mfa(ccobra.CCobraModel):
    def __init__(self, name='MFA'):
        super(CRT_mfa, self).__init__(name, ["crt"], ["single-choice"])

        # self.mfa = [sequence, [#false, #true]]
        self.mfa = []
        self.count = 0


    def pre_train(self, dataset, **kwargs):
        # Iterate over subjects
        for subject_data in dataset:
            # Iterate over tasks
            for task_data in subject_data:

                # Assumes sequence starts from 0 incrementing to n
                seq = task_data['item'].sequence_number
                resp = task_data['response']
                len_mfa = len(self.mfa)

                # Create new mfa class if existing
                if len_mfa <= seq:
                    self.mfa.append([seq, [0, 0]])
                
                # Count the true/false frequency
                if resp == [a]:
                    self.mfa[seq][1][0] += 1
                if resp == [b]:
                    self.mfa[seq][1][1] += 1


    def predict(self, item,  **kwargs):
        seq = item.sequence_number
        
        if self.mfa[seq][1][0] < self.mfa[seq][1][1]:
            p_resp = str("B")
        else:
            p_resp = str("A")

        return p_resp


    def adapt(self, item, truth, **kwargs):
        # Update the MFA model
        seq = item.sequence_number

        # Count the true/false frequency
        if truth == a:
            self.mfa[seq][1][0] += 1
        if truth == b:
            self.mfa[seq][1][1] += 1

        # TO DELETE
        #self.count += 1
        #print("count2:", self.count)