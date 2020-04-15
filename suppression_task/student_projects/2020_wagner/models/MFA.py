import collections

import numpy as np

import ccobra

class MFAModel(ccobra.CCobraModel):
    def __init__(self, name='MFAModel', k=1):
        super(MFAModel, self).__init__(name, ["nonmonotonic"], ["single-choice"])

        # Parameters
        self.k = k
        self.predictions = {}

    def pre_train(self, dataset):
        for subj_train_data in dataset:
            for seq_train_data in subj_train_data:
                task = seq_train_data['item'].task
                choices = seq_train_data['item'].choices
                c = []
                c += choices[0] + choices[1]
                task = task[0] + task[1]
                s = ''
                for el in task:
                    s += el
                if s not in self.predictions:
                    # choices = seq_train_data['choices']
                    c0 = c[0][0] + c[0][1]
                    c1 = c[1][0] + c[1][1]
                    di = {c0 : 0, c1 : 0}
                    self.predictions[s] = di
                self.adapt(seq_train_data['item'], seq_train_data['response'])

    def person_train(self, dataset):
        self.pre_train([dataset])

    def predict(self, item, **kwargs):
        task = item.task
        task = task[0] + task[1]
        s = ''
        for el in task:
            s += el
        resp_counts = self.predictions[s]

        target_value = sorted(
            np.unique(list(resp_counts.values())), reverse=True)[self.k - 1]
        resps = []
        for resp, cnt in resp_counts.items():
            if cnt == target_value:
                resps.append(resp)
        if(resps[np.random.randint(0, len(resps))] == "NotHolds"):
            return item.choices[1]
        else:
            return item.choices[0]
        # return resps[np.random.randint(0, len(resps))]

    def adapt(self, item, response, **kwargs):
        task = item.task
        task = task[0] + task[1]
        s = ''
        for el in task:
            s += el
        response = response[0][0]+response[0][1]
        self.predictions[s][response] += 1
