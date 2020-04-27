# Auto-ML for predicting relational reasoning. 
# The model is based on the auto-sklearn library.


import time

import collections

import numpy as np
from pandas.core.common import flatten

import ccobra

import autosklearn.classification




objct_mpping = {   "A": 0,
                    "B":1,
                    "C": 2,
                    "D": 3, 
                }

output_mpping = {1: True, 0: False}

def getValues(rel):
    if rel == 'Left':
        return [-1.0, 1.0]
    else:
        return [1.0, -1.0]


def encode(task):
    result = []
    for i in task:
        premise = [0] * 4
        val = getValues(i[0]) 
        premise[objct_mpping[i[1]]] = val[0]
        premise[objct_mpping[i[2]]] = val[1]
        result.append(premise)
    return result

def getTarget(targ):
    if targ:
        return [1]
    else: 
        return [0]
        


class AutoMLModel(ccobra.CCobraModel):
    def __init__(self, name='AutoML', k=1):
        super(AutoMLModel, self).__init__(name, ["spatial-relational"], ["verify"])

        self.automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=30, per_run_time_limit=5)



        self.n_epochs = 1



    def pre_train(self, dataset):
        
        x = []
        y = []

        for subj_train_data in dataset:
            for seq_train_data in subj_train_data:
                task = seq_train_data['item'].task

                premises = encode(task)

                choices = encode(seq_train_data['item'].choices[0])

                inp = list(flatten(premises)) + list(flatten(choices))

                target = getTarget(seq_train_data['response'])
                x.append(inp)


                y.append(target)
        
        self.train_x = np.array(x)
        self.train_y = np.array(y)

        self.train_network(self.train_x, self.train_y, self.n_epochs, verbose=True)



    def train_network(self, train_x, train_y, n_epochs, verbose=False):
        print('Starting training...')


        # Shuffle the training data
        perm_idxs = np.random.permutation(np.arange(len(train_x)))
        train_x = train_x[perm_idxs]
        train_y = np.ravel(train_y[perm_idxs])

       
        self.automl.fit(train_x, train_y)


    # Returns to prediction if the conclusions is 'True' or 'False'.
    def predict(self, item, **kwargs):
        task = item.task
        premises = encode(task)
        choices = encode(item.choices[0])
        x = np.array(list(flatten(premises)) + list(flatten(choices)))
        output = self.automl.predict(x.reshape(1, -1))


        label = output[0]

        self.prediction= output_mpping[label]

        return self.prediction
