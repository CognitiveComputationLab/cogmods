# Auto-ML for predicting relational reasoning. 
# The model is based on the auto-sklearn library.


import time

import collections

import numpy as np
from pandas.core.common import flatten

import ccobra

import autosklearn.classification



def getObjctMapping(task):
    objcts = []
    for i in task:
        objcts.append(i[1])
        objcts.append(i[2])
    objcts =  [*{*objcts}]

    return dict(zip( objcts, list(range(len(objcts)))))

output_mpping = {1: True, 0: False}

def encode(task, mpping):
    result = []
    for i in task:
        premise = [0] * 5
        premise[mpping[i[1]]] = -1
        premise[mpping[i[2]]] = 1
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

                objct_mpping = getObjctMapping(task)

                premises = encode(task, objct_mpping)

                choices = encode(seq_train_data['item'].choices[0], objct_mpping)

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
        train_y = train_y[perm_idxs]

        train_y = np.ravel(train_y)

       
        self.automl.fit(train_x, train_y)




    # Turns the predicted, one-hot encoded output into class-label, which is further turned into a cardinal direction.      
    def predict(self, item, **kwargs):
        task = item.task
        objct_mpping = getObjctMapping(task)
        premises = encode(task, objct_mpping)
        choices = encode(item.choices[0], objct_mpping)
        x = np.array(list(flatten(premises)) + list(flatten(choices)))
        output = self.automl.predict(x.reshape(1, -1))


        label = output[0]

        self.prediction= output_mpping[label]

        return self.prediction
