# Auto-ML for predicting relational reasoning. 
# The model is based on the auto-sklearn library.


import time

import collections

import numpy as np

import ccobra

import autosklearn.classification




# mapping of cardinal direction input
input_mppng =  {"north-west": [1,0,0,1], "north":[1,0,0,0], "north-east":[1,1,0,0],
                "west": [0,0,0,1], "east":[0,1,0,0],
                "south-west": [0,0,1,1], "south":[0,0,1,0], "south-east":[0,1,1,0]}

# mapping of cardinal direction output
output_mppng = {"north-west": [1,0,0,0,0,0,0,0], "north":[0,1,0,0,0,0,0,0], "north-east":[0,0,1,0,0,0,0,0],
                "west": [0,0,0,1,0,0,0,0], "east":[0,0,0,0,1,0,0,0],
                "south-west": [0,0,0,0,0,1,0,0], "south":[0,0,0,0,0,0,1,0], "south-east":[0,0,0,0,0,0,0,1]}

# Reverse mapping of turning a class label into a cardinal direction.
output_mpp_reverse = {0:"north-west", 1:"north", 2: "north-east",
                3:"west", 4:"east",
                5:"south-west", 6:"south", 7:"south-east"}




class AutoMLModel(ccobra.CCobraModel):
    def __init__(self, name='AutoML', k=1):
        super(AutoMLModel, self).__init__(name, ["spatial-relational"], ["single-choice"])

        self.automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=40, per_run_time_limit=8)



        self.n_epochs = 1



    def pre_train(self, dataset):
        
        x = []
        y = []

        for subj_train_data in dataset:
            for seq_train_data in subj_train_data:
                task = seq_train_data['item'].task

                inp = input_mppng[task[0][0]] + input_mppng[task[1][0]]
                target = output_mppng[seq_train_data['response'][0][0]]

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
        train_y = np.apply_along_axis(np.argmax,1,train_y)

        self.automl.fit(train_x, train_y)




    # Turns the predicted, one-hot encoded output into class-label, which is further turned into a cardinal direction.      
    def predict(self, item, **kwargs):
        task = item.task
        x = np.array(input_mppng[task[0][0]] + input_mppng[task[1][0]])
        output = self.automl.predict(x.reshape(1, -1))
        label = output[0]

        self.prediction= [output_mpp_reverse[label], task[-1][-1], task[0][1]]
        return self.prediction
