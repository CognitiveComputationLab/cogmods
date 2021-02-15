# Logistic regression -based model for predicting moral reasoning. 

import collections

import numpy as np
from sklearn.linear_model import LogisticRegression

import ccobra


# Input mapping for the different dilemmas
task_mppng =  {"Railroad": [5,-1,0], "Fireman":[5,-1,1], 
                "Motorboat":[2,-1,0], "Ship":[3,-1,0],
                "Pregnancy":[3,-1,1], "Accident":[2,-1,1]}
                
task_mppngALT =  {"Railroad": [1,0,0,0,0], "Fireman":[0,1,0,0,0], 
                "Motorboat":[0,0,1,0,0], "Ship":[0,0,0,1,0],
                "Pregnancy":[0,0,0,0,1], "Accident":[0,0,0,0,0]} 

contact_mppng = {"NOCONTACT":[0], "CONTACT":[1]}

output_mppng = {"IMPERMISSIBLE": 0, "PERMISSIBLE": 1}

output_mppngREV = {0 : "IMPERMISSIBLE", 1 : "PERMISSIBLE"}

def create_input(data):
    auxiliary = data['aux']

    dilemma = task_mppng[data['task'][0][0]]
    contact = contact_mppng[data['task'][0][1]]

    return (dilemma + contact)

class LogRegModel(ccobra.CCobraModel):
    def __init__(self, name='LogReg', k=1):
        super(LogRegModel, self).__init__(name, ["moral"], ["single-choice"])

        self.clf = LogisticRegression(C=100, penalty = 'l2')


        self.n_epochs = 1


    def pre_train(self, dataset):


        
        x = []
        y = []

        for subj_train_data in dataset:
            for seq_train_data in subj_train_data:
                
                seq_train_data['task'] = seq_train_data['item'].task
                inp = create_input(seq_train_data)
                target = float(output_mppng[seq_train_data['response'][0][0]])

                x.append(inp)

                y.append(target)
        x = np.array(x)
        y = np.array(y)

        self.train_x = x
        self.train_y = y


        self.train_network(self.train_x, self.train_y, self.n_epochs, verbose=True)



    def train_network(self, train_x, train_y, n_epochs, verbose=False):
            print('Starting training...')
            for epoch in range(self.n_epochs):
                
                # Shuffle the training data
                perm_idxs = np.random.permutation(np.arange(len(train_x)))
                train_x = train_x[perm_idxs]
                train_y = train_y[perm_idxs]

                self.clf.fit(train_x,train_y) 

                print('Mean accuracy:')
                print(self.clf.score(train_x, train_y))

    # Turns the predicted, one-hot encoded output into class-label, which is further turned into a cardinal direction.      
    def predict(self, item, **kwargs):
        input = {'task': item.task}
        input['aux'] = kwargs
        x = np.array(create_input(input)).reshape(1, -1)
        output = self.clf.predict(x)

        self.prediction = output_mppngREV[output[0]]

        return self.prediction



