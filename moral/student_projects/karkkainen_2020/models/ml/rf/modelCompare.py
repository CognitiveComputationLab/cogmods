# Random forests -based model for predicting moral reasoning. 
import collections

import numpy as np

import ccobra
from sklearn.ensemble import RandomForestClassifier

# Input mapping for the different dilemmas
dilemma_mppng =  {"PW-RT": [1,-3,3,-1,1,-5,5,-1], "OB-PW":[0,-4,3,-1,1,-3,3,-1], "RT-OB":[1,-5,5,-1,0,-4,3,-1]}

gender_mppng = {"m": [0], "w": [1]}

output_mppng = {1: [1,0,0], 2: [0,1,0], 3:[0,0,1]}

def create_input(data):
    auxiliary = data['aux']

    dilemma = dilemma_mppng[data['task'][0][0]]
    gender = gender_mppng[auxiliary['Geschlecht']]
    logical = [auxiliary['Logikerfahrung']/5]
    maths = [auxiliary['Mathekenntnisse']/5]
    age = [auxiliary['Alter']/100]
    return dilemma + gender + logical + maths + age


class RFModel(ccobra.CCobraModel):
    def __init__(self, name='RF', k=1):
        super(RFModel, self).__init__(name, ["moral"], ["single-choice"])

        self.clf = RandomForestClassifier(n_estimators= 1200, min_samples_split= 5, min_samples_leaf= 4, max_features= 'auto', max_depth= 80, bootstrap= True)


        self.n_epochs = 1




    def pre_train(self, dataset):


        
        x = []
        y = []

        for subj_train_data in dataset:
            for seq_train_data in subj_train_data:
                seq_train_data['task'] = seq_train_data['item'].task
                inp = create_input(seq_train_data)

                target = seq_train_data['response']

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

        self.prediction = output[0]
        return self.prediction

