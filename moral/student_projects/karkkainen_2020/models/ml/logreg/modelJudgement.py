# Logistic regression -based model for predicting moral reasoning. 

import collections

import numpy as np
from sklearn.linear_model import LogisticRegression

import ccobra


# Input mapping for the different dilemmas
dilemma_mppng =  {"Switch": [1,-5,5,-1,0,0], "Loop":[1,-5,5,-1,1,0], "Footbridge":[1,-5,5,-1,1,1]}

gender_mppng = {"Men":[0], "Women":[1]}

continent_mppng = {"Americas":[1,0,0,0], "Asia":[0,1,0,0], "Europe":[0,0,1,0], "Oc.":[0,0,0,1]}

education_mppng = {"No College":[0], "College":[1]}

def create_input(data):
    auxiliary = data['aux']

    dilemma = dilemma_mppng[data['task'][0][0]]
    gender = gender_mppng[auxiliary['survey.gender']]
    cont = continent_mppng[auxiliary['Continent']]
    edu = education_mppng[auxiliary['survey.education']]
    age = [auxiliary['survey.age']/100]
    politics = [auxiliary['survey.political']]
    religious = [auxiliary['survey.religious']]


    return (dilemma + gender + cont + edu + age + politics + religious)

class LogRegModel(ccobra.CCobraModel):
    def __init__(self, name='LogReg', k=1):
        super(LogRegModel, self).__init__(name, ["moral"], ["single-choice"])

        self.clf = LogisticRegression(C=0.01, penalty = 'l2')


        self.n_epochs = 1


    def pre_train(self, dataset):


        
        x = []
        y = []

        for subj_train_data in dataset:
            for seq_train_data in subj_train_data:
                
                seq_train_data['task'] = seq_train_data['item'].task
                inp = create_input(seq_train_data)

                target = float(seq_train_data['response'])

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



