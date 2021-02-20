# A neural network -based model for predicting moral reasoning. 



import time

import collections

import numpy as np

import ccobra

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms



class MLP(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, output_size=1):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        
        hidden = self.fc1(x)
        relu1 = self.relu(hidden)
        output = self.fc2(relu1)
        output = self.sigmoid(output)
        return output

# Input mapping for the different dilemmas
task_mppng =  {"Railroad": [5,-1,0], "Fireman":[5,-1,1], 
                "Motorboat":[2,-1,0], "Ship":[3,-1,0],
                "Pregnancy":[3,-1,1], "Accident":[2,-1,1]}

contact_mppng = {"NOCONTACT":[0], "CONTACT":[1]}

output_mppng = {"IMPERMISSIBLE": 0, "PERMISSIBLE": 1}

output_mppngREV = {0 : "IMPERMISSIBLE", 1 : "PERMISSIBLE"}

def create_input(data):
    auxiliary = data['aux']

    dilemma = task_mppng[data['task'][0][0]]
    contact = contact_mppng[data['task'][0][1]]

    return (dilemma + contact)

class MLPModel(ccobra.CCobraModel):
    def __init__(self, name='MLP', k=1):
        super(MLPModel, self).__init__(name, ["moral"], ["single-choice"])

        self.net = MLP()


        self.n_epochs = 75

        self.optimizer = optim.SGD(self.net.parameters(), lr= 0.1)
        self.loss = nn.BCELoss()

    def pre_train(self, dataset):
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        x = []
        y = []

        for subj_train_data in dataset:
            subj_x = []
            subj_y = []
            for seq_train_data in subj_train_data:
                
                seq_train_data['task'] = seq_train_data['item'].task
                inp = create_input(seq_train_data)

                target = float(output_mppng[seq_train_data['response'][0][0]])


                subj_x.append(inp)

                subj_y.append(target)

            x.append(subj_x)
            y.append(subj_y)
        x = np.array(x)
        y = np.array(y)


        self.train_x = torch.from_numpy(x).float()
        self.train_y = torch.from_numpy(y).float()


        self.train_network(self.train_x, self.train_y, self.n_epochs, verbose=True)



    def train_network(self, train_x, train_y, n_epochs, verbose=False):
            print('Starting training...')
            for epoch in range(self.n_epochs):
                start_time = time.time()

                # Shuffle the training data
                perm_idxs = np.random.permutation(np.arange(len(train_x)))
                train_x = train_x[perm_idxs]
                train_y = train_y[perm_idxs]


                losses = []
                for idx in range(len(train_x)):
                    cur_x = train_x[idx]
                    cur_y = train_y[idx]

                    inp = cur_x.view(1,-1,4)

                    outputs = self.net(inp)


                    loss = self.loss(outputs, cur_y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    losses.append(loss.item())

            print('Epoch {}/{} ({:.2f}s): {:.4f} ({:.4f})'.format(
                epoch + 1, n_epochs, time.time() - start_time, np.mean(losses), np.std(losses)))


            accs = []
            for subj_idx in range(len(self.train_x)):
                pred = torch.round(self.net(self.train_x[subj_idx]))

                truth = self.train_y[subj_idx]

                acc = torch.mean((pred == truth).float()).item()
                accs.append(acc)

            print('   acc mean: {:.2f}'.format(np.mean(accs)))
            print('   acc std : {:.2f}'.format(np.std(accs)))


            self.net.eval()


    
    def predict(self, item, **kwargs):
        input = {'task': item.task}
        input['aux'] = kwargs
        x = torch.FloatTensor(create_input(input))
        output = self.net(x.view(1, 1, -1))

        label = np.round(output.detach().numpy())

        self.prediction = output_mppngREV[label[0][0][0]]
        return self.prediction



