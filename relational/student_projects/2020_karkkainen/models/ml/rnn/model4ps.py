# RNN-model for predicting relational reasoning. 
# The model uses basic RNN-units.


import time

import collections

import numpy as np
from pandas.core.common import flatten
import ccobra

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms





class RNN(nn.Module):
    def __init__(self, input_size=16, hidden_size=64, output_size=1):
        super(RNN, self).__init__()
        
        self.n_layers = 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_size, self.n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state
        hidden = self.init_hidden(batch_size)

        out, hidden = self.rnn(x, hidden)

        #Reshaping output for the fully connected layer.
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros.
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return hidden



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
        

class RNNModel(ccobra.CCobraModel):
    def __init__(self, name='RNN', k=1):
        super(RNNModel, self).__init__(name, ["spatial-relational"], ["verify"])

        self.net = RNN()
        self.hidden = None


        self.n_epochs = 50

        self.optimizer = optim.Adam(self.net.parameters())
        self.loss = nn.BCELoss()

    def pre_train(self, dataset):
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        x = []
        y = []

        for subj_train_data in dataset:

            subj_x = []
            subj_y = []
            for seq_train_data in subj_train_data:
                task = seq_train_data['item'].task

                premises = encode(task)

                choices = encode(seq_train_data['item'].choices[0])

                inp = list(flatten(premises)) + list(flatten(choices))

                target = getTarget(seq_train_data['response'])
                subj_x.append(inp)

                subj_y.append(target)
 

            x.append(subj_x)
            y.append(subj_y)

    


        # Delete incomplete rows:
        x = list(filter(lambda i: len(i) == 48, x))
        y = list(filter(lambda i: len(i) == 48, y))


        x = np.array(x)
        y = np.array(y)
        self.train_x = torch.from_numpy(x).float()
        self.train_y = torch.from_numpy(y).float()

        self.train_network(self.train_x, self.train_y, self.n_epochs, verbose=True)



    def train_network(self, train_x, train_y, n_epochs, verbose=False):
        if verbose:
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


                inp = cur_x.view(-1, 1, 16)
                
                outputs, _ = self.net(inp)

                loss = self.loss(outputs, cur_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())

        if verbose:
            print('Epoch {}/{} ({:.2f}s): {:.4f} ({:.4f})'.format(
                epoch + 1, n_epochs, time.time() - start_time, np.mean(losses), np.std(losses)))

            accs = []
            for subj_idx in range(len(self.train_x)):
                pred, _ = self.net(self.train_x[subj_idx].view(-1,1,16))
                pred = pred.round()

                truth = self.train_y[subj_idx]


                acc = torch.mean((pred == truth).float()).item()
                accs.append(acc)

            print('   acc mean: {:.2f}'.format(np.mean(accs)))
            print('   acc std : {:.2f}'.format(np.std(accs)))


            self.net.eval()


    # Turns the prediction into an statement according if the given conclusion is perceived true or false.   
    def predict(self, item, **kwargs):
        task = item.task
        premises = encode(task)
        choices = encode(item.choices[0])
        x = torch.FloatTensor(list(flatten(premises)) + list(flatten(choices)))
        output, self.hidden = self.net(x.view(1, 1, -1))

        label = int(np.round(output.detach().numpy()[0][0]))

        self.prediction = output_mpping[label]
        return self.prediction
