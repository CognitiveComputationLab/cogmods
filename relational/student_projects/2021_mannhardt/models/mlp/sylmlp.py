import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import ccobra

import onehot
import TC

from onehot import task_to_string, create_input_output

Outputs = {
    "CAB" : 0,
    "BCA" : 1,
    "ACB" : 2,
    "CBA" : 3
}

OutputsRev = {
    0 : "CAB",
    1 : "BCA",
    2 : "ACB",
    3 : "CBA"
}


class SylMLP(nn.Module):
    def __init__(self):
        super(SylMLP, self).__init__()

        self.fc1 = nn.Linear(10, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MLPModel(ccobra.CCobraModel):
    def __init__(self, name='MLP'):
        super(MLPModel, self).__init__(name, ['syllogistic', 'spatial-relational'], ['single-choice', 'verify'])

        # Initialize the neural network
        self.net = SylMLP()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

        # General training properties
        self.n_epochs = 25
        self.n_epochs_adapt = 2
        self.batch_size = 8

    def pre_train(self, dataset, **kwargs):
        train_x = []
        train_y = []

        for subj_data in dataset:
            for task_data in subj_data:
                seq = task_data['item'].sequence_number
                r = task_to_string(task_data['response'])
                if seq != 3:
                    continue
                t = list()
                for prem in task_data['item'].task:
                    t.append(task_to_string(prem))
                t.append(task_to_string(task_data['item'].choices))
                inp, out = create_input_output(t, seq, r)
                if inp is None:
                    continue

                train_x.append(inp)
                train_y.append(out)

        self.train_x = torch.from_numpy(np.array(train_x)).float()
        self.train_y = torch.from_numpy(np.array(train_y)).float()

        self.train_network(self.train_x, self.train_y, self.batch_size, self.n_epochs, verbose=True)

    def train_network(self, train_x, train_y, batch_size, n_epochs, verbose=False):
        for epoch in range(n_epochs):
            start_time = time.time()

            # Shuffle the training data
            perm_idxs = np.random.permutation(np.arange(len(train_x)))
            train_x = train_x[perm_idxs]
            train_y = train_y[perm_idxs]

            # Batched training loop
            losses = []
            for batch_idx in range(len(train_x) // batch_size):
                start = batch_idx * batch_size
                end = start + batch_size

                epoch_x = train_x[start:end]
                epoch_y = train_y[start:end]

                self.optimizer.zero_grad()

                # Optimize
                outputs = self.net(epoch_x)
                loss = self.criterion(outputs, epoch_y)
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())

            # Print statistics
            if verbose:
                print('Epoch {} ({:.2f}s): {}'.format(
                    epoch + 1, time.time() - start_time, np.mean(losses)))

    def predict(self, item, **kwargs):
        seq = item.sequence_number
        if seq == 1:
            choice1 = item.choices[0][0][0]
            choice2 = item.choices[1][0][0]
            tc = TC.TransitiveClosure(item.task)
            tc.create_graph()
            if tc.is_valid_model(choice1):
                return choice1
            elif tc.is_valid_model(choice2):
                return choice2
        elif seq == 2:
            tc = TC.TransitiveClosure(model=item.task[0][0])
            counterFactCons = tc.check_counterfact(item.choices[0][0])
            return counterFactCons
        choice1 = item.choices[0][0][0]
        # Query the model
        t = list()
        for prem in item.task:
            t.append(task_to_string(prem))
        t.append(task_to_string(item.choices))
        inp, _ = create_input_output(t, seq)
        if inp is None:
            return choice1
        inp_tensor = torch.Tensor(inp).float()
        output = self.net(inp_tensor)

        # Return maximum response
        response = output.argmax().item()
        return OutputsRev[response]

    def adapt(self, item, truth, **kwargs):
        if item.sequence_number != 3:
            return
        r = task_to_string(truth)
        t = list()
        for prem in item.task:
            t.append(task_to_string(prem))
        t.append(task_to_string(item.choices))
        inp, out = create_input_output(t, item.sequence_number, r)
        if inp is None:
            return

        adapt_x = torch.Tensor(inp).reshape(1, -1).float()
        adapt_y = torch.Tensor(out).reshape(1, -1).float()

        self.train_network(adapt_x, adapt_y, 1, self.n_epochs_adapt, verbose=False)
